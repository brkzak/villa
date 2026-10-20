"""Plain-PyTorch versions of lasagna's uint8 sampling kernels, for devices without the CUDA extensions.

The streaming fit (``fit_data.load_3d_streaming``) reads the prediction through a sparse chunk cache whose three
CUDA kernels (``sparse_prefetch_chunks_kernel.cu``, ``sparse_grid_sample_3d_u8_kernel.cu`` and
``sparse_grid_sample_3d_u8_diff_kernel.cu``) store raw device pointers in the chunk table and JIT-compile with nvcc.
Neither works on Apple MPS (Metal buffers have no such pointers), on CPU, or on ROCm without a HIP compiler.

This module computes the same values with ordinary tensor ops on any device. Chunks live in one pooled tensor
(``ChunkPool``) and the chunk table holds ``slot + 1`` instead of a pointer, so 0 still means "not loaded" and the
cache's existing checks (``chunk_table != 0``) keep working. Each function mirrors one kernel line for line: same
chunk size and padding, same NaN/out-of-bounds handling, same corner order and rounding. Values can still differ
from the CUDA kernels in the last float bits (nvcc fuses multiply-adds), not more.

The dense ``grid_sample_3d_u8_diff`` kernel (the ``cyl_outside`` loss) is mirrored too
(``dense_grid_sample_3d_u8_diff``).

``use_torch_sampler`` decides which path a device takes; ``LASAGNA_SPARSE_SAMPLER`` (``auto``, ``torch``, ``cuda``)
overrides it, e.g. ``torch`` on an NVIDIA GPU to compare both paths, or ``cuda`` on ROCm when a HIP compiler is
installed and the hipified kernels are wanted.
"""
from __future__ import annotations

import os

import torch

CHUNK_SIZE = 32
PADDED = CHUNK_SIZE + 2  # 1-voxel margin on each side, as the caches write chunks
_CHUNK_VOXELS = PADDED ** 3
_MAX_CHUNK_INDEX = 1 << 30  # clamp before the int cast: huge finite coords would overflow int64

# Offsets of the 8 trilinear corners inside one padded chunk, in the kernels' order
# (w000, w100, w010, w110, w001, w101, w011, w111; the digits are x, y, z).
_CORNERS_XYZ = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))


def use_torch_sampler(device: torch.device | str) -> bool:
    """Whether the cache on ``device`` samples with this module instead of the CUDA extensions.

    ``auto`` (the default) uses the CUDA kernels only on NVIDIA CUDA. ROCm builds of torch also report
    ``device.type == "cuda"``, but compiling the kernels there needs a HIP compiler that a torch wheel does not
    ship, so ROCm takes the torch path unless ``LASAGNA_SPARSE_SAMPLER=cuda``.

    Raises:
        ValueError: for an unknown ``LASAGNA_SPARSE_SAMPLER`` value.
    """
    mode = os.environ.get("LASAGNA_SPARSE_SAMPLER", "auto").strip().lower()
    if mode == "torch":
        return True
    if mode == "cuda":
        return False
    if mode != "auto":
        raise ValueError(f"LASAGNA_SPARSE_SAMPLER={mode!r}; expected 'auto', 'torch' or 'cuda'")
    return torch.device(device).type != "cuda" or torch.version.hip is not None


class ChunkPool:
    """Padded uint8 chunks of one cache group in a single growable tensor ``(slots, C, 34**3)``.

    Capacity doubles when full, so adding n chunks costs amortised O(n) copies. Slots are never freed, as the
    CUDA caches never free their batches either.
    """

    def __init__(self, n_channels: int, device: torch.device) -> None:
        self.n_channels = int(n_channels)
        self.device = torch.device(device)
        self._data = torch.empty(0, self.n_channels, _CHUNK_VOXELS, dtype=torch.uint8, device=self.device)
        self.used = 0

    def add(self, batch: torch.Tensor) -> int:
        """Append ``batch`` ``(n, C, 34, 34, 34)`` uint8 and return the slot of its first chunk."""
        if batch.dtype != torch.uint8 or batch.dim() != 5 or tuple(batch.shape[1:]) != (
                self.n_channels, PADDED, PADDED, PADDED):
            raise ValueError(f"chunk batch must be uint8 (n, {self.n_channels}, {PADDED}, {PADDED}, {PADDED}), "
                             f"got {batch.dtype} {tuple(batch.shape)}")
        n = int(batch.shape[0])
        if self.used + n > self._data.shape[0]:
            capacity = max(self.used + n, 2 * self._data.shape[0], 64)
            grown = torch.empty(capacity, self.n_channels, _CHUNK_VOXELS, dtype=torch.uint8, device=self.device)
            grown[:self.used] = self._data[:self.used]
            self._data = grown
        start = self.used
        self._data[start:start + n] = batch.reshape(n, self.n_channels, _CHUNK_VOXELS).to(self.device)
        self.used += n
        return start

    @property
    def data(self) -> torch.Tensor:
        """The filled slots, ``(used, C, 34**3)`` uint8."""
        return self._data[:self.used]

    @property
    def nbytes(self) -> int:
        """Bytes allocated on the device, including unused capacity."""
        return self._data.numel() * self._data.element_size()


def set_table_slots(chunk_table: torch.Tensor, coords_zyx: list[tuple[int, int, int]], start: int) -> None:
    """Point the table entries at ``coords_zyx`` to pool slots ``start``, ``start + 1``, ... (stored as slot + 1).

    One indexed write instead of one per chunk: a first sync can bring thousands of chunks, and every small
    write is a separate kernel launch on MPS.
    """
    if not coords_zyx:
        return
    idx = torch.tensor(coords_zyx, dtype=torch.int64, device=chunk_table.device)
    slots = torch.arange(start + 1, start + 1 + len(coords_zyx), dtype=torch.int64, device=chunk_table.device)
    chunk_table[idx[:, 0], idx[:, 1], idx[:, 2]] = slots


def missing_chunks(
    xyz_fullres: torch.Tensor,
    chunk_table: torch.Tensor,
    origin: torch.Tensor,
    spacing: torch.Tensor,
) -> torch.Tensor:
    """Chunks to load before sampling at ``xyz_fullres``: int64 ``(N, 3)`` as (cz, cy, cx), on the table's device.

    Mirrors ``sparse_prefetch_chunks_kernel.cu``: each point marks its chunk (coordinates truncated toward zero,
    non-finite ones taken as 0, then clamped to the grid) and the 26 chunks around it; the result is the marked
    chunks whose table entry is 0. ``nonzero`` returns them in the same row-major order as the kernel.
    """
    cZ, cY, cX = (int(v) for v in chunk_table.shape)
    xyz = xyz_fullres.detach().reshape(-1, 3).to(torch.float32)
    origin = origin.to(device=xyz.device, dtype=torch.float32).view(1, 3)
    spacing = spacing.to(device=xyz.device, dtype=torch.float32).view(1, 3)
    needed = torch.zeros(cZ + 2, cY + 2, cX + 2, dtype=torch.bool, device=chunk_table.device)
    if xyz.shape[0] > 0:
        # Same operation order as the kernel: (x - origin) / spacing / 32, cast to int (truncation).
        c = (xyz - origin) / spacing / float(CHUNK_SIZE)
        c = torch.where(torch.isfinite(c), c, torch.zeros_like(c))
        c = c.clamp(-_MAX_CHUNK_INDEX, _MAX_CHUNK_INDEX).to(torch.int64)
        cx = c[:, 0].clamp(0, cX - 1)
        cy = c[:, 1].clamp(0, cY - 1)
        cz = c[:, 2].clamp(0, cZ - 1)
        # Marked in a grid with a 1-chunk border, so the 26-neighbourhood needs no bounds checks.
        flat = ((cz + 1) * (cY + 2) + (cy + 1)) * (cX + 2) + (cx + 1)
        centre = torch.zeros((cZ + 2) * (cY + 2) * (cX + 2), dtype=torch.bool, device=chunk_table.device)
        centre[flat.to(chunk_table.device)] = True
        centre = centre.view(cZ + 2, cY + 2, cX + 2)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    needed[1 + dz:cZ + 1 + dz, 1 + dy:cY + 1 + dy, 1 + dx:cX + 1 + dx] |= centre[1:-1, 1:-1, 1:-1]
    needed = needed[1:-1, 1:-1, 1:-1]
    return torch.nonzero(needed & chunk_table.eq(0))


def _locate(chunk_table: torch.Tensor, grid: torch.Tensor, offset: torch.Tensor, inv_scale: torch.Tensor):
    """Per sample point: whether it hits a loaded chunk, its pool slot, fractions and corner offsets.

    Follows the sparse kernels: p = (grid - offset) * inv_scale; non-finite p, a chunk outside the grid or an empty
    table entry give a point that samples 0 (``valid`` False). Inside a chunk the local coordinate is
    ``p - 32 * chunk + 1`` (the padding), floored and clamped to [0, 33].

    Returns:
        valid (N,) bool, slot (N,) int64 (0 where not valid), frac (N, 3) float32 as (fx, fy, fz),
        corner (N, 8) int64 voxel offsets inside a padded chunk in ``_CORNERS_XYZ`` order.
    """
    cZ, cY, cX = (int(v) for v in chunk_table.shape)
    p = (grid.reshape(-1, 3) - offset.view(1, 3)) * inv_scale.view(1, 3)
    finite = torch.isfinite(p).all(dim=1)
    p = torch.where(finite.unsqueeze(1), p, torch.zeros_like(p))
    ci = torch.floor(p / float(CHUNK_SIZE))
    ci_i = ci.clamp(-1, _MAX_CHUNK_INDEX).to(torch.int64)
    ci_x, ci_y, ci_z = ci_i[:, 0], ci_i[:, 1], ci_i[:, 2]
    in_grid = (finite & (ci_x >= 0) & (ci_x < cX) & (ci_y >= 0) & (ci_y < cY) & (ci_z >= 0) & (ci_z < cZ))
    flat = (ci_z.clamp(0, cZ - 1) * cY + ci_y.clamp(0, cY - 1)) * cX + ci_x.clamp(0, cX - 1)
    entry = chunk_table.reshape(-1)[flat]
    valid = in_grid & (entry != 0)
    slot = torch.where(valid, entry - 1, torch.zeros_like(entry))

    local = p - ci * float(CHUNK_SIZE) + 1.0
    local0 = torch.floor(local)
    frac = local - local0
    i0 = local0.clamp(-1, PADDED).to(torch.int64).clamp(0, PADDED - 1)
    i1 = (i0 + 1).clamp(0, PADDED - 1)
    ix = (i0[:, 0], i1[:, 0])
    iy = (i0[:, 1], i1[:, 1])
    iz = (i0[:, 2], i1[:, 2])
    corner = torch.stack([iz[z] * (PADDED * PADDED) + iy[y] * PADDED + ix[x] for x, y, z in _CORNERS_XYZ], dim=1)
    return valid, slot, frac, corner


def _corner_values(pool: torch.Tensor, slot: torch.Tensor, corner: torch.Tensor) -> torch.Tensor:
    """The 8 corner values per point and channel as float32 ``(N, 8, C)`` from ``pool`` ``(slots, C, 34**3)``."""
    if pool.shape[0] == 0:
        return torch.zeros(slot.shape[0], 8, pool.shape[1], dtype=torch.float32, device=slot.device)
    # Advanced indices separated by a slice put their broadcast shape (N, 8) first: result (N, 8, C).
    return pool[slot.unsqueeze(1), :, corner].to(torch.float32)


def _weights(frac: torch.Tensor) -> list[torch.Tensor]:
    """Trilinear weights ``(N,)`` each, in ``_CORNERS_XYZ`` order, written as the kernels write them."""
    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
    gx, gy, gz = 1.0 - fx, 1.0 - fy, 1.0 - fz
    return [gx * gy * gz, fx * gy * gz, gx * fy * gz, fx * fy * gz,
            gx * gy * fz, fx * gy * fz, gx * fy * fz, fx * fy * fz]


def _interpolate(frac: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Sum of weight x value over the 8 corners, accumulated in the kernels' order: ``(N, C)`` float32."""
    out = torch.zeros(values.shape[0], values.shape[2], dtype=torch.float32, device=values.device)
    for k, w in enumerate(_weights(frac)):
        out = out + w.unsqueeze(1) * values[:, k, :]
    return out


def _frac_derivatives(frac: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """d(interp)/d(fx, fy, fz) per point and channel, ``(N, C)`` each, as in the diff kernels' backward."""
    fx, fy, fz = (frac[:, i].unsqueeze(1) for i in range(3))
    gx, gy, gz = 1.0 - fx, 1.0 - fy, 1.0 - fz
    v000, v100, v010, v110, v001, v101, v011, v111 = (v[:, k, :] for k in range(8))
    dfx = gy * gz * (v100 - v000) + fy * gz * (v110 - v010) + gy * fz * (v101 - v001) + fy * fz * (v111 - v011)
    dfy = gx * gz * (v010 - v000) + fx * gz * (v110 - v100) + gx * fz * (v011 - v001) + fx * fz * (v111 - v101)
    dfz = gx * gy * (v001 - v000) + fx * gy * (v101 - v100) + gx * fy * (v011 - v010) + fx * fy * (v111 - v110)
    return dfx, dfy, dfz


def _check_inputs(pool: torch.Tensor, chunk_table: torch.Tensor, grid: torch.Tensor) -> None:
    if pool.dtype != torch.uint8 or pool.dim() != 3 or pool.shape[2] != _CHUNK_VOXELS:
        raise ValueError(f"pool must be uint8 (slots, C, {_CHUNK_VOXELS}), got {pool.dtype} {tuple(pool.shape)}")
    if chunk_table.dtype != torch.int64 or chunk_table.dim() != 3:
        raise ValueError(f"chunk_table must be int64 (cZ, cY, cX), got {chunk_table.dtype} {tuple(chunk_table.shape)}")
    if grid.dtype != torch.float32 or grid.dim() != 4 or grid.shape[3] != 3:
        raise ValueError(f"grid must be float32 (D, H, W, 3), got {grid.dtype} {tuple(grid.shape)}")


def sparse_grid_sample_3d_u8(
    pool: torch.Tensor,
    chunk_table: torch.Tensor,
    grid: torch.Tensor,
    offset: torch.Tensor,
    inv_scale: torch.Tensor,
) -> torch.Tensor:
    """Trilinear sample from the pooled chunk cache, rounded to uint8: ``(C, D, H, W)``.

    Mirrors ``sparse_grid_sample_3d_u8_kernel.cu``; ``chunk_table`` holds ``slot + 1`` into ``pool`` instead of a
    device pointer. Rounding is half away from zero (``roundf``), not torch's half to even.
    """
    _check_inputs(pool, chunk_table, grid)
    D, H, W, _ = grid.shape
    with torch.no_grad():
        valid, slot, frac, corner = _locate(chunk_table, grid, offset, inv_scale)
        val = _interpolate(frac, _corner_values(pool, slot, corner))
        # The interpolated values are >= 0, so floor(v + 0.5) is roundf.
        out = torch.floor(val + 0.5).clamp(0, 255).to(torch.uint8)
        out = torch.where(valid.unsqueeze(1), out, torch.zeros_like(out))
    return out.t().reshape(pool.shape[1], D, H, W)


class _SparseGridSample3DU8Diff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pool, chunk_table, grid, offset, inv_scale):
        valid, slot, frac, corner = _locate(chunk_table, grid, offset, inv_scale)
        val = _interpolate(frac, _corner_values(pool, slot, corner))
        val = torch.where(valid.unsqueeze(1), val, torch.zeros_like(val))
        # Recomputed in backward rather than kept, like the kernel: only the grid is saved.
        ctx.save_for_backward(grid)
        ctx.pool, ctx.chunk_table, ctx.offset, ctx.inv_scale = pool, chunk_table, offset, inv_scale
        D, H, W, _ = grid.shape
        return val.t().reshape(pool.shape[1], D, H, W)

    @staticmethod
    def backward(ctx, grad_output):
        grid, = ctx.saved_tensors
        valid, slot, frac, corner = _locate(ctx.chunk_table, grid, ctx.offset, ctx.inv_scale)
        v = _corner_values(ctx.pool, slot, corner)
        dfx, dfy, dfz = _frac_derivatives(frac, v)
        go = grad_output.reshape(grad_output.shape[0], -1).t()  # (N, C)
        isx, isy, isz = ctx.inv_scale.reshape(3)
        grad = torch.stack([(go * dfx).sum(1) * isx, (go * dfy).sum(1) * isy, (go * dfz).sum(1) * isz], dim=1)
        grad = torch.where(valid.unsqueeze(1), grad, torch.zeros_like(grad))
        return None, None, grad.reshape(grid.shape), None, None


def sparse_grid_sample_3d_u8_diff(
    pool: torch.Tensor,
    chunk_table: torch.Tensor,
    grid: torch.Tensor,
    offset: torch.Tensor,
    inv_scale: torch.Tensor,
) -> torch.Tensor:
    """Trilinear sample from the pooled chunk cache, float32 ``(C, D, H, W)``, differentiable w.r.t. ``grid``.

    Mirrors ``sparse_grid_sample_3d_u8_diff_kernel.cu`` (forward unrounded, backward the analytic trilinear
    derivative times ``inv_scale``; 0 for points that are not on a loaded chunk).
    """
    _check_inputs(pool, chunk_table, grid)
    return _SparseGridSample3DU8Diff.apply(pool, chunk_table, grid, offset, inv_scale)


def _locate_dense(volume: torch.Tensor, grid: torch.Tensor, offset: torch.Tensor, inv_scale: torch.Tensor):
    """Per point of the dense kernel: fractions, corner voxel indices into a flattened (Z, Y, X) channel and
    which corners are inside the volume. Non-finite coordinates sample 0 here (the CUDA kernel leaves them
    undefined)."""
    _, Z, Y, X = (int(v) for v in volume.shape)
    p = (grid.reshape(-1, 3) - offset.view(1, 3)) * inv_scale.view(1, 3)
    finite = torch.isfinite(p).all(dim=1)
    p = torch.where(finite.unsqueeze(1), p, torch.zeros_like(p))
    p0 = torch.floor(p)
    frac = p - p0
    i0 = p0.clamp(-2, _MAX_CHUNK_INDEX).to(torch.int64)
    sizes = (X, Y, Z)
    idx: list[tuple[torch.Tensor, torch.Tensor]] = []
    ok: list[tuple[torch.Tensor, torch.Tensor]] = []
    for a in range(3):
        lo, hi = i0[:, a], i0[:, a] + 1
        ok.append(((lo >= 0) & (lo < sizes[a]), (hi >= 0) & (hi < sizes[a])))
        idx.append((lo.clamp(0, sizes[a] - 1), hi.clamp(0, sizes[a] - 1)))
    corner = torch.stack([idx[2][z] * (Y * X) + idx[1][y] * X + idx[0][x] for x, y, z in _CORNERS_XYZ], dim=1)
    inside = torch.stack([ok[0][x] & ok[1][y] & ok[2][z] & finite for x, y, z in _CORNERS_XYZ], dim=1)
    return frac, corner, inside


def _dense_corner_values(volume: torch.Tensor, corner: torch.Tensor, inside: torch.Tensor) -> torch.Tensor:
    flat = volume.reshape(volume.shape[0], -1)
    v = flat[:, corner].to(torch.float32).permute(1, 2, 0)  # (N, 8, C)
    return torch.where(inside.unsqueeze(2), v, torch.zeros_like(v))


class _DenseGridSample3DU8Diff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, grid, offset, inv_scale):
        frac, corner, inside = _locate_dense(volume, grid, offset, inv_scale)
        val = _interpolate(frac, _dense_corner_values(volume, corner, inside))
        ctx.save_for_backward(grid)
        ctx.volume, ctx.offset, ctx.inv_scale = volume, offset, inv_scale
        D, H, W, _ = grid.shape
        return val.t().reshape(volume.shape[0], D, H, W)

    @staticmethod
    def backward(ctx, grad_output):
        grid, = ctx.saved_tensors
        frac, corner, inside = _locate_dense(ctx.volume, grid, ctx.offset, ctx.inv_scale)
        dfx, dfy, dfz = _frac_derivatives(frac, _dense_corner_values(ctx.volume, corner, inside))
        go = grad_output.reshape(grad_output.shape[0], -1).t()
        isx, isy, isz = ctx.inv_scale.reshape(3)
        grad = torch.stack([(go * dfx).sum(1) * isx, (go * dfy).sum(1) * isy, (go * dfz).sum(1) * isz], dim=1)
        return None, grad.reshape(grid.shape), None, None


def dense_grid_sample_3d_u8_diff(
    volume: torch.Tensor,
    grid: torch.Tensor,
    offset: torch.Tensor,
    inv_scale: torch.Tensor,
) -> torch.Tensor:
    """Trilinear sample of a dense uint8 volume ``(C, Z, Y, X)``, float32 ``(C, D, H, W)``, differentiable
    w.r.t. ``grid``. Mirrors ``grid_sample_3d_u8_diff_kernel.cu``: each corner outside the volume counts as 0."""
    if volume.dtype != torch.uint8 or volume.dim() != 4:
        raise ValueError(f"volume must be uint8 (C, Z, Y, X), got {volume.dtype} {tuple(volume.shape)}")
    if grid.dtype != torch.float32 or grid.dim() != 4 or grid.shape[3] != 3:
        raise ValueError(f"grid must be float32 (D, H, W, 3), got {grid.dtype} {tuple(grid.shape)}")
    return _DenseGridSample3DU8Diff.apply(volume, grid, offset, inv_scale)
