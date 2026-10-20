"""The torch sampler (sparse_sample_torch) against line-by-line NumPy ports of lasagna's CUDA kernels.

The streaming fit on MPS, CPU and ROCm samples the prediction only through sparse_sample_torch. If it drifted from
the CUDA kernels (chunk lookup, padding offset, rounding, gradient), a fit off NVIDIA would converge to different
surfaces than the same fit in the cloud, without any error. The references below follow the .cu files statement
by statement, in float32.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import sparse_sample_torch as sst

f32 = np.float32
P = sst.PADDED


def _ref_locate(table_slots, point, offset, inv_scale):
	"""Kernel preamble of sparse_grid_sample_3d_u8(_diff)_kernel.cu: (slot, fx, fy, fz, corner indices) or None."""
	cZ, cY, cX = table_slots.shape
	px, py, pz = ((f32(point[a]) - f32(offset[a])) * f32(inv_scale[a]) for a in range(3))
	if not (np.isfinite(px) and np.isfinite(py) and np.isfinite(pz)):
		return None
	ci_x, ci_y, ci_z = (int(np.floor(v / f32(32.0))) for v in (px, py, pz))
	if not (0 <= ci_x < cX and 0 <= ci_y < cY and 0 <= ci_z < cZ):
		return None
	entry = int(table_slots[ci_z, ci_y, ci_x])
	if entry == 0:
		return None
	lx = px - f32(ci_x * 32.0) + f32(1.0)
	ly = py - f32(ci_y * 32.0) + f32(1.0)
	lz = pz - f32(ci_z * 32.0) + f32(1.0)
	fx, fy, fz = lx - np.floor(lx), ly - np.floor(ly), lz - np.floor(lz)
	ix0, iy0, iz0 = (max(0, min(33, int(np.floor(v)))) for v in (lx, ly, lz))
	ix1, iy1, iz1 = (max(0, min(33, i + 1)) for i in (ix0, iy0, iz0))
	return entry - 1, fx, fy, fz, (ix0, ix1), (iy0, iy1), (iz0, iz1)


def _ref_corners(chunk, ix, iy, iz):
	"""The 8 corner values of one channel in the kernels' order (v000, v100, v010, v110, v001, v101, v011, v111)."""
	return [f32(chunk[iz[z], iy[y], ix[x]]) for x, y, z in sst._CORNERS_XYZ]


def _ref_weights(fx, fy, fz):
	one = f32(1.0)
	return [(one - fx) * (one - fy) * (one - fz), fx * (one - fy) * (one - fz), (one - fx) * fy * (one - fz),
		fx * fy * (one - fz), (one - fx) * (one - fy) * fz, fx * (one - fy) * fz, (one - fx) * fy * fz, fx * fy * fz]


def _ref_dfrac(fx, fy, fz, v):
	one = f32(1.0)
	v000, v100, v010, v110, v001, v101, v011, v111 = v
	dfx = (one - fy) * (one - fz) * (v100 - v000) + fy * (one - fz) * (v110 - v010) \
		+ (one - fy) * fz * (v101 - v001) + fy * fz * (v111 - v011)
	dfy = (one - fx) * (one - fz) * (v010 - v000) + fx * (one - fz) * (v110 - v100) \
		+ (one - fx) * fz * (v011 - v001) + fx * fz * (v111 - v101)
	dfz = (one - fx) * (one - fy) * (v001 - v000) + fx * (one - fy) * (v101 - v100) \
		+ (one - fx) * fy * (v011 - v010) + fx * fy * (v111 - v110)
	return dfx, dfy, dfz


def _ref_sparse(pool, table, points, offset, inv_scale, grad_out=None):
	"""Forward values (N, C) unrounded, and with grad_out (N, C) the backward grad (N, 3)."""
	C = pool.shape[1]
	vals = np.zeros((len(points), C), dtype=f32)
	grads = np.zeros((len(points), 3), dtype=f32)
	for n, point in enumerate(points):
		loc = _ref_locate(table, point, offset, inv_scale)
		if loc is None:
			continue
		slot, fx, fy, fz, ix, iy, iz = loc
		for c in range(C):
			v = _ref_corners(pool[slot, c].reshape(P, P, P), ix, iy, iz)
			acc = f32(0.0)
			for w, vk in zip(_ref_weights(fx, fy, fz), v):
				acc = acc + w * vk
			vals[n, c] = acc
			if grad_out is not None:
				dfx, dfy, dfz = _ref_dfrac(fx, fy, fz, v)
				go = f32(grad_out[n, c])
				grads[n] += np.array([go * dfx * f32(inv_scale[0]), go * dfy * f32(inv_scale[1]),
					go * dfz * f32(inv_scale[2])], dtype=f32)
	return vals, grads


def _ref_missing(points, table, origin, spacing):
	"""sparse_prefetch_chunks_kernel.cu: truncation toward zero, non-finite -> 0, clamp, 26-neighbourhood."""
	cZ, cY, cX = table.shape
	needed = np.zeros(table.shape, dtype=bool)
	for x, y, z in points:
		c = []
		for v, o, s, size in ((x, origin[0], spacing[0], cX), (y, origin[1], spacing[1], cY),
				(z, origin[2], spacing[2], cZ)):
			ci = int((f32(v) - f32(o)) / f32(s) / f32(32.0)) if np.isfinite(v) else 0
			c.append(max(0, min(size - 1, ci)))
		cx, cy, cz = c
		needed[max(0, cz - 1):cz + 2, max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = True
	return np.argwhere(needed & (table == 0))


def _make_cache(seed=0, C=2, grid=(3, 4, 5), empty_frac=0.3):
	"""A random pool + table (slot + 1, some chunks empty) for a (cZ, cY, cX) chunk grid."""
	rng = np.random.default_rng(seed)
	cZ, cY, cX = grid
	n = cZ * cY * cX
	pool = rng.integers(0, 256, size=(n, C, P ** 3), dtype=np.uint8)
	table = (rng.permutation(n) + 1).reshape(grid).astype(np.int64)
	table[rng.random(grid) < empty_frac] = 0
	return pool, table


def _points(seed, grid, n=400, spacing=(2.0, 2.0, 2.0), offset=(10.0, -5.0, 3.0)):
	"""Points over the chunk grid and a margin around it, many on chunk and voxel borders, plus NaN/Inf."""
	rng = np.random.default_rng(seed)
	cZ, cY, cX = grid
	extent = np.array([cX, cY, cZ], dtype=np.float64) * 32.0
	local = rng.uniform(-40.0, extent + 40.0, size=(n, 3))
	local[: n // 4] = np.round(local[: n // 4])  # whole voxels
	local[n // 4: n // 2] = np.round(local[n // 4: n // 2] / 32.0) * 32.0 + rng.choice([-1e-3, 0.0, 1e-3], size=(n // 4, 3))
	full = local * np.array(spacing) + np.array(offset)
	full[0] = [np.nan, 1.0, 1.0]
	full[1] = [1.0, np.inf, 1.0]
	return full.astype(f32)


class SparseSampleTests(unittest.TestCase):
	spacing = (2.0, 2.0, 2.0)
	offset = (10.0, -5.0, 3.0)

	def _tensors(self, seed=0):
		pool, table = _make_cache(seed)
		points = _points(seed + 1, table.shape, spacing=self.spacing, offset=self.offset)
		grid = torch.from_numpy(points).reshape(4, 10, 10, 3)
		offset = torch.tensor(self.offset, dtype=torch.float32)
		inv_scale = torch.tensor([1.0 / s for s in self.spacing], dtype=torch.float32)
		return pool, table, points, grid, offset, inv_scale

	def test_forward_matches_kernel(self) -> None:
		"""Rounded uint8 samples equal the kernel's roundf of the same trilinear sum, 0 off loaded chunks."""
		pool, table, points, grid, offset, inv_scale = self._tensors()
		out = sst.sparse_grid_sample_3d_u8(torch.from_numpy(pool), torch.from_numpy(table), grid, offset, inv_scale)
		ref, _ = _ref_sparse(pool, table, points, self.offset, [1.0 / s for s in self.spacing])
		expected = np.clip(np.floor(ref + 0.5), 0, 255).astype(np.uint8)
		self.assertEqual(tuple(out.shape), (2, 4, 10, 10))
		np.testing.assert_array_equal(out.reshape(2, -1).T.numpy(), expected)
		self.assertTrue((expected == 0).any(axis=1).any() and (expected != 0).any())

	def test_diff_forward_and_backward_match_kernel(self) -> None:
		"""Unrounded values and the analytic gradient w.r.t. the grid follow the diff kernel's fwd and bwd."""
		pool, table, points, grid, offset, inv_scale = self._tensors(seed=3)
		grid = grid.clone().requires_grad_(True)
		out = sst.sparse_grid_sample_3d_u8_diff(torch.from_numpy(pool), torch.from_numpy(table), grid, offset, inv_scale)
		go = torch.from_numpy(np.random.default_rng(9).normal(size=tuple(out.shape)).astype(f32))
		out.backward(go)
		ref_vals, ref_grad = _ref_sparse(pool, table, points, self.offset, [1.0 / s for s in self.spacing],
			grad_out=go.reshape(2, -1).T.numpy())
		np.testing.assert_allclose(out.detach().reshape(2, -1).T.numpy(), ref_vals, rtol=0, atol=1e-3)
		np.testing.assert_allclose(grid.grad.reshape(-1, 3).numpy(), ref_grad, rtol=1e-5, atol=1e-3)

	def test_rounds_half_away_from_zero(self) -> None:
		"""The kernel rounds with roundf (2.5 -> 3); torch.round would give 2 and shift every half-way sample."""
		chunk = torch.zeros(1, 1, P, P, P, dtype=torch.uint8)
		chunk[0, 0, :, :, 1] = 2
		chunk[0, 0, :, :, 2] = 3
		table = torch.ones(1, 1, 1, dtype=torch.int64)
		grid = torch.tensor([[[[0.5, 4.0, 4.0]]]])  # local x 1.5: halfway between 2 and 3
		out = sst.sparse_grid_sample_3d_u8(chunk.reshape(1, 1, -1), table, grid, torch.zeros(3), torch.ones(3))
		self.assertEqual(int(out.reshape(-1)[0]), 3)

	def test_noncontiguous_grid(self) -> None:
		"""The kernels read the grid through its strides; a transposed grid must sample like its contiguous copy."""
		pool, table, _, grid, offset, inv_scale = self._tensors(seed=5)
		strided = grid.permute(1, 0, 2, 3)
		self.assertFalse(strided.is_contiguous())
		pool_t, table_t = torch.from_numpy(pool), torch.from_numpy(table)
		torch.testing.assert_close(
			sst.sparse_grid_sample_3d_u8(pool_t, table_t, strided, offset, inv_scale),
			sst.sparse_grid_sample_3d_u8(pool_t, table_t, strided.contiguous(), offset, inv_scale))

	def test_empty_pool_samples_zero(self) -> None:
		"""Before the first sync the pool has no slots; sampling must give zeros, not an index error."""
		table = torch.zeros(2, 2, 2, dtype=torch.int64)
		pool = torch.zeros(0, 3, P ** 3, dtype=torch.uint8)
		grid = torch.full((1, 2, 2, 3), 5.0)
		out = sst.sparse_grid_sample_3d_u8(pool, table, grid, torch.zeros(3), torch.ones(3))
		self.assertEqual(int(out.abs().sum()), 0)

	def test_missing_chunks_matches_kernel(self) -> None:
		"""The prefetch set (the point's chunk and its 26 neighbours, minus loaded ones) is the kernel's, in order."""
		_, table = _make_cache(seed=7, grid=(4, 5, 6), empty_frac=0.6)
		points = _points(8, table.shape, n=40, spacing=self.spacing, offset=self.offset)
		points[2] = [-1e30, 5.0, 1e30]  # clamped, not overflowing
		got = sst.missing_chunks(torch.from_numpy(points).reshape(4, 10, 3), torch.from_numpy(table),
			torch.tensor(self.offset), torch.tensor(self.spacing))
		np.testing.assert_array_equal(got.numpy(), _ref_missing(points, table, self.offset, self.spacing))


class DenseSampleTests(unittest.TestCase):
	def test_dense_matches_kernel(self) -> None:
		"""cyl_outside samples a dense uint8 field; each corner outside the volume counts as 0 (the CUDA kernel)."""
		rng = np.random.default_rng(1)
		vol = rng.integers(0, 256, size=(1, 6, 7, 8), dtype=np.uint8)
		pts = rng.uniform(-2.0, 9.0, size=(50, 3)).astype(f32)
		pts[:10] = np.round(pts[:10])
		grid = torch.from_numpy(pts).reshape(5, 10, 1, 3).requires_grad_(True)
		out = sst.dense_grid_sample_3d_u8_diff(torch.from_numpy(vol), grid, torch.zeros(3), torch.ones(3))
		out.sum().backward()
		ref = np.zeros(50, dtype=f32)
		ref_grad = np.zeros((50, 3), dtype=f32)
		_, Z, Y, X = vol.shape
		for n, (px, py, pz) in enumerate(pts):
			fx, fy, fz = px - np.floor(px), py - np.floor(py), pz - np.floor(pz)
			i0 = [int(np.floor(v)) for v in (px, py, pz)]
			v = []
			for x, y, z in sst._CORNERS_XYZ:
				ix, iy, iz = i0[0] + x, i0[1] + y, i0[2] + z
				v.append(f32(vol[0, iz, iy, ix]) if 0 <= ix < X and 0 <= iy < Y and 0 <= iz < Z else f32(0.0))
			for w, vk in zip(_ref_weights(fx, fy, fz), v):
				ref[n] += w * vk
			ref_grad[n] = _ref_dfrac(fx, fy, fz, v)
		np.testing.assert_allclose(out.detach().reshape(-1).numpy(), ref, atol=1e-3)
		np.testing.assert_allclose(grid.grad.reshape(-1, 3).numpy(), ref_grad, rtol=1e-5, atol=1e-3)


class PoolAndBackendTests(unittest.TestCase):
	def test_pool_grows_and_keeps_slots(self) -> None:
		"""Growing the pool must keep earlier chunks at their slots, or the table would point at the wrong data."""
		pool = sst.ChunkPool(2, torch.device("cpu"))
		a = torch.randint(0, 256, (50, 2, P, P, P), dtype=torch.uint8)
		b = torch.randint(0, 256, (30, 2, P, P, P), dtype=torch.uint8)
		self.assertEqual(pool.add(a), 0)
		self.assertEqual(pool.add(b), 50)
		torch.testing.assert_close(pool.data, torch.cat([a, b]).reshape(80, 2, -1))
		with self.assertRaises(ValueError):
			pool.add(torch.zeros(1, 3, P, P, P, dtype=torch.uint8))

	def test_set_table_slots(self) -> None:
		table = torch.zeros(2, 3, 4, dtype=torch.int64)
		sst.set_table_slots(table, [(1, 2, 3), (0, 0, 1)], start=5)
		self.assertEqual(int(table[1, 2, 3]), 6)
		self.assertEqual(int(table[0, 0, 1]), 7)
		self.assertEqual(int((table != 0).sum()), 2)

	def test_backend_choice(self) -> None:
		"""NVIDIA CUDA keeps the kernels; CPU, MPS and ROCm take the torch path; the variable overrides."""
		with mock.patch.dict(os.environ, {}, clear=False):
			os.environ.pop("LASAGNA_SPARSE_SAMPLER", None)
			self.assertTrue(sst.use_torch_sampler("cpu"))
			self.assertTrue(sst.use_torch_sampler("mps"))
			with mock.patch.object(torch.version, "hip", None):
				self.assertFalse(sst.use_torch_sampler("cuda"))
			with mock.patch.object(torch.version, "hip", "7.2.53211"):
				self.assertTrue(sst.use_torch_sampler("cuda"))
		with mock.patch.dict(os.environ, {"LASAGNA_SPARSE_SAMPLER": "torch"}):
			with mock.patch.object(torch.version, "hip", None):
				self.assertTrue(sst.use_torch_sampler("cuda"))
		with mock.patch.dict(os.environ, {"LASAGNA_SPARSE_SAMPLER": "nvcc"}):
			with self.assertRaises(ValueError):
				sst.use_torch_sampler("cpu")


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip is None, "needs an NVIDIA GPU and nvcc")
class CudaKernelParityTests(unittest.TestCase):
	def test_torch_sampler_matches_cuda_kernels(self) -> None:
		"""On an NVIDIA GPU, the torch path against the real kernels (pointer table) on the same chunks and points."""
		from sparse_grid_sample_3d_u8 import sparse_grid_sample_3d_u8 as cuda_fwd
		from sparse_grid_sample_3d_u8_diff import sparse_grid_sample_3d_u8_diff as cuda_diff
		from sparse_prefetch_chunks import missing_chunks as cuda_missing
		dev = torch.device("cuda")
		pool, table = _make_cache(seed=11)
		pool_t = torch.from_numpy(pool).to(dev)
		table_t = torch.from_numpy(table).to(dev)
		chunk_bytes = pool.shape[1] * P ** 3
		ptr_table = torch.where(table_t > 0, pool_t.data_ptr() + (table_t - 1) * chunk_bytes, torch.zeros_like(table_t))
		points = _points(12, table.shape, n=4000)
		grid = torch.from_numpy(points).reshape(40, 10, 10, 3).to(dev)
		offset = torch.tensor((10.0, -5.0, 3.0), device=dev)
		inv_scale = torch.full((3,), 0.5, device=dev)
		C = pool.shape[1]
		fwd_t = sst.sparse_grid_sample_3d_u8(pool_t, table_t, grid, offset, inv_scale)
		fwd_c = cuda_fwd(ptr_table, C, grid, offset, inv_scale)
		# nvcc fuses multiply-adds, so a sum within float rounding of .5 may round the other way.
		self.assertLessEqual(int((fwd_t.int() - fwd_c.int()).abs().max()), 1)
		self.assertLess(float((fwd_t != fwd_c).float().mean()), 1e-3)
		g_t = grid.clone().requires_grad_(True)
		g_c = grid.clone().requires_grad_(True)
		out_t = sst.sparse_grid_sample_3d_u8_diff(pool_t, table_t, g_t, offset, inv_scale)
		out_c = cuda_diff(ptr_table, C, g_c, offset, inv_scale)
		torch.testing.assert_close(out_t, out_c, rtol=0, atol=1e-3)
		go = torch.randn_like(out_t)
		out_t.backward(go)
		out_c.backward(go)
		torch.testing.assert_close(g_t.grad, g_c.grad, rtol=1e-4, atol=1e-3)
		spacing = torch.full((3,), 2.0, device=dev)
		torch.testing.assert_close(sst.missing_chunks(grid, table_t, offset, spacing),
			cuda_missing(grid, table_t, offset, spacing))


class TensorStoreCacheOnCpuTests(unittest.TestCase):
	def test_streaming_cache_samples_volume_on_cpu(self) -> None:
		"""End to end on CPU: prefetch, sync and sample a local zarr through the cache, against direct trilinear
		interpolation of the array. This is the path the streaming fit takes off NVIDIA."""
		try:
			import zarr
			import sparse_tensorstore_cache
		except ImportError as exc:  # pragma: no cover - lasagna's env has both
			self.skipTest(f"needs zarr and tensorstore: {exc}")
		rng = np.random.default_rng(4)
		vol = rng.integers(0, 256, size=(2, 70, 50, 40), dtype=np.uint8)
		with tempfile.TemporaryDirectory() as tmp:
			path = os.path.join(tmp, "vol.zarr")
			arr = zarr.create_array(path, shape=vol.shape, chunks=(1, 32, 32, 32), dtype="uint8", zarr_format=2)
			arr[:] = vol
			cache = sparse_tensorstore_cache.TensorStoreSparseChunkGroupCache(
				channels=["a", "b"], zarr_path=path, vol_shape_zyx=(70, 50, 40),
				channel_indices={"a": 0, "b": 1}, is_3d_zarr=False, device=torch.device("cpu"),
				cache_pool_bytes=1 << 24)
			pts = rng.uniform(0.0, [38.9, 48.9, 68.9], size=(60, 3)).astype(f32)  # whole cells: every corner inside
			grid = torch.from_numpy(pts).reshape(6, 10, 1, 3)
			cache.prefetch(grid, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
			cache.sync()
			self.assertGreater(cache.loaded_chunks(), 0)
			out = cache.grid_sample(grid, torch.zeros(3), torch.ones(3), diff=True)
		ref = np.zeros((60, 2), dtype=f32)
		for n, (px, py, pz) in enumerate(pts):
			fx, fy, fz = px - np.floor(px), py - np.floor(py), pz - np.floor(pz)
			ix, iy, iz = int(np.floor(px)), int(np.floor(py)), int(np.floor(pz))
			for (x, y, z), w in zip(sst._CORNERS_XYZ, _ref_weights(fx, fy, fz)):
				ref[n] += w * vol[:, iz + z, iy + y, ix + x].astype(f32)
		np.testing.assert_allclose(out.reshape(2, -1).T.numpy(), ref, atol=1e-3)


if __name__ == "__main__":
	unittest.main()
