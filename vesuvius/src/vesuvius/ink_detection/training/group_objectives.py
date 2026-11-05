"""Training objectives over groups of the data: GroupDRO, PCGrad and how far the groups' gradients agree.

A group is a set of datasets that the config names with the same ``objective_group`` (for example one scroll, so
each scroll is a group; ``config.GroupObjectiveConfig``). The plain objective is the loss of the whole batch. These
objectives look at each group of a batch on its own:

- **GroupDRO** (:class:`GroupDRO`; Sagawa et al. 2020, "Distributionally Robust Neural Networks for Group Shifts").
  Every group has a weight q (they sum to 1). At each step the weight of each group in the batch is multiplied by
  ``exp(step_size * its loss)`` and the weights are normalised again, so a group whose loss stays high gets a larger
  share. The batch's objective is the groups' losses weighted by q, over the groups in the batch (their q normalised
  to sum to 1). Groups not in the batch keep their weight.
- **PCGrad** (:func:`pcgrad`; Yu et al. 2020, "Gradient Surgery for Multi-Task Learning"). Each group's loss gets
  its own gradient. Where two gradients point against each other (negative dot product), the conflicting part of one
  is removed: it is projected onto the normal plane of the other. The step is the mean of the projected gradients.
  The order in which a gradient is projected onto the others is drawn from a seeded generator, as in the paper, so
  a run is reproducible.
- **Agreement** (:class:`GradientAgreement`): with PCGrad, the cosine of every pair of groups' gradients in each
  batch where both are present, before the projection. Its mean and how often it is negative say which groups pull
  the model the same way and which pull against each other. Written to ``group_objective.json`` with the
  groups' mean losses (and GroupDRO's weights).

ROCm (2026-11-05): the first PCGrad fine-tune on the gpurig's RX 7700 XT (torch 2.12.1+rocm7.2) died at step ~104
of 4000 with SIGFPE: an integer divide by zero in libamdhip64.so's kernel-launch code (a work-group size of 0), with
no Python exception. Two GPU paths here ran only with a group objective, and both are now gone:

- A group's samples were taken with an index tensor (``x[index]``), whose backward is ``index_put_`` with
  accumulation (with deterministic kernels, a sort-based kernel whose launch shape depends on the indices). They
  are now taken with basic slices (:func:`_take`), whose backward is a copy into zeros: the same values.
- PCGrad converted the full-model gradients to float64 for every pair's dot product, O(k^2) model-sized copies and
  dots per step. It now takes the groups' Gram matrix once (:func:`gram`, chunked) and does the projection on the
  k x k matrix on the CPU (:func:`pcgrad_weights`): each projected gradient is a combination of the original ones,
  so its dot products are combinations of the Gram matrix's entries. The step is the same up to float rounding.

ROCm, second crash (2026-11-05): the next PCGrad fine-tune on the gpurig died in the backward pass of step 1 with
SIGSEGV in the autograd thread, inside libgcc_s's C++ unwinder (a native error that crashed while it was thrown), with
the GPU's memory at 96% and no other kernel message: most likely a failed GPU allocation. The groups' gradient stack
(one model-sized float32 row per group in the batch, up to the batch size: roughly 1 GiB for reader-v3 at 8 groups) and its float64 Gram chunks were on
the GPU, allocated before the first group's backward pass, so every backward pass ran with that much less memory than
the plain training's. The stack, the Gram matrix and the combination are now on the CPU (:func:`gradient_stack`): the
GPU holds one group's gradient at a time, as in the plain training. Only the combined step goes back to the GPU. The
values are the same; the cost is one device-to-host copy of each group's gradient per step.

Not done here: assigning samples to groups (the dataset's ``dataset_index`` and the config's ``dataset_groups``),
the training loop (``train``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch

LossFn = Callable[[Any, Any], torch.Tensor]


def sample_groups(dataset_index: torch.Tensor, dataset_groups: Sequence[int | None]) -> torch.Tensor:
    """Each sample's group index, from the dataset it came from.

    Args:
        dataset_index: (B,) int, the config dataset of each sample (``InkDataset`` with ``emit_dataset_index``).
        dataset_groups: the group index of each config dataset, None for one without a group.

    Returns:
        (B,) long on ``dataset_index``'s device.

    Raises:
        ValueError: a sample of a dataset without a group, or a dataset index outside the config.
    """
    out = []
    for value in dataset_index.reshape(-1).tolist():
        if not 0 <= int(value) < len(dataset_groups):
            raise ValueError(f"dataset index {value} is not one of the {len(dataset_groups)} config datasets")
        group = dataset_groups[int(value)]
        if group is None:
            raise ValueError(f"a training sample of datasets[{int(value)}], which has no objective_group")
        out.append(int(group))
    return torch.tensor(out, dtype=torch.long, device=dataset_index.device)


def _runs(rows: Sequence[int]) -> list[tuple[int, int]]:
    """``rows`` (ascending sample indices) as ``[start, stop)`` runs of consecutive indices."""
    out: list[tuple[int, int]] = []
    for row in rows:
        if out and out[-1][1] == row:
            out[-1] = (out[-1][0], row + 1)
        else:
            out.append((row, row + 1))
    return out


def _take(tensor: torch.Tensor, rows: Sequence[int]) -> torch.Tensor:
    """The samples ``rows`` (ascending, non-empty) of ``tensor``, by basic slices along the batch dimension.

    Not ``tensor[index_tensor]``: that backward is ``index_put_`` with accumulation, a GPU path the plain training
    never runs (module docstring, ROCm). A slice's backward copies into zeros; the values are the same.
    """
    if not rows:
        raise ValueError("no samples to take")
    parts = [tensor[start:stop] for start, stop in _runs(rows)]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def _select(value: Any, rows: Sequence[int]) -> Any:
    """``value`` (a tensor, or a list or tuple of them for deep supervision) with only the samples ``rows``."""
    if isinstance(value, (list, tuple)):
        return type(value)(_take(item, rows) for item in value)
    return _take(value, rows)


def group_losses(loss_fn: LossFn, predictions: Any, targets: Any, groups: torch.Tensor) -> dict[int, torch.Tensor]:
    """The loss of each group in the batch, on its samples alone.

    Args:
        loss_fn: ``(predictions, targets) -> scalar``, the training loss.
        predictions, targets: the batch as the loss takes it (batch first; lists for deep supervision).
        groups: (B,) long, each sample's group (:func:`sample_groups`).

    Returns:
        ``{group: scalar loss}`` for each group present, in group order.
    """
    out: dict[int, torch.Tensor] = {}
    per_sample = [int(g) for g in groups.reshape(-1).tolist()]
    for group in sorted(set(per_sample)):
        rows = [row for row, g in enumerate(per_sample) if g == group]
        out[group] = loss_fn(_select(predictions, rows), _select(targets, rows))
    return out


class GroupDRO:
    """GroupDRO's group weights and objective (module docstring).

    Args:
        n_groups: how many groups there are.
        step_size: the exponent's step size (``group_objective.dro_step_size``); 0.01 is the paper's.
    """

    def __init__(self, n_groups: int, step_size: float) -> None:
        if n_groups < 2:
            raise ValueError(f"GroupDRO needs two or more groups, got {n_groups}")
        if not step_size > 0.0:
            raise ValueError(f"GroupDRO step size must be positive, got {step_size}")
        self.step_size = float(step_size)
        # float64 on the CPU: the weights are few, and exp of a long run's sum must not lose its smallest ones.
        self.weights = torch.full((n_groups,), 1.0 / n_groups, dtype=torch.float64)

    def objective(self, losses: dict[int, torch.Tensor]) -> torch.Tensor:
        """Update the weights from ``losses`` (one step) and return the weighted loss over the groups in the batch.

        The update comes first, as in the paper's algorithm: the batch's own losses already move its weights.
        """
        if not losses:
            raise ValueError("GroupDRO got a batch with no group")
        for group, loss in losses.items():
            self.weights[group] *= math.exp(self.step_size * float(loss.detach()))
        self.weights /= self.weights.sum()
        present = sorted(losses)
        share = self.weights[present]
        share = share / share.sum()
        return sum(float(share[i]) * losses[g] for i, g in enumerate(present))

    def state(self) -> list[float]:
        """The weights, in group order."""
        return [float(w) for w in self.weights]


# Columns of the gradient stack converted to float64 at a time in :func:`gram`: 2^22 columns of k groups is k x 32 MiB,
# small next to the stack itself (k x the model's parameters in float32).
GRAM_CHUNK = 1 << 22


def gram(stack: torch.Tensor, chunk: int = GRAM_CHUNK) -> torch.Tensor:
    """The Gram matrix of the rows of ``stack``: every pair's dot product, in float64.

    Args:
        stack: (k, N) float, one flat gradient per row.
        chunk: how many columns are converted to float64 at a time (memory, not the result).

    Returns:
        (k, k) float64 on the CPU.
    """
    if stack.ndim != 2:
        raise ValueError(f"gram needs a (k, N) stack, got shape {tuple(stack.shape)}")
    k, n = stack.shape
    # float64: a large model's sums in float32 can flip the sign of a near-zero conflict.
    out = torch.zeros((k, k), dtype=torch.float64, device=stack.device)
    for start in range(0, n, chunk):
        part = stack[:, start:start + chunk].double()
        out += part @ part.T
    return out.cpu()


def gradient_stack(k: int, n: int, dtype: torch.dtype) -> torch.Tensor:
    """Zeros for ``k`` flat gradients of ``n`` values, on the CPU whatever the model's device (module docstring, ROCm
    second crash): the stack must not take GPU memory from the backward passes.

    Returns:
        (k, n) of ``dtype`` on the CPU.
    """
    if k < 1 or n < 1:
        raise ValueError(f"a gradient stack needs one or more rows and values, got ({k}, {n})")
    return torch.zeros((k, n), dtype=dtype, device="cpu")


def pcgrad_weights(gram_matrix: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """PCGrad's step as weights of the original gradients (module docstring): ``step = sum_j w_j g_j``.

    The rule is the paper's: each g_i is projected against every other original g_j, in an order drawn from
    ``generator``, and when ``p_i . g_j < 0``, ``p_i -= (p_i . g_j) / |g_j|^2 * g_j``. Each p_i stays a combination
    ``sum_m c_m g_m``, so ``p_i . g_j = c . G[:, j]`` with G the Gram matrix; the step is the mean of the p_i. A
    zero gradient takes no part as the other side of a projection.

    Args:
        gram_matrix: (k, k) float64, :func:`gram` of the groups' gradients.
        generator: a CPU generator, seeded once per run; one ``randperm`` per gradient when k > 1.

    Returns:
        (k,) float64 on the CPU. One gradient gets weight 1.
    """
    k = int(gram_matrix.shape[0])
    if k < 1 or gram_matrix.shape != (k, k):
        raise ValueError(f"pcgrad needs a square Gram matrix of one or more gradients, got {tuple(gram_matrix.shape)}")
    g = gram_matrix.to(device="cpu", dtype=torch.float64)
    if k == 1:
        return torch.ones(1, dtype=torch.float64)
    norms = [float(g[j, j]) for j in range(k)]
    weights = torch.zeros(k, dtype=torch.float64)
    for i in range(k):
        coeff = torch.zeros(k, dtype=torch.float64)
        coeff[i] = 1.0
        for j in torch.randperm(k, generator=generator).tolist():
            if j == i or norms[j] == 0.0:
                continue
            dot = float(coeff @ g[:, j])
            if dot < 0.0:
                coeff[j] -= dot / norms[j]
        weights += coeff
    return weights / k


def combine(stack: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``sum_j weights[j] * stack[j]``: (N,) in ``stack``'s dtype and device."""
    return weights.to(device=stack.device, dtype=stack.dtype) @ stack


def pcgrad(grads: Sequence[torch.Tensor], generator: torch.Generator) -> torch.Tensor:
    """PCGrad's step from the groups' flat gradients (module docstring): the mean of each gradient after its
    conflicting parts are projected out (:func:`pcgrad_weights` on their :func:`gram`).

    Args:
        grads: one flat gradient per group in the batch, all the same shape, dtype and device.
        generator: a CPU generator, seeded once per run.

    Returns:
        The combined flat gradient. One gradient is returned unchanged (a copy).

    Raises:
        ValueError: no gradient, or gradients of different shapes.
    """
    if not grads:
        raise ValueError("pcgrad needs one or more gradients")
    shape = grads[0].shape
    if any(g.shape != shape for g in grads):
        raise ValueError("pcgrad needs gradients of one shape")
    if len(grads) == 1:
        return grads[0].clone()
    stack = torch.stack([g.reshape(-1) for g in grads])
    return combine(stack, pcgrad_weights(gram(stack), generator)).view(shape)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    """The cosine of two flat gradients, None when either is zero."""
    na, nb = float(torch.linalg.vector_norm(a.double())), float(torch.linalg.vector_norm(b.double()))
    if na == 0.0 or nb == 0.0:
        return None
    return float(torch.dot(a.double(), b.double())) / (na * nb)


class GradientAgreement:
    """How far the groups' gradients agree over a run (module docstring), and each group's loss.

    Args:
        groups: the group names, in group order.
    """

    def __init__(self, groups: Sequence[str]) -> None:
        self.groups = list(groups)
        n = len(self.groups)
        self.pairs = [[0] * n for _ in range(n)]  # batches with both groups present
        self.cos_sum = [[0.0] * n for _ in range(n)]
        self.conflicts = [[0] * n for _ in range(n)]  # of those, batches with a negative cosine
        self.batches = [0] * n
        self.loss_sum = [0.0] * n

    def add_losses(self, losses: dict[int, torch.Tensor]) -> None:
        """Count one batch's group losses."""
        for group, loss in losses.items():
            self.batches[group] += 1
            self.loss_sum[group] += float(loss.detach())

    def add_gradients(self, grads: dict[int, torch.Tensor]) -> None:
        """Count the cosine of every pair of one batch's group gradients (before PCGrad's projection)."""
        present = sorted(grads)
        if present:
            self.add_gram(present, gram(torch.stack([grads[g].reshape(-1) for g in present])))

    def add_gram(self, present: Sequence[int], gram_matrix: torch.Tensor) -> None:
        """:meth:`add_gradients` from the batch's Gram matrix: row r of ``gram_matrix`` is group ``present[r]``.
        A group whose gradient is not finite (its squared norm is not) takes no part, as a zero one."""
        g = gram_matrix.to(device="cpu", dtype=torch.float64)
        norms = [float(g[r, r]) for r in range(len(present))]
        for x in range(len(present)):
            for y in range(x + 1, len(present)):
                i, j = present[x], present[y]
                if not (math.isfinite(norms[x]) and math.isfinite(norms[y])) or norms[x] == 0.0 or norms[y] == 0.0:
                    continue
                value = float(g[x, y]) / math.sqrt(norms[x] * norms[y])
                for a, b in ((i, j), (j, i)):
                    self.pairs[a][b] += 1
                    self.cos_sum[a][b] += value
                    self.conflicts[a][b] += int(value < 0.0)

    def summary(self, *, mode: str, step: int, dro_weights: list[float] | None = None) -> dict:
        """The record written to ``group_objective.json``: per group its batches and mean loss (and GroupDRO
        weight), and per pair (PCGrad) the batches with both, the mean cosine and the share of them in conflict.
        Means of no batch are None."""
        n = len(self.groups)
        out: dict[str, Any] = {
            "mode": mode,
            "step": int(step),
            "groups": [
                {
                    "name": name,
                    "batches": self.batches[g],
                    "mean_loss": self.loss_sum[g] / self.batches[g] if self.batches[g] else None,
                    **({"dro_weight": dro_weights[g]} if dro_weights is not None else {}),
                }
                for g, name in enumerate(self.groups)
            ],
        }
        if mode == "pcgrad":
            out["agreement"] = {
                "pairs": [[self.pairs[i][j] for j in range(n)] for i in range(n)],
                "mean_cosine": [
                    [self.cos_sum[i][j] / self.pairs[i][j] if self.pairs[i][j] else None for j in range(n)]
                    for i in range(n)
                ],
                "conflict_share": [
                    [self.conflicts[i][j] / self.pairs[i][j] if self.pairs[i][j] else None for j in range(n)]
                    for i in range(n)
                ],
            }
        return out


class GroupObjective:
    """One run's group objective (module docstring): the per-group losses of a batch, the backward pass of GroupDRO
    or PCGrad, and the record of both.

    Args:
        mode: ``group_dro`` or ``pcgrad``.
        groups: the group names, in group order.
        dataset_groups: each config dataset's group name, None for one without.
        dro_step_size: GroupDRO's step size.
        seed: the run's seed; PCGrad's projection order is drawn from a generator seeded with it.
    """

    def __init__(self, mode: str, groups: Sequence[str], dataset_groups: Sequence[str | None], *,
                 dro_step_size: float, seed: int) -> None:
        if mode not in ("group_dro", "pcgrad"):
            raise ValueError(f"unknown group objective {mode!r}")
        self.mode = mode
        self.groups = list(groups)
        index = {name: i for i, name in enumerate(self.groups)}
        unknown = sorted({name for name in dataset_groups if name is not None and name not in index})
        if unknown:
            raise ValueError(f"objective groups {unknown} are not among {self.groups}")
        self.dataset_groups = [None if name is None else index[name] for name in dataset_groups]
        self.dro = GroupDRO(len(self.groups), dro_step_size) if mode == "group_dro" else None
        self.generator = torch.Generator().manual_seed(int(seed))
        self.record = GradientAgreement(self.groups)

    def check_datasets(self, dataset_indices: Sequence[int]) -> None:
        """Every dataset with a training patch has a group (a sample without one could not be placed).

        Raises:
            ValueError: the first dataset without one.
        """
        for value in sorted({int(v) for v in dataset_indices}):
            if self.dataset_groups[value] is None:
                raise ValueError(f"datasets[{value}] has training patches but no objective_group")

    def losses(self, loss_fn: LossFn, predictions: Any, targets: Any,
               dataset_index: torch.Tensor) -> dict[int, torch.Tensor]:
        """The batch's per-group losses (:func:`group_losses`), counted in the record."""
        groups = sample_groups(dataset_index.to(device="cpu"), self.dataset_groups)
        out = group_losses(loss_fn, predictions, targets, groups.to(dataset_index.device))
        for group, loss in out.items():
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss of group {self.groups[group]}")
        self.record.add_losses(out)
        return out

    def backward(self, losses: dict[int, torch.Tensor], parameters: Sequence[torch.nn.Parameter],
                 backward_fn: Callable[..., None]) -> None:
        """Set the parameters' gradients for the batch's ``losses``.

        GroupDRO: one backward pass of the weighted objective. PCGrad: one backward pass per group (``retain_graph``
        until the last), each group's gradient flattened, the pairs' cosines recorded, and the parameters' gradients
        set to :func:`pcgrad`'s step. ``backward_fn(loss, **kwargs)`` is the training's backward (accelerate's, which
        scales an fp16 loss: every group's gradient has the same scale, and the projection does not depend on it).

        Raises:
            ValueError: a parameter whose gradient was already set (the caller zeroes them first).
        """
        if self.dro is not None:
            backward_fn(self.dro.objective(losses))
            return
        params = [p for p in parameters if p.requires_grad]
        if any(p.grad is not None and bool(p.grad.any()) for p in params):
            raise ValueError("PCGrad needs zeroed gradients before the batch")
        order = sorted(losses)
        dtypes = {p.dtype for p in params}
        if len(dtypes) != 1:
            raise ValueError(f"PCGrad needs parameters of one dtype, got {sorted(map(str, dtypes))}")
        # One row per group, written in place, on the CPU (module docstring, ROCm second crash).
        stack = gradient_stack(len(order), sum(p.numel() for p in params), params[0].dtype)
        for k, group in enumerate(order):
            for p in params:
                p.grad = None
            backward_fn(losses[group], retain_graph=k < len(order) - 1)
            offset = 0
            for p in params:
                n = p.numel()
                if p.grad is not None:  # a parameter this group's loss does not reach keeps its zeros
                    stack[k, offset:offset + n].copy_(p.grad.reshape(-1))
                offset += n
            for p in params:  # free this group's gradients on the GPU before the next group's backward pass
                p.grad = None
        gram_matrix = gram(stack)
        self.record.add_gram(order, gram_matrix)
        combined = stack[0].clone() if len(order) == 1 else combine(stack, pcgrad_weights(gram_matrix, self.generator))
        del stack
        offset = 0
        for p in params:
            n = p.numel()
            p.grad = combined[offset:offset + n].view_as(p).to(device=p.device, dtype=p.dtype, copy=True)
            offset += n

    def summary(self, step: int) -> dict:
        """The record for ``group_objective.json`` (:meth:`GradientAgreement.summary`)."""
        return self.record.summary(mode=self.mode, step=step,
                                   dro_weights=self.dro.state() if self.dro is not None else None)
