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


def _select(value: Any, index: torch.Tensor) -> Any:
    """``value`` (a tensor, or a list or tuple of them for deep supervision) with only the samples ``index``."""
    if isinstance(value, (list, tuple)):
        return type(value)(item[index] for item in value)
    return value[index]


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
    for group in sorted(set(groups.tolist())):
        index = (groups == group).nonzero(as_tuple=True)[0]
        out[int(group)] = loss_fn(_select(predictions, index), _select(targets, index))
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


def pcgrad(grads: Sequence[torch.Tensor], generator: torch.Generator) -> torch.Tensor:
    """PCGrad's step from the groups' flat gradients (module docstring): the mean of each gradient after its
    conflicting parts are projected out.

    Each gradient g_i is projected against every other original gradient g_j, in an order drawn from
    ``generator``: when ``g_i . g_j < 0``, ``g_i -= (g_i . g_j) / |g_j|^2 * g_j``. A zero gradient takes no part as
    the other side of a projection.

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
    # Dot products in float64: a large model's sums in float32 can flip the sign of a near-zero conflict.
    norms = [float(torch.dot(g.double(), g.double())) for g in grads]
    out = torch.zeros_like(grads[0])
    for i, g in enumerate(grads):
        projected = g.clone()
        for j in torch.randperm(len(grads), generator=generator).tolist():
            if j == i or norms[j] == 0.0:
                continue
            dot = float(torch.dot(projected.double(), grads[j].double()))
            if dot < 0.0:
                projected -= (dot / norms[j]) * grads[j]
        out += projected
    return out / len(grads)


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
        for x, i in enumerate(present):
            for j in present[x + 1:]:
                value = cosine(grads[i], grads[j])
                if value is None:
                    continue
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
        flat: dict[int, torch.Tensor] = {}
        order = sorted(losses)
        for k, group in enumerate(order):
            for p in params:
                p.grad = None
            backward_fn(losses[group], retain_graph=k < len(order) - 1)
            flat[group] = torch.cat([
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params
            ])
        self.record.add_gradients({g: v for g, v in flat.items() if bool(torch.isfinite(v).all())})
        combined = pcgrad([flat[g] for g in order], self.generator)
        offset = 0
        for p in params:
            n = p.numel()
            p.grad = combined[offset:offset + n].view_as(p).to(dtype=p.dtype).clone()
            offset += n

    def summary(self, step: int) -> dict:
        """The record for ``group_objective.json`` (:meth:`GradientAgreement.summary`)."""
        return self.record.summary(mode=self.mode, step=step,
                                   dro_weights=self.dro.state() if self.dro is not None else None)
