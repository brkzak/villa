"""GroupDRO, PCGrad and gradient agreement over groups of the data (``training.group_objectives``).

These objectives change what a training step does, so the tests pin their maths by hand-computed cases, their
determinism (PCGrad's seeded order), how a sample finds its group, the config's checks, and one real training entry
on synthetic data.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vesuvius.ink_detection.training.train as train_module
from vesuvius.ink_detection.config import TrainingConfig, resolve_training_mapping
from vesuvius.ink_detection.training.group_objectives import (
    GradientAgreement,
    GroupDRO,
    GroupObjective,
    gram,
    group_losses,
    pcgrad,
    sample_groups,
)
from vesuvius.ink_detection.training.train import stage_training_request

from .test_model_foundation import _config_mapping


def _grouped_mapping(mode: str = "pcgrad", groups=("A", "B")) -> dict:
    authored = _config_mapping(depth=1, side=4)
    authored.update({"num_iterations": 4, "out_dir": "/tmp/ink-output", "seed": 5})
    base = authored["datasets"][0]
    authored["datasets"] = [dict(base, objective_group=g) for g in groups]
    authored["group_objective"] = {"mode": mode}
    return authored


def _parse(authored: dict) -> TrainingConfig:
    return TrainingConfig.from_mapping(resolve_training_mapping(authored))


def test_config_names_groups_in_sorted_order_and_keeps_each_datasets_group():
    """The group order is the index every record uses; datasets keep their own group, None when they have none."""
    authored = _grouped_mapping(groups=("B", "A"))
    authored["datasets"].append(dict(authored["datasets"][0]))
    del authored["datasets"][2]["objective_group"]
    config = _parse(authored).group_objective
    assert config.mode == "pcgrad"
    assert config.groups == ("A", "B")
    assert config.dataset_groups == ("B", "A", None)
    assert config.dro_step_size == 0.01 and config.write_every == 50


def test_config_without_group_objective_is_unchanged():
    """A run without the setting trains exactly as before: no group objective at all."""
    authored = _config_mapping(depth=1, side=4)
    authored.update({"num_iterations": 4, "out_dir": "/tmp/ink-output", "seed": 5})
    assert _parse(authored).group_objective is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda a: a["group_objective"].update(mode="sum"), "mode must be one of"),
        (lambda a: a.update(datasets=a["datasets"][:1]), "two or more objective_groups"),
        (lambda a: a["group_objective"].update(dro_step_size=0), "dro_step_size must be positive"),
        (lambda a: a["group_objective"].update(write_every=0), "write_every must be >= 1"),
        (lambda a: a.update(group_objective=None), "group_objective is not set"),
        (lambda a: a.update(grad_acc_steps=2), "requires grad_acc_steps=1"),
    ],
)
def test_config_fails_loudly_on_a_wrong_group_objective(change, message):
    """A wrong setting must stop the run before it trains, never fall back to the plain loss."""
    authored = _grouped_mapping()
    change(authored)
    with pytest.raises(ValueError, match=message):
        _parse(authored)


def test_sample_groups_maps_datasets_and_refuses_an_ungrouped_sample():
    """A sample's group comes from its dataset; a sample that has none cannot be placed."""
    assert sample_groups(torch.tensor([2, 0, 1]), [1, 0, 1]).tolist() == [1, 1, 0]
    with pytest.raises(ValueError, match="no objective_group"):
        sample_groups(torch.tensor([1]), [0, None])
    with pytest.raises(ValueError, match="not one of"):
        sample_groups(torch.tensor([3]), [0, 1])


def test_group_losses_are_the_loss_of_each_groups_samples_alone_also_for_deep_supervision():
    """Each group's loss must see only its own samples, at every deep-supervision level."""
    def loss_fn(pred, target):
        if isinstance(pred, list):
            return sum(torch.mean((p - t) ** 2) for p, t in zip(pred, target))
        return torch.mean((pred - target) ** 2)

    pred = torch.tensor([[1.0], [2.0], [4.0]])
    target = torch.zeros(3, 1)
    groups = torch.tensor([1, 0, 1])
    out = group_losses(loss_fn, pred, target, groups)
    assert sorted(out) == [0, 1]
    assert float(out[0]) == 4.0 and float(out[1]) == pytest.approx((1 + 16) / 2)
    deep = group_losses(loss_fn, [pred, pred * 2], [target, target], groups)
    assert float(deep[0]) == 4.0 + 16.0


def test_group_dro_raises_the_weight_of_the_group_with_the_higher_loss():
    """Sagawa et al.'s update: q_g *= exp(eta * loss_g), normalised; the objective weighs the batch's groups by q."""
    dro = GroupDRO(3, step_size=1.0)
    losses = {0: torch.tensor(1.0), 1: torch.tensor(0.0)}
    objective = dro.objective(losses)
    q = torch.tensor([torch.e, 1.0, 1.0], dtype=torch.float64) / (torch.e + 2)
    assert dro.state() == pytest.approx(q.tolist())
    share0 = q[0] / (q[0] + q[1])
    assert float(objective) == pytest.approx(float(share0))
    # Group 2 was not in the batch: its weight only changes by the normalisation.
    assert dro.state()[2] == pytest.approx(1 / (torch.e + 2))


def test_pcgrad_projects_out_the_conflicting_part_only():
    """The paper's rule: g_i loses its component along g_j only when they conflict (negative dot product)."""
    gen = torch.Generator().manual_seed(0)
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])
    # a.b = -1: a -> a - (-1/2) b = [0.5, 0.5]; b -> b - (-1/1) a = [0, 1]. Mean: [0.25, 0.75].
    assert pcgrad([a, b], gen).tolist() == pytest.approx([0.25, 0.75])
    c = torch.tensor([1.0, 1.0])
    assert pcgrad([a, c], gen).tolist() == pytest.approx([1.0, 0.5])  # no conflict: the plain mean
    assert pcgrad([a], gen).tolist() == [1.0, 0.0]
    assert pcgrad([a, torch.zeros(2)], gen).tolist() == pytest.approx([0.5, 0.0])  # a zero gradient projects nothing


def test_pcgrad_is_reproducible_from_its_seed():
    """With three or more groups the projection order matters; the same seed must give the same step."""
    grads = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([-1.0, 1.0, 0.2]), torch.tensor([0.3, -1.0, 1.0])]
    first = [pcgrad(grads, torch.Generator().manual_seed(7)) for _ in range(2)]
    torch.testing.assert_close(first[0], first[1])


def test_agreement_records_mean_cosine_and_conflict_share_per_pair():
    """The agreement report is what B reads to find scrolls that look alike: its numbers must be exact."""
    rec = GradientAgreement(["A", "B", "C"])
    rec.add_gradients({0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0]), 2: torch.tensor([-1.0, 0.0])})
    rec.add_gradients({0: torch.tensor([1.0, 0.0]), 1: torch.tensor([1.0, 0.0])})
    rec.add_losses({0: torch.tensor(0.5), 2: torch.tensor(1.5)})
    out = rec.summary(mode="pcgrad", step=2)
    assert out["agreement"]["pairs"] == [[0, 2, 1], [2, 0, 1], [1, 1, 0]]
    assert out["agreement"]["mean_cosine"][0][1] == pytest.approx(0.5)
    assert out["agreement"]["mean_cosine"][0][2] == pytest.approx(-1.0)
    assert out["agreement"]["conflict_share"][0][2] == 1.0 and out["agreement"]["conflict_share"][0][1] == 0.0
    assert out["agreement"]["mean_cosine"][0][0] is None
    assert [g["mean_loss"] for g in out["groups"]] == [0.5, None, 1.5]


def test_group_objective_pcgrad_sets_the_combined_gradient_and_records_agreement():
    """The backward pass must leave exactly PCGrad's step in the parameters' gradients."""
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.5, -0.5]]))
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([[1.0], [1.0]])
    objective = GroupObjective("pcgrad", ["A", "B"], ["A", "B"], dro_step_size=0.01, seed=3)

    def loss_fn(pred, target):
        return torch.mean((pred - target) ** 2)

    losses = objective.losses(loss_fn, model(x), y, torch.tensor([0, 1]))
    objective.backward(losses, list(model.parameters()), lambda loss, **kw: loss.backward(**kw))
    # Per group: A's gradient [2(0.5-1), 0] = [-1, 0]; B's [0, 2(-0.5-1)] = [0, -3]. Orthogonal: the plain mean.
    assert model.weight.grad[0].tolist() == pytest.approx([-0.5, -1.5])
    assert objective.summary(1)["agreement"]["mean_cosine"][0][1] == pytest.approx(0.0)


def test_group_objective_refuses_a_dataset_with_training_patches_but_no_group():
    """Every training sample must belong to a group, or its loss would silently drop out."""
    objective = GroupObjective("group_dro", ["A", "B"], ["A", "B", None], dro_step_size=0.01, seed=0)
    objective.check_datasets([0, 1])
    with pytest.raises(ValueError, match=r"datasets\[2\]"):
        objective.check_datasets([0, 2])


class _GroupedDataset(torch.utils.data.Dataset):
    """Two training patches, one of each config dataset, that carry their dataset index when asked."""

    def __init__(self, config, *, do_augmentations, patches=None, segments=None, emit_dataset_index=False):
        del config, do_augmentations
        self.training_patches = [SimpleNamespace(segment=SimpleNamespace(dataset_idx=i)) for i in (0, 1)]
        self.validation_patches = []
        self.segments = list(segments or [])
        self._patches = self.training_patches if patches is None else list(patches)
        self.emit = emit_dataset_index

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = {
            "image": torch.full((1, 1, 4, 4), float(index + 1)),
            "inklabels": torch.zeros(1, 1, 4, 4),
            "supervision_mask": torch.ones(1, 1, 4, 4),
        }
        if self.emit:
            sample["dataset_index"] = torch.tensor(self._patches[index].segment.dataset_idx)
        return sample


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"ink": image[:, :1, 0] * self.value}


class _Loss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((prediction - target[:, :1]) ** 2)


@pytest.mark.parametrize("mode", ["group_dro", "pcgrad"])
def test_real_training_entry_writes_the_group_record(tmp_path: Path, monkeypatch, mode):
    """The training loop must pass each sample's dataset through, train, and write group_objective.json."""
    import vesuvius.ink_detection.data.dataset as dataset_module
    import vesuvius.ink_detection.models.model as model_module
    import vesuvius.ink_detection.training.losses as losses_module
    import vesuvius.ink_detection.training.optimizers as optimizers_module
    import vesuvius.ink_detection.training.samplers as samplers_module

    authored = _grouped_mapping(mode)
    authored.update({
        "num_iterations": 3, "out_dir": str(tmp_path / "output"), "mixed_precision": "no",
        "dataloader_workers": 0, "pin_memory": False, "val_every": 99, "save_every": 3, "log_every": 99,
        "batch_size": 2, "scheduler": {"name": "cosine_annealing"},
    })
    authored["group_objective"]["write_every"] = 2
    authored["model_config"]["pretrained_backbone"] = "synthetic"
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(authored), encoding="utf-8")
    monkeypatch.setattr(dataset_module, "InkDataset", _GroupedDataset)
    monkeypatch.setattr(model_module, "make_model", lambda config: _Model())
    monkeypatch.setattr(losses_module, "create_loss", lambda config: _Loss())
    monkeypatch.setattr(optimizers_module, "create_training_optimizer",
                        lambda model, config: torch.optim.SGD(model.parameters(), lr=0.01))
    monkeypatch.setattr(samplers_module, "build_sampling_policy", lambda patches, config, batch_size: SimpleNamespace(
        batch_sampler=None, shuffle=False, sampler=None, generator=None, audit={}))

    assert train_module._run_training(stage_training_request(config_path)) == 0

    record = json.loads((tmp_path / "output" / "group_objective.json").read_text())
    assert record["mode"] == mode and record["step"] == 3
    assert [g["name"] for g in record["groups"]] == ["A", "B"]
    assert [g["batches"] for g in record["groups"]] == [3, 3]
    if mode == "pcgrad":
        # Both groups' gradients move the one weight the same way: they agree in every batch.
        assert record["agreement"]["pairs"][0][1] == 3
        assert record["agreement"]["mean_cosine"][0][1] == pytest.approx(1.0)
    else:
        assert sum(g["dro_weight"] for g in record["groups"]) == pytest.approx(1.0)
        # B's tiles are brighter, so its loss is higher and its weight grows.
        assert record["groups"][1]["dro_weight"] > record["groups"][0]["dro_weight"]


def _reference_pcgrad(grads, generator):
    """The first PCGrad (2026-11-04): the paper's loop on the full gradients, every dot product on them."""
    if len(grads) == 1:
        return grads[0].clone()
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


@pytest.mark.parametrize("k", [2, 3, 8])
def test_pcgrad_on_the_gram_matrix_gives_the_papers_step(k):
    """PCGrad now projects on the k x k Gram matrix (fewer GPU ops; the ROCm SIGFPE of 2026-11-05). The step and
    the generator's draws must stay those of the paper's loop on the full gradients, including a zero gradient."""
    data = torch.Generator().manual_seed(k)
    grads = [torch.randn(257, generator=data) for _ in range(k)]
    grads[-1] = -0.7 * grads[0] + 0.1 * grads[-1]  # a strong conflict
    if k > 2:
        grads[1] = torch.zeros(257)
    new_gen, ref_gen = torch.Generator().manual_seed(11), torch.Generator().manual_seed(11)
    torch.testing.assert_close(pcgrad(grads, new_gen), _reference_pcgrad(grads, ref_gen), rtol=1e-5, atol=1e-6)
    # The same number of draws: the next step's order is unchanged too.
    assert torch.randperm(5, generator=new_gen).tolist() == torch.randperm(5, generator=ref_gen).tolist()


def test_gram_is_the_same_in_chunks():
    """The Gram matrix is summed over column chunks to bound memory; the chunk size must not change it."""
    stack = torch.randn(4, 1003, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(gram(stack, chunk=100), (stack.double() @ stack.double().T))


def test_group_losses_take_samples_by_slices_not_an_index_tensor():
    """A group's samples are taken by basic slices: the same values, and a backward without index_put_ (the GPU path
    the plain training never runs; module docstring, ROCm)."""
    x = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2).requires_grad_()
    groups = torch.tensor([1, 0, 1, 1, 0, 1])
    seen = {}

    def loss_fn(pred, target):
        seen[len(seen)] = pred
        return (pred * target).sum()

    losses = group_losses(loss_fn, x, x.detach(), groups)
    torch.testing.assert_close(seen[0], x[[1, 4]])
    torch.testing.assert_close(seen[1], x[[0, 2, 3, 5]])

    def ops(fn, out):
        if fn is None or fn in out:
            return out
        out.add(fn)
        for nxt, _ in fn.next_functions:
            ops(nxt, out)
        return out

    names = {type(fn).__name__ for loss in losses.values() for fn in ops(loss.grad_fn, set())}
    assert not any("Index" in name for name in names), names
    sum(losses.values()).backward()
    torch.testing.assert_close(x.grad, x.detach())


def test_pcgrad_backward_with_a_parameter_one_group_does_not_reach():
    """A parameter outside one group's graph keeps a zero row in that group's gradient, as before."""
    a, b = nn.Parameter(torch.tensor([1.0])), nn.Parameter(torch.tensor([2.0]))
    objective = GroupObjective("pcgrad", ["A", "B"], ["A", "B"], dro_step_size=0.01, seed=0)
    losses = {0: (a * 3.0).sum(), 1: (b * 5.0).sum()}
    objective.backward(losses, [a, b], lambda loss, **kw: loss.backward(**kw))
    assert a.grad.tolist() == [1.5] and b.grad.tolist() == [2.5]
