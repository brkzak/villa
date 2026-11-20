"""Numerical witnesses for smoothing, masks, weighting, metrics, and clDice."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.training.losses import (
    CompositeLoss,
    LabelSmoothedDCAndBCELoss,
    WeightedLossTerm,
    create_loss,
)

from .test_model_foundation import _config_mapping


def _smoothed_bce_loss(*, weight_dice: float = 0.0):
    return LabelSmoothedDCAndBCELoss(
        bce_kwargs={},
        soft_dice_kwargs={"label_smoothing": 0.0},
        weight_dice=weight_dice,
        weight_ce=1.0,
        use_ignore_label=True,
        bce_label_smoothing=0.5,
    )


def test_half_epsilon_maps_binary_bce_targets_to_exact_quarters():
    loss = _smoothed_bce_loss()

    smoothed = loss._smooth_bce_targets(torch.tensor([0.0, 1.0]))

    torch.testing.assert_close(
        smoothed,
        torch.tensor([0.25, 0.75]),
        rtol=0.0,
        atol=0.0,
    )


def test_smoothed_bce_matches_hand_calculated_quarter_target_value():
    loss = _smoothed_bce_loss()
    logit = math.log(3.0)
    logits_BCHW = torch.tensor([[[[logit]]]])
    target_with_ignore_BCHW = torch.tensor([[[[0.0]], [[0.0]]]])

    actual = loss(logits_BCHW, target_with_ignore_BCHW)
    expected = math.log(4.0) - 0.25 * math.log(3.0)

    torch.testing.assert_close(
        actual,
        torch.tensor(expected),
        rtol=1e-6,
        atol=1e-7,
    )


def test_ignore_channel_removes_value_and_gradient_at_ignored_pixels():
    loss = _smoothed_bce_loss()
    logits_BCHW = torch.zeros((1, 1, 1, 2), requires_grad=True)
    target_with_ignore_BCHW = torch.tensor(
        [[[[0.0, 1.0]], [[0.0, 1.0]]]]
    )

    value = loss(logits_BCHW, target_with_ignore_BCHW)
    value.backward()

    torch.testing.assert_close(
        value,
        torch.tensor(math.log(2.0)),
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        logits_BCHW.grad,
        torch.tensor([[[[0.25, 0.0]]]]),
        rtol=0.0,
        atol=0.0,
    )


def test_combined_smoothed_bce_and_dice_matches_all_valid_literal():
    loss = _smoothed_bce_loss(weight_dice=0.25)
    logits_BCHW = torch.tensor(
        [[[[math.log(3.0), -math.log(3.0)]]]],
        requires_grad=True,
    )
    target_with_ignore_BCHW = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 0.0]]]]
    )

    actual = loss(logits_BCHW, target_with_ignore_BCHW)
    expected = math.log(4.0 / 3.0) + 0.25 * math.log(3.0) - 5.0 / 24.0

    assert expected == pytest.approx(0.3540018113, abs=5e-11)
    torch.testing.assert_close(
        actual,
        torch.tensor(expected),
        rtol=1e-6,
        atol=5e-8,
    )


def test_combined_smoothed_bce_and_dice_excludes_ignored_second_pixel():
    loss = _smoothed_bce_loss(weight_dice=0.25)
    logits_BCHW = torch.tensor(
        [[[[math.log(3.0), -math.log(3.0)]]]],
        requires_grad=True,
    )
    target_with_ignore_BCHW = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]]]]
    )

    actual = loss(logits_BCHW, target_with_ignore_BCHW)
    actual.backward()
    expected = math.log(4.0 / 3.0) + 0.25 * math.log(3.0) - 5.0 / 22.0

    assert expected == pytest.approx(0.3350624173, abs=5e-11)
    torch.testing.assert_close(
        actual,
        torch.tensor(expected),
        rtol=1e-6,
        atol=5e-8,
    )
    assert logits_BCHW.grad[0, 0, 0, 1].item() == 0.0


class _LiteralLoss(nn.Module):
    def __init__(self, value: float, auxiliary: float | None = None) -> None:
        super().__init__()
        self.value = value
        self.auxiliary = auxiliary

    def forward(self, net_output, target):
        value = net_output.sum() * 0.0 + self.value
        if self.auxiliary is None:
            return value
        return value, {"Edge Score": value * 0.0 + self.auxiliary}


def test_composite_applies_ordered_weights_and_records_literal_metrics():
    loss = CompositeLoss(
        [
            WeightedLossTerm("first", 2.0, _LiteralLoss(1.5), "Same Name"),
            WeightedLossTerm("second", -0.5, _LiteralLoss(4.0, 7.0), "Same Name"),
        ]
    )

    total = loss(torch.ones(1), torch.zeros(1))

    torch.testing.assert_close(total, torch.tensor(1.0))
    assert loss.latest_metrics == {
        "loss_terms/same_name_raw": 1.5,
        "loss_terms/same_name_weighted": 3.0,
        "loss_terms/same_name_1_raw": 4.0,
        "loss_terms/same_name_1_weighted": -2.0,
        "loss_aux/same_name_1/edge_score": 7.0,
        "loss/total": 1.0,
    }


def test_configured_composite_retains_term_weights_and_smoothing():
    authored = _config_mapping()
    authored["loss"] = {
        "terms": [
            {
                "name": "LabelSmoothedDCAndBCELoss",
                "metric_name": "ink base",
                "weight": 1.75,
                "weight_dice": 0.0,
                "weight_ce": 0.5,
                "bce_label_smoothing": 0.5,
            }
        ]
    }

    loss = create_loss(InkConfig.from_mapping(authored))

    assert loss.terms[0].weight == 1.75
    assert loss.terms[0].metric_name == "ink base"
    assert loss.terms[0].module.weight_ce == 0.5
    assert loss.terms[0].module.bce_label_smoothing == 0.5


def _class_weighted_bce(ink: float, background: float):
    return LabelSmoothedDCAndBCELoss(
        bce_kwargs={},
        soft_dice_kwargs={"label_smoothing": 0.0},
        weight_dice=0.0,
        weight_ce=1.0,
        use_ignore_label=True,
        bce_label_smoothing=0.5,
        bce_ink_weight=ink,
        bce_background_weight=background,
    )


def test_unit_class_weights_leave_the_loss_bit_for_bit_unchanged():
    """Weights of 1.0 must be the old loss exactly: every run before the weights existed trained with it."""
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 4, 4)
    target = torch.cat(
        [(torch.rand(2, 1, 4, 4) > 0.7).float(), (torch.rand(2, 1, 4, 4) > 0.8).float()], dim=1
    )

    plain = _smoothed_bce_loss()(logits, target)
    weighted = _class_weighted_bce(1.0, 1.0)(logits, target)

    assert torch.equal(plain, weighted)


def test_class_weights_scale_each_pixel_by_its_label_over_the_supervised_mean():
    """An ink pixel's BCE counts ink_weight times, a background pixel's background_weight times, and the mean stays
    over the supervised pixels (an ignored pixel adds nothing, whatever its label)."""
    logits = torch.tensor([[[[0.3, -1.2, 2.0]]]])
    labels = torch.tensor([[[[1.0, 0.0, 1.0]]]])
    ignore = torch.tensor([[[[0.0, 0.0, 1.0]]]])
    target = torch.cat([labels, ignore], dim=1)

    actual = _class_weighted_bce(3.0, 2.0)(logits, target)

    t = labels * 0.5 + 0.25
    pixel = nn.functional.binary_cross_entropy_with_logits(logits, t, reduction="none")
    expected = (3.0 * pixel[..., 0] + 2.0 * pixel[..., 1]).sum() / 2.0
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_class_weights_keep_the_best_prediction_at_the_smoothed_target():
    """Unlike torch's pos_weight, a per-pixel weight must not move a background pixel's optimum off 0.25: the
    gradient at p = t is zero for any weight."""
    logit = torch.tensor([[[[math.log(0.25 / 0.75)]]]], requires_grad=True)
    target = torch.tensor([[[[0.0]], [[0.0]]]])

    _class_weighted_bce(3.0, 5.0)(logit, target).backward()

    assert abs(float(logit.grad)) < 1e-6


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_class_weights_reject_values_that_are_not_positive_and_finite(value):
    """A zero, negative or infinite weight would silently drop or blow up a class: refuse it at construction."""
    with pytest.raises(ValueError):
        _class_weighted_bce(value, 1.0)
    with pytest.raises(ValueError):
        _class_weighted_bce(1.0, value)


def test_configured_class_weights_reach_the_loss_from_the_top_level_and_the_term():
    """A config's top-level bce_ink_weight / bce_background_weight apply to every term; a term's own wins."""
    authored = _config_mapping()
    authored["bce_ink_weight"] = 3.0
    authored["bce_background_weight"] = 2.0
    authored["loss"] = {
        "terms": [
            {"name": "LabelSmoothedDCAndBCELoss", "metric_name": "a"},
            {"name": "LabelSmoothedDCAndBCELoss", "metric_name": "b", "bce_background_weight": 4.0},
        ]
    }

    loss = create_loss(InkConfig.from_mapping(authored))

    assert loss.terms[0].module.bce_ink_weight == 3.0
    assert loss.terms[0].module.bce_background_weight == 2.0
    assert loss.terms[1].module.bce_background_weight == 4.0


def test_configs_without_class_weights_train_unweighted():
    """A config written before the weights existed has none, and must build the unweighted loss."""
    loss = create_loss(InkConfig.from_mapping(_config_mapping()))

    assert loss.terms[0].module.bce_ink_weight == 1.0
    assert loss.terms[0].module.bce_background_weight == 1.0


def _pu_bce(prior: float | None, *, smoothing: float = 0.5, ink: float = 1.0, background: float = 1.0,
            weight_dice: float = 0.0):
    return LabelSmoothedDCAndBCELoss(
        bce_kwargs={},
        soft_dice_kwargs={"label_smoothing": 0.0},
        weight_dice=weight_dice,
        weight_ce=1.0,
        use_ignore_label=True,
        bce_label_smoothing=smoothing,
        bce_ink_weight=ink,
        bce_background_weight=background,
        pu_prior=prior,
    )


def _pu_target(labels, ignore):
    return torch.cat([torch.tensor(labels).reshape(1, 1, 1, -1), torch.tensor(ignore).reshape(1, 1, 1, -1)], dim=1)


def test_pu_loss_matches_the_hand_computed_nnpu_risk():
    """The PU loss must be the nnPU risk of the class docstring: labelled pixels at their own BCE, every unlabeled
    pixel at prior * mean_P l+ + |mean_U l- - prior * mean_P l-|, all over |P| + |N| + |U|."""
    logits = torch.tensor([[[[1.5, 0.4, -2.0, 0.3, -0.7, 2.2]]]])
    # P, P, N, U, U, ignored (a plain ignored pixel: target 0, not 0.5)
    target = _pu_target([1.0, 1.0, 0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    prior = 0.3

    actual = _pu_bce(prior, smoothing=0.0, ink=2.0, background=1.5)(logits, target)

    bce = nn.functional.binary_cross_entropy_with_logits
    x = logits.flatten()
    l_ink = bce(x, torch.ones_like(x), reduction="none") * 2.0
    l_pap = bce(x, torch.zeros_like(x), reduction="none") * 1.5
    labelled = l_ink[0] + l_ink[1] + l_pap[2]
    negative = l_pap[3:5].mean() - prior * l_pap[0:2].mean()
    risk_u = prior * l_ink[0:2].mean() + negative.abs()
    expected = (labelled + 2 * risk_u) / 5
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_pu_loss_reverses_the_gradient_of_a_negative_risk_estimate():
    """When U is predicted far less ink than the prior allows (the estimate of its not-ink loss below 0), the loss
    must push U's predictions up (nnPU's defence), not further down."""
    logits = torch.tensor([[[[3.0, -6.0, -6.0]]]], requires_grad=True)
    target = _pu_target([1.0, 0.5, 0.5], [0.0, 1.0, 1.0])

    _pu_bce(0.9, smoothing=0.0)(logits, target).backward()

    assert float(logits.grad[0, 0, 0, 1]) < 0  # descent raises the U logit
    assert float(logits.grad[0, 0, 0, 2]) < 0


def test_pu_loss_without_unlabeled_or_positive_pixels_is_the_plain_loss():
    """A batch without U pixels, or without P pixels to estimate from, must train exactly as without PU."""
    torch.manual_seed(1)
    logits = torch.randn(1, 1, 1, 4)
    no_u = _pu_target([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0])
    no_p = _pu_target([0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0])
    no_p_plain = _pu_target([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0])

    assert torch.equal(_pu_bce(0.4)(logits, no_u), _pu_bce(None)(logits, no_u))
    assert torch.equal(_pu_bce(0.4)(logits, no_p), _pu_bce(None)(logits, no_p_plain))


def test_pu_loss_keeps_unlabeled_pixels_out_of_dice():
    """Dice must see only labelled pixels: an unlabeled pixel's prediction must not move the Dice term."""
    logits_a = torch.tensor([[[[2.0, -2.0, 5.0]]]])
    logits_b = torch.tensor([[[[2.0, -2.0, -5.0]]]])
    target = _pu_target([1.0, 0.0, 0.5], [0.0, 0.0, 1.0])
    dice_only = LabelSmoothedDCAndBCELoss(bce_kwargs={}, soft_dice_kwargs={"label_smoothing": 0.0}, weight_dice=1.0,
                                          weight_ce=0.0, use_ignore_label=True, pu_prior=0.5)

    assert torch.equal(dice_only(logits_a, target), dice_only(logits_b, target))


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_pu_prior_must_be_strictly_between_zero_and_one(value):
    """A prior of 0 or 1 says U is all papyrus or all ink: no PU problem, and the risk degenerates. Refuse it."""
    with pytest.raises(ValueError):
        _pu_bce(value)


def test_configured_pu_prior_reaches_the_loss_and_sets_the_unlabeled_label():
    """A top-level pu_prior reaches every term, and the config then names the unlabeled label (128 by default)."""
    authored = _config_mapping()
    authored["pu_prior"] = 0.35

    config = InkConfig.from_mapping(authored)
    loss = create_loss(config)

    assert loss.terms[0].module.pu_prior == 0.35
    assert config.loss.pu_unlabeled_label == 128


def test_configs_without_pu_prior_train_without_pu():
    """A config written before the PU loss existed has no prior and no unlabeled label: the loss as before."""
    config = InkConfig.from_mapping(_config_mapping())

    assert create_loss(config).terms[0].module.pu_prior is None
    assert config.loss.pu_unlabeled_label is None


@pytest.mark.parametrize("value", [0, 255, 12.5, True])
def test_pu_unlabeled_label_must_be_neither_no_ink_nor_ink(value):
    """The unlabeled label must be an integer strictly between 0 (no ink) and 255 (ink)."""
    authored = _config_mapping()
    authored["pu_prior"] = 0.3
    authored["pu_unlabeled_label"] = value
    with pytest.raises(ValueError):
        InkConfig.from_mapping(authored)


def test_pu_unlabeled_label_without_a_prior_is_refused():
    """An unlabeled label without a prior would silently train those pixels as plain ink: refuse it."""
    authored = _config_mapping()
    authored["pu_unlabeled_label"] = 100
    with pytest.raises(ValueError):
        InkConfig.from_mapping(authored)


def test_prepare_loss_inputs_marks_unlabeled_pixels_ignored_with_the_pu_target():
    """In flat mode a supervised pixel whose ink label is the unlabeled value in every inked layer is ignored with
    target 0.5; a layer with real ink makes it ink; an unsupervised one stays plainly ignored; without PU nothing
    changes."""
    from vesuvius.ink_detection.training.train import prepare_loss_inputs

    # (B, C, Z, H, W): 2 layers, 4 pixels: U, ink+U layers, papyrus, U-valued but unsupervised
    ink = torch.tensor([[[[[128.0, 255.0, 0.0, 128.0]], [[128.0, 128.0, 0.0, 0.0]]]]])
    sup = torch.tensor([[[[[255.0, 255.0, 255.0, 0.0]], [[255.0, 255.0, 255.0, 0.0]]]]])
    batch = {"inklabels": ink, "supervision_mask": sup}
    predictions = torch.zeros(1, 1, 1, 4)

    _, targets, ignore = prepare_loss_inputs(predictions, batch, mode="flat", pu_unlabeled_label=128)
    _, plain_targets, plain_ignore = prepare_loss_inputs(predictions, batch, mode="flat")

    assert targets.flatten().tolist() == [0.5, 1.0, 0.0, 1.0]
    assert ignore.flatten().tolist() == [1.0, 0.0, 0.0, 1.0]
    assert plain_targets.flatten().tolist() == [1.0, 1.0, 0.0, 1.0]
    assert plain_ignore.flatten().tolist() == [0.0, 0.0, 0.0, 1.0]


def test_prepare_loss_inputs_refuses_pu_in_full_3d():
    """Only the flat read marks unlabeled pixels; a 3D mode with PU must fail rather than train them as ink."""
    from vesuvius.ink_detection.training.train import prepare_loss_inputs

    batch = {"inklabels": torch.zeros(1, 1, 2, 2, 2), "supervision_mask": torch.ones(1, 1, 2, 2, 2),
             "image": torch.zeros(1, 1, 2, 2, 2)}
    with pytest.raises(ValueError):
        prepare_loss_inputs(torch.zeros(1, 1, 2, 2, 2), batch, mode="full_3d", pu_unlabeled_label=128)


def test_pu_labels_reach_the_configured_loss_through_the_training_glue():
    """From label zarr values (ink 255 / unlabeled 128) through prepare_loss_inputs and the ignore concatenation to
    create_loss's term, a PU config must give the hand-computed PU loss and the same config without PU must not."""
    from vesuvius.ink_detection.training.deep_supervision import concatenate_deep_supervision_ignore
    from vesuvius.ink_detection.training.train import prepare_loss_inputs

    authored = _config_mapping()
    authored.update({"pu_prior": 0.3, "bce_label_smoothing": 0.0, "loss": {"dice_weight": 0.0}})
    config = InkConfig.from_mapping(authored)
    ink = torch.tensor([[[[[255.0, 128.0, 128.0, 0.0]]]]])  # P, U, U, N
    batch = {"inklabels": ink, "supervision_mask": torch.full_like(ink, 255.0)}
    logits = torch.tensor([[[[0.8, 0.2, -1.0, -1.5]]]])

    _, targets, ignore = prepare_loss_inputs(logits, batch, mode="flat",
                                             pu_unlabeled_label=config.loss.pu_unlabeled_label)
    actual = create_loss(config)(logits, concatenate_deep_supervision_ignore(targets, ignore, logits))

    bce = nn.functional.binary_cross_entropy_with_logits
    x = logits.flatten()
    l_ink, l_pap = bce(x, torch.ones_like(x), reduction="none"), bce(x, torch.zeros_like(x), reduction="none")
    risk_u = 0.3 * l_ink[0] + (l_pap[1:3].mean() - 0.3 * l_pap[0]).abs()
    torch.testing.assert_close(actual, (l_ink[0] + l_pap[3] + 2 * risk_u) / 4, rtol=1e-6, atol=1e-7)
