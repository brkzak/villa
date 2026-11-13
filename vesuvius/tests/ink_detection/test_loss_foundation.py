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
