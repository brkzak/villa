"""Ink segmentation objectives and scalar weighting."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from vesuvius.ink_detection.config import InkConfig
from vesuvius.models.training.loss.nnunet_losses import DC_and_BCE_loss

# The target value of an unlabeled pixel for the PU loss (LabelSmoothedDCAndBCELoss, PU loss). train.prepare_loss_inputs
# writes it with the pixel's ignore flag set, so every other user of the targets (Dice, metrics, previews) leaves the
# pixel out as before; 0.5 survives the nearest-neighbour downsampling of deep supervision and is never a hard label.
PU_UNLABELED_TARGET = 0.5


class LabelSmoothedDCAndBCELoss(DC_and_BCE_loss):
    """Combine Dice and BCE with optional smoothing of BCE targets.

    ``bce_ink_weight`` and ``bce_background_weight`` scale each supervised pixel's BCE by its label: ink pixels by the
    first, labelled background pixels by the second (a soft label in between is weighted linearly). The mean stays
    over the supervised pixels, so weights above 1 also make the BCE term larger. They weight pixels by their label,
    not the two terms of the BCE as ``torch.nn.BCEWithLogitsLoss(pos_weight=...)`` does: with smoothed targets
    (0.25 / 0.75) every pixel carries both terms, and a ``pos_weight`` of 3 would move the best prediction on a
    background pixel from 0.25 to 0.5. A per-pixel weight keeps each pixel's best prediction at its smoothed target
    and only changes how much the errors of one class count against the other. The Dice term is not weighted. Both
    1.0 (the default) is the unweighted loss, bit for bit.

    PU loss (``pu_prior``, positive-unlabeled learning): a pixel can be labelled ink (P), labelled not ink (N), or
    unlabeled (U): ignored, with the target :data:`PU_UNLABELED_TARGET` (``train.prepare_loss_inputs``). Of U, a share
    ``pu_prior`` is ink, but which pixels is unknown. The BCE then counts every U pixel at the non-negative PU risk
    of Kiryo et al. 2017 (nnPU), taken over the whole batch:

        R_U = prior * mean_P l+  +  |mean_U l-  -  prior * mean_P l-|

    ``l+`` / ``l-`` is a pixel's BCE against the (smoothed) ink / not-ink target, times the ink / background weight.
    ``mean_U l- - prior * mean_P l-`` estimates the loss of U's not-ink pixels from U and P alone; it assumes the
    labelled ink looks like the ink hidden in U. When the model overfits, that estimate falls below 0 (it cannot be in
    truth); the absolute value then reverses its gradient (nnPU's defence with beta 0, gamma 1) instead of driving it
    lower. The BCE is ``(sum over P and N of the pixel BCE + |U| * R_U) / (|P| + |N| + |U|)``, so a U pixel weighs as
    much as a labelled one. A batch without a P pixel or without a U pixel has no estimate; its U pixels then add
    nothing and the BCE is the labelled pixels' mean, as without PU. The Dice term never sees U. None (the default)
    is the loss without PU, bit for bit.
    """

    def __init__(
        self,
        *args,
        bce_label_smoothing: float = 0.0,
        bce_ink_weight: float = 1.0,
        bce_background_weight: float = 1.0,
        pu_prior: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= bce_label_smoothing <= 1.0:
            raise ValueError(
                f"bce_label_smoothing must be between 0 and 1, got {bce_label_smoothing}"
            )
        for name, value in (
            ("bce_ink_weight", bce_ink_weight),
            ("bce_background_weight", bce_background_weight),
        ):
            if not (math.isfinite(float(value)) and float(value) > 0.0):
                raise ValueError(f"{name} must be a finite number above 0, got {value}")
        if pu_prior is not None and not (math.isfinite(float(pu_prior)) and 0.0 < float(pu_prior) < 1.0):
            raise ValueError(f"pu_prior must be strictly between 0 and 1, got {pu_prior}")
        if pu_prior is not None and not self.use_ignore_label:
            raise ValueError("pu_prior needs the ignore channel (use_ignore_label): it carries the unlabeled pixels")
        self.bce_label_smoothing = float(bce_label_smoothing)
        self.bce_ink_weight = float(bce_ink_weight)
        self.bce_background_weight = float(bce_background_weight)
        self.pu_prior = None if pu_prior is None else float(pu_prior)

    def _pu_bce_sum(
        self, net_output: torch.Tensor, target_regions: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The U pixels' part of the BCE (class docstring, PU loss): ``(|U| * R_U, |U|)``, both 0 when the batch has
        no P or no U pixel."""
        unlabeled = (~mask) & (target_regions == PU_UNLABELED_TARGET)
        positive = mask & (target_regions == 1)
        # Counts stay tensors (no host sync per step); a missing P or U zeroes the result through ``present``.
        n_unlabeled = unlabeled.sum().to(net_output.dtype)
        n_positive = positive.sum().to(net_output.dtype)
        present = ((n_unlabeled > 0) & (n_positive > 0)).to(net_output.dtype)
        n_unlabeled_safe = torch.clip(n_unlabeled, min=1.0)
        n_positive_safe = torch.clip(n_positive, min=1.0)
        ink_target = self._smooth_bce_targets(torch.ones_like(target_regions))
        papyrus_target = self._smooth_bce_targets(torch.zeros_like(target_regions))
        loss_as_ink = self.ce(net_output, ink_target) * self.bce_ink_weight
        loss_as_papyrus = self.ce(net_output, papyrus_target) * self.bce_background_weight
        positive_f, unlabeled_f = positive.float(), unlabeled.float()
        risk_p_ink = (loss_as_ink * positive_f).sum() / n_positive_safe
        risk_p_papyrus = (loss_as_papyrus * positive_f).sum() / n_positive_safe
        risk_u_papyrus = (loss_as_papyrus * unlabeled_f).sum() / n_unlabeled_safe
        negative = risk_u_papyrus - self.pu_prior * risk_p_papyrus
        risk_u = self.pu_prior * risk_p_ink + negative.abs()
        return risk_u * n_unlabeled * present, n_unlabeled * present

    def _class_weights(self, target_regions: torch.Tensor) -> torch.Tensor | None:
        """Per-pixel BCE weights from the unsmoothed label (None when both weights are 1: no multiply at all)."""
        if self.bce_ink_weight == 1.0 and self.bce_background_weight == 1.0:
            return None
        target = target_regions.float()
        return self.bce_background_weight + (self.bce_ink_weight - self.bce_background_weight) * target

    def _smooth_bce_targets(self, target_regions: torch.Tensor) -> torch.Tensor:
        target_regions = target_regions.float()
        if self.bce_label_smoothing == 0.0:
            return target_regions
        return target_regions * (1.0 - self.bce_label_smoothing) + 0.5 * self.bce_label_smoothing

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor = None,
    ):
        if self.use_ignore_label:
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = loss_mask

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        smoothed_targets = self._smooth_bce_targets(target_regions)
        weights = self._class_weights(target_regions)
        if mask is not None:
            pixel_ce = self.ce(net_output, smoothed_targets)
            if weights is not None:
                pixel_ce = pixel_ce * weights
            if self.pu_prior is None:
                denom = torch.clip(mask.sum(), min=1e-8)
                ce_loss = (pixel_ce * mask).sum() / denom
            else:
                unlabeled_sum, n_unlabeled = self._pu_bce_sum(net_output, target_regions, mask)
                denom = torch.clip(mask.sum() + n_unlabeled, min=1e-8)
                ce_loss = ((pixel_ce * mask).sum() + unlabeled_sum) / denom
        elif weights is not None:
            raise ValueError("bce_ink_weight / bce_background_weight need per-pixel BCE (use_ignore_label or a mask)")
        else:
            ce_loss = self.ce(net_output, smoothed_targets)
        return self.weight_ce * ce_loss + self.weight_dice * dc_loss


class WeightedLossTerm(nn.Module):
    """A named scalar-weighted objective and its metric label."""

    def __init__(
        self,
        name: str,
        weight: float,
        module: nn.Module,
        metric_name: str | None = None,
    ) -> None:
        super().__init__()
        self.name = str(name)
        self.weight = float(weight)
        self.module = module
        self.metric_name = str(metric_name or name)


def _sanitize_metric_name(name: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in str(name))
    cleaned = cleaned.strip("_")
    return cleaned or "term"


def _split_loss_output(
    output: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(output, tuple):
        if len(output) != 2:
            raise ValueError(
                f"Expected (loss, aux_dict) tuple, got tuple of length {len(output)}"
            )
        loss_value, auxiliary = output
        if not isinstance(auxiliary, dict):
            raise TypeError(
                f"Expected aux metrics dict, got {type(auxiliary).__name__}"
            )
        return loss_value.reshape(()), auxiliary
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected loss tensor, got {type(output).__name__}")
    return output.reshape(()), {}


class CompositeLoss(nn.Module):
    """Sum weighted terms and expose their detached scalar metrics."""

    def __init__(self, terms: list[WeightedLossTerm]) -> None:
        super().__init__()
        if not terms:
            raise ValueError("CompositeLoss requires at least one term")
        self.terms = nn.ModuleList(terms)
        self.latest_metrics: dict[str, float] = {}

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        total = net_output.new_zeros(())
        metrics: dict[str, float] = {}
        used_metric_names: set[str] = set()

        for index, term in enumerate(self.terms):
            raw_output = term.module(net_output, target)
            raw_loss, auxiliary = _split_loss_output(raw_output)
            weighted_loss = raw_loss * term.weight
            total = total + weighted_loss

            metric_name = _sanitize_metric_name(term.metric_name)
            if metric_name in used_metric_names:
                metric_name = f"{metric_name}_{index}"
            used_metric_names.add(metric_name)
            metrics[f"loss_terms/{metric_name}_raw"] = float(
                raw_loss.detach().item()
            )
            metrics[f"loss_terms/{metric_name}_weighted"] = float(
                weighted_loss.detach().item()
            )
            for key, value in auxiliary.items():
                auxiliary_key = _sanitize_metric_name(key)
                metrics[f"loss_aux/{metric_name}/{auxiliary_key}"] = float(
                    value.detach().item()
                )

        metrics["loss/total"] = float(total.detach().item())
        self.latest_metrics = metrics
        return total


def create_loss(config: InkConfig) -> CompositeLoss:
    """Build the ordered segmentation objective selected by InkConfig."""

    terms = []
    for term in config.loss.terms:
        module = LabelSmoothedDCAndBCELoss(
            bce_kwargs=dict(term.bce_kwargs),
            soft_dice_kwargs={
                "label_smoothing": term.dice_label_smoothing,
            },
            weight_dice=term.weight_dice,
            weight_ce=term.weight_ce,
            use_ignore_label=True,
            bce_label_smoothing=term.bce_label_smoothing,
            bce_ink_weight=term.bce_ink_weight,
            bce_background_weight=term.bce_background_weight,
            pu_prior=term.pu_prior,
        )
        terms.append(
            WeightedLossTerm(
                name=term.name,
                weight=term.weight,
                module=module,
                metric_name=term.metric_name,
            )
        )
    return CompositeLoss(terms)
