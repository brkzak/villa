"""In-plane scale augmentation (``augmentation_inplane_scale``): config, transform and preset wiring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vesuvius.ink_detection.config import AugmentationConfig
from vesuvius.ink_detection.data.augmentations import (
    InPlaneScaleSpatialTransform,
    build_augmentations,
    split_augmentations_by_geometry,
)


def _sample(shape=(5, 32, 32)):
    image = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(1, *shape)
    labels = torch.zeros(1, *shape)
    labels[:, :, 12:20, 12:20] = 1.0
    labels[:, :, 14:16, 14:16] = 128.0  # a PU "unlabeled" code must survive exactly
    supervision = torch.ones(1, *shape)
    return {"image": image, "inklabels": labels, "supervision_mask": supervision}


def test_absent_setting_is_off_and_adds_nothing():
    """No ``augmentation_inplane_scale`` must leave the preset as it was, so every earlier config trains bit for
    bit as before (no new transform, no extra random draw)."""
    config = AugmentationConfig.from_mapping({})
    assert config.inplane_scale_probability == 0.0
    plain = build_augmentations("default", (5, 32, 32), rotation_axes=(0,))
    off = build_augmentations("default", (5, 32, 32), rotation_axes=(0,), inplane_scale=(0.0, 0.0))
    assert [type(t) for t in off.transforms] == [type(t) for t in plain.transforms]
    assert build_augmentations("none", (5, 32, 32), rotation_axes=None, inplane_scale=(0.0, 0.0)) is None


@pytest.mark.parametrize(
    "value",
    [{"probability": 0.0, "range": 0.1}, {"probability": 1.5, "range": 0.1}, {"probability": 0.5, "range": 0.0},
     {"probability": 0.5, "range": 0.6}, {"probability": 0.5}, {"probability": 0.5, "range": 0.1, "x": 1}, 0.2],
)
def test_bad_setting_is_refused(value):
    """A probability outside (0, 1] or a range outside (0, 0.5] must fail loudly, never train silently with
    another augmentation than asked."""
    with pytest.raises(ValueError, match="augmentation_inplane_scale"):
        AugmentationConfig.from_mapping({"augmentation_inplane_scale": value})


def test_setting_is_put_first_and_counts_as_geometry():
    """With a probability the scale transform must run before every other one (the rest sees the zoomed sample)
    and count as geometric, so pseudo-label training takes its image after it, as for flips and turns."""
    config = AugmentationConfig.from_mapping({"augmentation_inplane_scale": {"probability": 0.5, "range": 0.15}})
    assert (config.inplane_scale_probability, config.inplane_scale_range) == (0.5, 0.15)
    built = build_augmentations("default", (5, 32, 32), rotation_axes=(0,), inplane_scale=(0.5, 0.15))
    first = built.transforms[0]
    assert first.apply_probability == 0.5 and isinstance(first.transform, InPlaneScaleSpatialTransform)
    geometric, _ = split_augmentations_by_geometry(built)
    assert geometric.transforms[0] is first
    only = build_augmentations("none", (5, 32, 32), rotation_axes=None, inplane_scale=(1.0, 0.15))
    assert len(only.transforms) == 1


def test_depth_layers_stay_exact_and_labels_keep_their_values():
    """The depth axis must never be resampled (each layer is a fixed distance from the surface), and label
    values must stay exactly the codes they were (nearest neighbour), whatever the zoom."""
    np.random.seed(0)
    transform = InPlaneScaleSpatialTransform((5, 32, 32), 0.3)
    for _ in range(5):
        data = transform(**_sample())
        assert data["image"].shape == (1, 5, 32, 32)
        assert set(torch.unique(data["inklabels"]).tolist()) <= {0.0, 1.0, 128.0}
        assert set(torch.unique(data["supervision_mask"]).tolist()) <= {0.0, 1.0}
        # Each layer of the input is a constant offset of 32*32 from the one before; a z resample would mix them.
        # The centre 16 x 16 stays inside the sample at any zoom up to 1.3 (bilinear blends edges with padding).
        layer_step = data["image"][0, 1:, 8:24, 8:24] - data["image"][0, :-1, 8:24, 8:24]
        assert torch.allclose(layer_step, torch.full_like(layer_step, 32 * 32), atol=1e-2)


def test_zoom_changes_stroke_size_as_documented():
    """A factor above 1 samples a wider area, so the ink square must shrink; below 1 it must grow; outside the
    sample a zoom-out leaves the pixels unsupervised."""
    transform = InPlaneScaleSpatialTransform((5, 32, 32), 0.4)
    sample = _sample()
    ink_before = int((sample["inklabels"] > 0).sum())
    params = transform.get_parameters(**sample)
    params_out = {**params, "affine": np.diag([1.0, 1.3, 1.3])}
    zoomed_out = transform.apply({k: v.clone() for k, v in sample.items()}, **params_out)
    assert int((zoomed_out["inklabels"] > 0).sum()) < ink_before
    assert float(zoomed_out["supervision_mask"][0, :, 0, 0].max()) == 0.0
    params_in = {**params, "affine": np.diag([1.0, 0.75, 0.75])}
    zoomed_in = transform.apply({k: v.clone() for k, v in sample.items()}, **params_in)
    assert int((zoomed_in["inklabels"] > 0).sum()) > ink_before
    assert float(zoomed_in["supervision_mask"].min()) == 1.0


def test_two_dimensional_patch_is_supported():
    """2.5D configs have 2D patches; the transform must zoom them the same way."""
    transform = InPlaneScaleSpatialTransform((32, 32), 0.2)
    sample = {"image": torch.rand(1, 32, 32), "inklabels": torch.zeros(1, 32, 32)}
    assert transform(**sample)["image"].shape == (1, 32, 32)
