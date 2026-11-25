"""Segmentation array examples used by the blog post.

Basic run:
    python -m pip install numpy opencv-python-headless
    python examples.py

Optional PyTorch input/loss check:
    python examples.py --torch-check

The scenes are synthetic. Exact RGB matching is used to explain array operations,
not as a general object detector for photographs.
"""

import argparse

import cv2
import numpy as np


H, W = 120, 200
CLASS_NAMES = ["sky", "ground", "ball"]
C = len(CLASS_NAMES)
BALL_RGB = (230, 60, 50)
PALETTE = np.array(
    [[222, 231, 240], [73, 128, 93], [185, 43, 62]], dtype=np.uint8
)


def make_scene():
    """Return an RGB image and its per-pixel class IDs."""
    toy = np.zeros((H, W, 3), dtype=np.uint8)
    toy[:] = (150, 200, 255)
    toy[90:] = (90, 160, 70)

    # This array is interpreted as RGB; circle writes the supplied components.
    # A file loaded with cv2.imread normally needs BGR -> RGB conversion first.
    cv2.circle(toy, (55, 60), 25, BALL_RGB, -1)
    cv2.circle(toy, (145, 60), 20, BALL_RGB, -1)

    semantic = np.zeros((H, W), dtype=np.uint8)
    semantic[90:] = 1
    component_matches = toy == BALL_RGB       # (H, W, 3), bool
    ball = np.all(component_matches, axis=-1) # (H, W), bool
    semantic[ball] = 2
    return toy, semantic, ball


def check_class_ids(mask, num_classes):
    """Validate the class-ID array before using its values as row indices."""
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Expected a 2-D integer class-ID mask.")
    if num_classes <= 0 or mask.size == 0:
        raise ValueError("The class count and mask size must be positive.")
    if mask.min() < 0 or mask.max() >= num_classes:
        raise ValueError("Class IDs must be in [0, num_classes).")


def instance_masks(binary_mask, connectivity=8):
    """Return connected-region labels and a (region_count, H, W) bool array."""
    if binary_mask.ndim != 2 or binary_mask.dtype != np.bool_:
        raise ValueError("Expected a 2-D boolean mask.")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")
    num_labels, labeled = cv2.connectedComponents(
        binary_mask.astype(np.uint8), connectivity=connectivity
    )
    # Label 0 is background; num_labels includes this background label.
    masks = [labeled == i for i in range(1, num_labels)]
    # np.stack([]) raises an error, so keep an explicit zero-object shape.
    instances = (
        np.stack(masks)
        if masks
        else np.empty((0, *binary_mask.shape), dtype=bool)
    )
    return num_labels, labeled, instances


def overlay(image, mask, alpha=0.45):
    """Tint only ball pixels; preserve the original sky and ground RGB values."""
    check_class_ids(mask, len(PALETTE))
    if image.shape != (*mask.shape, 3) or image.dtype != np.uint8:
        raise ValueError("Expected a matching uint8 RGB image.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1].")
    overlay_palette = PALETTE.copy()
    overlay_palette[2] = (255, 220, 0)
    color = overlay_palette[mask].astype(np.float32) # (H, W, 3)
    mixed = image.astype(np.float32) * (1 - alpha) + color * alpha
    out = image.copy()
    ball = mask == 2
    # Only selected pixels are changed; both other classes retain their RGB.
    out[ball] = np.clip(mixed[ball], 0, 255).astype(np.uint8)
    return out


def torch_check(image, mask):
    """Check shapes and gradients using an untrained one-layer model."""
    import torch
    from torch import nn

    image_chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    x = torch.from_numpy(image_chw).float().unsqueeze(0) / 255.0
    # IDs stay 0, 1, 2: do not divide the target mask by 255.
    target = torch.from_numpy(mask.copy()).long().unsqueeze(0)

    torch.manual_seed(0)
    model = nn.Conv2d(3, C, kernel_size=3, padding=1)
    logits = model(x)
    # Raw per-class scores enter the loss; do not argmax or softmax them first.
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, target)
    loss.backward()
    pred = logits.detach().argmax(dim=1)

    assert x.shape == (1, 3, H, W)
    assert logits.shape == (1, C, H, W)
    assert target.shape == pred.shape == (1, H, W)
    assert target.dtype == torch.long
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()
    assert torch.isfinite(loss)
    print("torch image:", tuple(x.shape), x.dtype)
    print("torch target:", tuple(target.shape), target.dtype)
    print("torch logits:", tuple(logits.shape))
    print("torch prediction:", tuple(pred.shape))
    print("CrossEntropyLoss and backward: OK (randomly initialized model; no accuracy evaluation)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-check", action="store_true")
    args = parser.parse_args()

    toy, semantic, ball = make_scene()
    check_class_ids(semantic, C)

    # A lookup table turns one class ID into C membership values.
    onehot = np.eye(C, dtype=np.uint8)[semantic].transpose(2, 0, 1)
    back = onehot.argmax(axis=0)
    counts = np.bincount(semantic.ravel(), minlength=C)
    num_labels, labeled, instances = instance_masks(ball)

    assert toy.shape == (H, W, 3) and toy.dtype == np.uint8
    assert ball.shape == (H, W) and ball.dtype == np.bool_
    assert np.array_equal(np.unique(semantic), [0, 1, 2])
    assert np.all(onehot.sum(axis=0) == 1)
    assert np.array_equal(back, semantic)
    assert np.array_equal(counts, onehot.sum(axis=(1, 2)))
    assert counts.sum() == H * W
    assert num_labels == 3 and instances.shape == (2, H, W)
    assert np.array_equal(instances.any(axis=0), ball)

    # Rebuild class IDs using each region's mask, preserving the ground class.
    rebuilt = np.zeros((H, W), dtype=np.uint8)
    rebuilt[90:] = 1
    for region in instances:
        rebuilt[region] = 2
    assert np.array_equal(rebuilt, semantic)

    # Geometry changes must use the same transformation for both arrays.
    img_flip = toy[:, ::-1, :].copy()
    mask_flip = semantic[:, ::-1].copy()
    assert np.array_equal(np.all(img_flip == BALL_RGB, axis=-1), mask_flip == 2)
    assert np.array_equal(mask_flip[:, ::-1], semantic)

    new_H, new_W = 60, 100
    img_small = cv2.resize(toy, (new_W, new_H), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(
        semantic, (new_W, new_H), interpolation=cv2.INTER_NEAREST
    )
    assert img_small.shape[:2] == mask_small.shape == (new_H, new_W)
    assert set(np.unique(mask_small)) <= set(np.unique(semantic))

    tinted = overlay(toy, semantic)
    assert np.array_equal(tinted[semantic == 0], toy[semantic == 0])
    assert np.array_equal(tinted[semantic == 1], toy[semantic == 1])
    assert toy[ball].shape == (int(ball.sum()), 3)

    # No foreground and touching foreground are both valid input cases.
    _, _, empty = instance_masks(np.zeros((H, W), dtype=bool))
    assert empty.shape == (0, H, W)
    touching = np.zeros((64, 96), dtype=np.uint8)
    cv2.circle(touching, (32, 32), 16, 1, -1)
    cv2.circle(touching, (64, 32), 16, 1, -1)
    _, _, merged = instance_masks(touching.astype(bool))
    assert merged.shape[0] == 1

    diagonal = np.array([[True, False], [False, True]])
    assert instance_masks(diagonal, connectivity=4)[2].shape[0] == 2
    assert instance_masks(diagonal, connectivity=8)[2].shape[0] == 1

    # Invalid IDs must be rejected before identity/palette row indexing.
    for bad in (np.array([[-1, 0]]), np.array([[0, C]])):
        try:
            check_class_ids(bad, C)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid class IDs were accepted.")

    print("image:", toy.shape, toy.dtype)
    print("semantic:", semantic.shape, semantic.dtype, np.unique(semantic))
    print("ball:", ball.shape, ball.dtype)
    print("onehot:", onehot.shape, onehot.dtype)
    print("onehot -> class IDs:", np.array_equal(back, semantic))
    print("class counts:", dict(zip(CLASS_NAMES, counts.tolist())))
    print("ball pixels:", int(ball.sum()))
    print("ball fraction: {:.4f}%".format(100 * ball.sum() / semantic.size))
    print("num_labels (including background):", num_labels)
    print("instances:", instances.shape, instances.dtype)
    print("region areas (sorted):", sorted(instances.sum(axis=(1, 2)).tolist()))
    print("empty instances:", empty.shape)
    print("touching regions:", merged.shape[0])
    print("paired flip / nearest resize / overlay / ID checks: OK")
    if args.torch_check:
        torch_check(toy, semantic)


if __name__ == "__main__":
    main()
