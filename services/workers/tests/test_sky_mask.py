import numpy as np
import pytest

pytest.importorskip("cv2")

from memoryscapes_workers.visual.sky_mask import sky_mask  # noqa: E402


def frame(sky=True):
    rng = np.random.default_rng(0)
    h, w = 240, 320
    img = (rng.integers(60, 140, (h, w, 3))).astype(np.uint8)  # textured ground and buildings
    if sky:
        img[:100] = (110, 160, 235)  # blue sky
        img[20:45, 60:140] = (245, 245, 248)  # a cloud
        img[100:110] = (40, 40, 40)  # skyline
    img[150:200, 30:90] = (235, 235, 235)  # white wall, not touching the top
    img[160:190, 200:260] = (60, 110, 220)  # a blue car, not touching the top
    return img


def test_masks_sky_and_clouds_only():
    m = sky_mask(frame())
    assert (m[5:95, 5:315] == 0).mean() > 0.95  # sky, cloud included
    assert (m[25:40, 70:130] == 0).all()
    assert (m[150:200, 30:90] == 255).all() and (m[160:190, 200:260] == 255).all()
    assert (m[120:] == 255).mean() > 0.97  # ground untouched (dilation may nibble the edge)


def test_frame_without_sky_is_left_alone():
    assert (sky_mask(frame(sky=False)) == 255).all()
