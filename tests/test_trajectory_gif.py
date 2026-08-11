from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from CytoBridge.pl.trajectory import plot_trajectory_gif


def test_trajectory_gif_uses_opaque_full_frames_without_title_ghosting(tmp_path):
    points = np.asarray(
        [
            [[0.05, 0.10], [0.30, 0.80], [0.90, 0.20]],
            [[0.10, 0.20], [0.45, 0.70], [0.85, 0.30]],
            [[0.20, 0.30], [0.55, 0.60], [0.75, 0.40]],
        ],
        dtype=float,
    )
    # The first title is deliberately much wider than the later titles. A
    # retained overlay therefore leaves visible digits around the later title.
    times = [123456789.0, 1.0, 2.0]
    target_index = 1

    reference_animation = plot_trajectory_gif(points, times, fps=3)
    reference_animation._func(target_index)
    reference_figure = reference_animation._fig
    reference_figure.canvas.draw()
    expected = np.asarray(reference_figure.canvas.buffer_rgba())[..., :3].copy()

    out_path = tmp_path / "trajectory.gif"
    assert plot_trajectory_gif(points, times, out_path=str(out_path), fps=3) == str(
        out_path
    )

    decoded = []
    with Image.open(out_path) as image:
        assert image.n_frames == len(times)
        full_extent = (0, 0, image.width, image.height)
        for frame_index in range(image.n_frames):
            image.seek(frame_index)
            assert image.disposal_method == 2
            assert image.dispose_extent == full_extent
            rgba = np.asarray(image.convert("RGBA"))
            assert np.all(rgba[..., 3] == 255)
            decoded.append(rgba[..., :3])

    actual = decoded[target_index]
    assert actual.shape == expected.shape
    difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert float(np.mean(difference)) < 1.5

    renderer = reference_figure.canvas.get_renderer()
    title_bbox = reference_figure.axes[0].title.get_window_extent(renderer=renderer)
    height, width = actual.shape[:2]
    y_start = max(0, height - int(np.ceil(title_bbox.y1)) - 4)
    y_stop = min(height, height - int(np.floor(title_bbox.y0)) + 4)
    x_start = max(0, width // 2 - 180)
    x_stop = min(width, width // 2 + 180)
    title_difference = difference[y_start:y_stop, x_start:x_stop]
    assert float(np.mean(title_difference)) < 2.0
    assert float(np.mean(np.max(title_difference, axis=2) > 40)) < 0.01

    reference_animation._draw_was_started = True
    plt.close(reference_figure)
