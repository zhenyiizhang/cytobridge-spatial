"""Draw the Figure 5e group means in the paper's original layout."""
from pathlib import Path
import fitz
import numpy as np

TIME_COLORS = ('#0d0887', '#4c02a1', '#7e03a8', '#aa2395', '#cc4778',
               '#e66c5c', '#f89540', '#fdc527', '#f0f921')


def plot_growth_interaction(grouped, stem):
    """One vector circle per time/cell-type group, sized by cell count.

    The PDF asset contains axes, labels, a colour bar and the two original
    annotation arrows. Every data circle is drawn here from ``grouped``.
    """
    table = grouped.sort_values(['time', 'celltype'], kind='stable').copy()
    values = table[['interaction_mean', 'growth_mean']].to_numpy(dtype=float)
    ranges = np.ptp(values, axis=0)
    if not np.isfinite(values).all() or np.any(ranges <= 0):
        raise ValueError('Growth and interaction must have finite, nonconstant values.')
    fractions = (values - values.min(axis=0) + .05 * ranges) / (1.1 * ranges)
    centers = fractions * np.array([220.0726, 149.75158]) + np.array([298.3127, 24.99231])
    # Convert the original Illustrator bottom-left coordinates to this crop.
    centers[:, 0] -= 216.
    centers[:, 1] = 841.89 - centers[:, 1] - 630.
    radius = .29366254548 * np.sqrt(np.clip(table['n'].to_numpy(), 20, 400))
    path = Path(__file__).parent / 'data/figure5e_labels.pdf'
    document = fitz.open(path)
    page = document[0]
    shapes = page.new_shape()
    for center, r, time in zip(centers, radius, table.time):
        index = int(round(float(time) * 2))
        if index not in range(9) or not np.isclose(time, index / 2):
            raise ValueError('Figure 5e uses model times 0, 0.5, ..., 4.')
        hex_color = TIME_COLORS[index]
        color = tuple(int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5))
        shapes.draw_circle(tuple(center), float(r))
        shapes.finish(color=(1, 1, 1), fill=color, width=.3,
                      fill_opacity=.800003, stroke_opacity=.800003)
    shapes.commit(overlay=False)
    stem = Path(stem)
    paths = [stem.with_suffix('.pdf'), stem.with_suffix('.png')]
    document.save(paths[0], garbage=4, deflate=True)
    page.get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False).save(paths[1])
    document.close()
    return paths
