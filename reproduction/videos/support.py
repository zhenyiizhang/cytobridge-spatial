"""Count the cell edges supporting the displayed MOSTA interaction pairs."""
import numpy as np
from . import _mosta_style as mosta


def attach_mosta_support(root, frames, communications):
    records = []
    for index, frame in enumerate(frames):
        labels, time = frame['labels'], frame['time']
        # The generator evaluates each frame separately with seed 42 and
        # at most 12,000 cells. Reuse that ordering for the saved edge indices.
        take = (np.sort(np.random.default_rng(42).choice(len(labels), 12000, replace=False))
                if len(labels) > 12000 else np.arange(len(labels)))
        edges = np.load(root / 'attention' / f'edge_index_interp_t{time}.npy')
        weights = np.load(root / 'attention' / f'attn_mean_interp_t{time}.npy').reshape(-1)
        if edges.ndim != 2 or edges.shape[0] != 2 or len(weights) != edges.shape[1]:
            raise ValueError(f'Attention arrays do not match at time {time}.')
        if edges.size and (edges.min() < 0 or edges.max() >= len(take)):
            raise ValueError(f'Attention indices exceed the sampled population at time {time}.')
        comm = communications[str(index)]
        rows = []
        for score, source, target in mosta._get_focus_edges(
                comm, focus_type='Brain', mode='both', top_k=3):
            mask = ((labels[take][edges[0]] == source)
                    & (labels[take][edges[1]] == target) & (weights > 0))
            pair_edges = np.unique(edges[:, mask].T, axis=0)
            rows.append(dict(source=source, target=target, score=float(score),
                             cell_edges=len(pair_edges),
                             source_cells=len(np.unique(pair_edges[:, 0])),
                             target_cells=len(np.unique(pair_edges[:, 1]))))
        comm['display_edge_support'] = rows
        comm['display_min_edges'] = 2
        records.append(dict(frame=index, time=time, top3=rows))
    return records
