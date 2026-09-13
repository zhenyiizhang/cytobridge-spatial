import numpy as np
import pytest


def test_zebrafish_lineage_uses_direct_predictions(monkeypatch):
    from reproduction.zebrafish import classifier as m
    from types import SimpleNamespace
    calls = []
    def predict(**kwargs):
        calls.append(kwargs['knn_neighbors'])
        return ['A','B']
    monkeypatch.setattr(m.cb.tl,'predict_labels_for_points',predict)
    cache=SimpleNamespace(model=object(),label_encoder=object())
    points=np.zeros((2,52))
    m.assign_cell_types(points,1.,cache,spatial_smoothing=False)
    m.assign_cell_types(points,1.,cache)
    assert calls == [1,10]


@pytest.mark.parametrize('anchor',[False,True])
def test_s21_keeps_wnt_flow_with_figure5_filter_order(anchor):
    from CytoBridge.pl import plot_sankey
    labels=[['reaEGC']*261,
            ['reaEGC']*209+['wntEGC']*37+['ribEGC']*6+['mpEX']*5+['dpEX']*2+['sfrpEGC']+['MCG']]
    figure=plot_sankey(labels,min_flow=10,keep_source_cumfrac=.85,
                      min_flow_before_cumulative=True,lineage_anchor_mode=anchor)
    assert sorted(figure.data[0].link.value)==[37,209]
