import anndata as ad
import numpy as np
import pandas as pd

import CytoBridge.pp.spatial_align as alignment


def prepared_coordinates(xy):
    stages = np.repeat(['D4', 'D7'], len(xy)//2)
    data = ad.AnnData(
        np.ones((len(xy), 2), dtype=np.float32),
        obs=pd.DataFrame({'stage':stages,
                          'time_point_processed':np.repeat([0.,1.],len(xy)//2)},
                         index=[f'cell-{i}' for i in range(len(xy))]),
    )
    data.obsm['spatial'] = xy.copy()
    data.obsm['X_latent'] = np.zeros((len(xy),2),dtype=np.float32)
    cfg = alignment.AlignConfig(center_x=True,center_y=True)
    result,batches = alignment._prepare_adata_for_alignment(data,'stage',cfg)
    return result,batches,cfg


def normalized_tensor_coordinates(xy):
    data,batches,cfg = prepared_coordinates(xy)
    shared = alignment._compute_global_spatial_scaling(data,cfg,'stage')
    result = np.empty_like(xy,dtype=np.float64)
    for batch in batches:
        mask = np.asarray(data.obs.stage==batch)
        result[mask] = alignment._scale_spatial_coords(
            data.obsm['spatial'][mask],cfg,shared_scale_base=shared)
    return result.astype(np.float32)


def test_preserve_coordinate_precision_before_centering():
    xy = np.array([[1e8, 2e8], [1e8+.25,2e8+.5],
                   [1e8+.5,2e8+.75], [1e8+1.,2e8+1.5]])
    data,_,_ = prepared_coordinates(xy)
    assert data.obsm['spatial'].dtype == np.float64
    np.testing.assert_array_equal(data.obsm['spatial'],xy)
    np.testing.assert_array_equal(data.obsm['spatial_original'],xy)


def test_centered_coordinates_do_not_depend_on_stage_translation():
    rng = np.random.default_rng(13)
    xy = rng.normal(size=(64,2))*2000 + np.array([21000.,43000.])
    shifted = xy.copy()
    shifted[:32] += [527.143,-193.269]
    shifted[32:] += [-208.479,638.947]
    np.testing.assert_array_equal(normalized_tensor_coordinates(xy),
                                  normalized_tensor_coordinates(shifted))
