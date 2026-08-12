Preprocessing API
=================

Raw-count preprocessing
-----------------------

.. automodule:: CytoBridge.pp.preprocess
   :members: preprocess

Spatial alignment
-----------------

.. automodule:: CytoBridge.pp.spatial_align
   :members: AlignConfig, align_spatial, preprocess_and_align, preprocess_align_to_files
   :show-inheritance:

Interaction graph and edge prediction
-------------------------------------

.. automodule:: CytoBridge.pp.interaction_graph
   :members: estimate_neighborhood_threshold_from_aligned_spatial, generate_interaction_graph, sanitize_interaction_graph_uns

.. automodule:: CytoBridge.pp.edge_prediction
   :members: train_edge_predictor

Legacy input adapter
--------------------

.. automodule:: CytoBridge.pp.legacy_model_input
   :members: legacy_model_input_csv_to_adata, write_legacy_model_input_h5ad
