Workflow API
============

The workflow API reads the same example dataset configurations as ``cytobridge
workflow``. For a new run, set ``train=True`` and provide a raw H5AD. When the
configuration keeps downstream as its default and preprocessing is enabled,
CytoBridge preprocesses, trains, and then runs the configured analyses. This
also applies to JSON files exported from an included configuration. Selecting
preprocessing alone writes an aligned H5AD for
inspection but does not fit an edge predictor or model. Inspecting a plan does
not start a calculation, and fitting occurs only when training is explicitly
enabled.

Start a model from raw data
---------------------------

The Python API below performs the same operation as ``cytobridge workflow
--train``: preprocessing, training, and the analyses selected in the
configuration.
``build_workflow_plan`` organizes the requested steps, while ``run_workflow``
starts them.

.. code-block:: python

   from pathlib import Path

   from CytoBridge.workflow import (
       WorkflowOptions,
       build_workflow_plan,
       load_workflow_config,
       plan_missing_inputs,
       render_workflow_plan,
       run_workflow,
   )

   config, source = load_workflow_config("zebrafish")
   options = WorkflowOptions(
       input_h5ad=Path("inputs/zebrafish_raw.h5ad"),
       output_dir=Path("outputs/zebrafish"),
       train=True,
       device="cuda",
   )

   plan = build_workflow_plan(config, source=source, options=options)
   print(render_workflow_plan(plan))
   missing_options = plan_missing_inputs(plan)
   if missing_options:
       raise ValueError(f"Add these workflow options: {missing_options}")

   result = run_workflow(config, options=options)

The plan checks whether the selected steps have the required options. It does
not open the H5AD or verify its columns, layers, coordinates, or values.
Preprocessing performs those data checks when ``run_workflow`` starts.

Repeat downstream analysis
--------------------------

The first call already writes ``outputs/zebrafish/downstream``. To repeat that
analysis with different options, use the aligned H5AD and model directory from
the same run and choose a new output directory.

.. code-block:: python

   downstream = WorkflowOptions(
       aligned_h5ad=Path(
           "outputs/zebrafish/preprocess/zebrafish_aligned.h5ad"
       ),
       model_dir=Path("outputs/zebrafish/training"),
       output_dir=Path("outputs/zebrafish_downstream_rerun"),
       steps=("downstream",),
       device="cuda",
   )
   downstream_result = run_workflow(config, options=downstream)

``run_workflow`` returns a dictionary of the files and directories written by
the selected steps. Dataset tutorials show the corresponding command-line
form and the exact fields expected by each included configuration.

.. automodule:: CytoBridge.workflow
   :members: WorkflowOptions, available_workflow_configs, load_workflow_config, build_workflow_plan, render_workflow_plan, plan_missing_inputs, run_workflow
   :show-inheritance:
