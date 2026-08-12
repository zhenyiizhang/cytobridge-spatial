Workflow API
============

The workflow API reads the same wheel-bundled presets as ``cytobridge
workflow``. Building a plan is read-only; fitting occurs only when training is
explicitly enabled.

.. automodule:: CytoBridge.workflow
   :members: WorkflowOptions, available_workflow_configs, load_workflow_config, build_workflow_plan, render_workflow_plan, plan_missing_inputs, run_workflow
   :show-inheritance:
