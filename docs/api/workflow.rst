Workflow API
============

The workflow API reads the same example dataset configurations as ``cytobridge
workflow``. Inspecting the steps does not start a calculation; fitting occurs only when training is
explicitly enabled.

.. automodule:: CytoBridge.workflow
   :members: WorkflowOptions, available_workflow_configs, load_workflow_config, build_workflow_plan, render_workflow_plan, plan_missing_inputs, run_workflow
   :show-inheritance:
