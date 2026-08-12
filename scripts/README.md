# Script status

The supported public entry point is `cytobridge workflow`. Its implementation
lives in `CytoBridge.workflow`, so new datasets and scientific fixes should be
added to the package rather than by copying a full pipeline into another
script.

The source distribution also includes maintained helpers for preprocessing,
training, checkpoint conversion, notebook and wheel smoke tests, training-cost
summaries, the matched spatiotemporal benchmark, and the reviewer analyses
documented in this repository. `complete_downstream.py` is a compatibility
alias for `cytobridge workflow`.

`verify_historical_artifact_compatibility.py` is a read-only maintainer check
for comparing a checkpoint through its original source loader and the current
package loader. Start from
`historical_artifact_compatibility.example.json`; private machine paths are not
stored in this repository.

Other top-level files under `scripts/` in the Git repository are retained as
historical research records. Some contain workstation-specific paths or calls
from earlier package versions. They are not installed, are not included in the
source distribution, and should not be used as a starting point for new work.
