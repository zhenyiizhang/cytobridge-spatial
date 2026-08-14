# Bundled workflow interaction databases

These CSV files are the ligand-receptor inputs used by the formal CytoBridge
workflow presets:

- Zebrafish: `CellChatDB.ligrec.zebrafish.csv`
- MOSTA and AD mouse: `CellChatDB.ligrec.mouse.csv`
- ARISTA: `CellChatDB.ligrec.human.csv`
- Chicken heart: `CellChatDB.ligrec.human.csv`, used as an explicitly labelled
  conserved-symbol proxy because CellChatDB has no Gallus gallus release

The human and mouse files are the formal project copies from `database/`. The
zebrafish file is the formal project copy from
`cb_reproducibility/assets/zebrafish/`. They are bundled unchanged so an
installed workflow can reproduce graph construction from an H5AD without a
repository checkout.

Chicken-heart preprocessing records exact case-insensitive symbol coverage for
the human proxy.  Those results describe model-prior-supported programs; they
must not be described as a chicken-specific curated CellChat database.

The underlying interaction collection is CellChatDB from the
[CellChat project](https://github.com/jinworks/CellChat), distributed under
GPL-3.0. CytoBridge is also GPL-3.0. Users should cite the CellChat papers
listed by that project when using CellChatDB-derived results.
