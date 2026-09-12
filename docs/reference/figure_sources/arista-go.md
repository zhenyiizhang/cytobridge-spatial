---
orphan: true
---

# ARISTA GO analysis

Supplementary Figure S22 uses `clusterProfiler::enrichGO` with mouse
`org.Mm.eg.db` annotations. Axolotl ortholog symbols are matched to mouse
symbols without regard to letter case. The background contains all detected
genes whose recorded identifiers match a mouse symbol. For the background,
the recorded gene identifiers are retained as in the paper analysis. Compound
identifiers are not converted to additional symbols. The analysis uses biological-process terms,
gene-set sizes of 5–500, and Benjamini–Hochberg correction.

In the [ARISTA notebook](../../tutorials/paper_figures/arista_figures.ipynb),
`calculate_gene_programs` reconstructs expression, groups the temporal gene
profiles, and calls `reproduction/arista/enrich_go.R`. Install R with
`clusterProfiler`, `AnnotationDbi`, and `org.Mm.eg.db` before this step.
The paper used clusterProfiler 4.10.0 and org.Mm.eg.db 3.18.0. Annotation updates
can change the enrichment results.

The function writes full results to `genes/clusterprofiler/`, including
adjusted P values and the number of significant terms. `draw_supplementary`
uses these files for S22c/d. The displayed 20 terms are ranked by unadjusted
P value, which also sets their color. The caption reports significance after
correction. The paper's numerical results and R session record are included
in `reproduction/arista/data/s22_clusterprofiler/`.

The GO BP 2023 GMT retained with earlier analyses is not the annotation
source for these paper panels.

Gene Ontology data are copyright the Gene Ontology Consortium and contributors,
provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) without
warranty. Follow the [GO citation and license policy](https://geneontology.org/docs/go-citation-policy/).
Cite the Gene Ontology Consortium and the original GO paper (Ashburner et al.,
2000, [10.1038/75556](https://doi.org/10.1038/75556)).
