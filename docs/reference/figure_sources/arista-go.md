---
orphan: true
---

# ARISTA gene-set input

S22 uses the Enrichr **GO Biological Process 2023** gene-set library. The exact
1.4-MB GMT used for the paper is included in the CytoBridge code folder at
`reproduction/arista/data/GO_Biological_Process_2023.gmt`.
The [ARISTA S19–S24 notebook](../../tutorials/paper_figures/arista_figures.ipynb)
recalculates enrichment from its newly reconstructed gene programs and expressed
gene background. It does not read precomputed enrichment statistics.

The original library is available in the
[Enrichr library collection](https://maayanlab.cloud/Enrichr/#libraries).
The retained input does not identify a more specific daily GO release. The file
has not been modified; the included `GO_SOURCE.md` records its identity and origin.

Gene Ontology data are copyright the Gene Ontology Consortium and contributors,
provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) without
warranty. Follow the [GO citation and license policy](https://geneontology.org/docs/go-citation-policy/).
Cite the Gene Ontology Consortium and the original GO paper (Ashburner et al.,
2000, [10.1038/75556](https://doi.org/10.1038/75556)), and the Enrichr resource
(Chen et al., 2013, [10.1186/1471-2105-14-128](https://doi.org/10.1186/1471-2105-14-128)).
