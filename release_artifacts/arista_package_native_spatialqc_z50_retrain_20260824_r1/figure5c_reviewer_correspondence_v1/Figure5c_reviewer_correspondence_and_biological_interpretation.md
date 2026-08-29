# Figure 5c reviewer correspondence and biological interpretation

**Dataset:** ARISTA axolotl telencephalon regeneration  
**Analysis stage:** 5 days post injury (5 DPI; model time `t = 1`)  
**Figure:** Local interaction domains underlying the heterogeneous Figure 5c spatial-velocity pattern  
**Document status:** Correspondence-ready draft, 25 August 2026  

![Figure 5c reviewer-response analysis](/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_reviewer_figure_clean_v9_mechanism_final/FigureS_ARISTA_Figure5c_local_interaction_niches_clean.png)

## 作者内部中文备注（投稿前删除）

### 一句话结论

Figure 5c 右图中看似零散、没有沿着单一伤口轮廓分布的高 cosine 信号，并不是一组无组织的异常点。它可以被解析为两个空间上连续、内部 interaction network 显著富集、并且具有不同候选分子程序的局部修复微环境：一个以 sfrpEGC 和 VLMC 为核心，偏向 ECM 重塑和 trophic support；另一个以 reaEGC 和 wntEGC 为核心，偏向 reactive adhesion、neural guidance 和局部细胞状态协调。

### 这张图整体在讲什么

这张图不是想证明“某一个 ligand–receptor pair 直接造成了脑再生”，而是要回答 reviewer 更基础、也更重要的问题：Figure 5c 右边的异质空间信号到底是不是有组织的细胞网络，以及这些局部网络可能在做什么。

四个 panel 构成一条连续的证据链：

1. **Panel a 先把细胞客观地提取出来。** 对每个细胞分别将 full spatial velocity 和 interaction spatial velocity 投影到 spatial basis，再计算逐细胞 cosine。取 ROI 内 cosine 最高的 25% 细胞，在训练时使用的物理邻接半径上构建 connected components。整个分区过程不使用 cell-type label。最终得到 203-cell 和 77-cell 两个连续区域。
2. **Panel b 检查这些区域是否真的具有组织性。** 在相同 ROI 内随机抽取细胞，并严格保持每一种 cell type 的数量不变。这样排除了“这里只是刚好聚集了更多某类细胞”这一解释。两个真实区域的 selected attention 分别是 null 的 1.69 倍和 3.06 倍，9,999 次随机抽样的 empirical P 都是 0.0001。因此，这些区域的 interaction enrichment 超出了 cell-type composition 本身能够解释的范围。
3. **Panel c 判断两个区域的候选生物学功能是否不同。** N1 显著富集 AGRN、LAMININ、TENASCIN、FGF 和 THBS，形成相对一致的 matrix/trophic program。N2 显著富集 GRN、L1CAM、NRXN、SEMA3 和 FN1，形成 reactive adhesion/guidance program。这说明两个区域并非同一种热点在不同位置的重复，而是具有不同的分子组织方式。
4. **Panel d 把 pathway 落到具体的 sender→receiver 机制。** 每个 bar 都给出 ligand、receptor 或 receptor complex，以及主要贡献该得分的 sender 和 receiver cell state。所有展示的 pair 都经过 531 个候选 LR pairs 的 BH correction，而不是只画未经校正的高分基因对。

### N1 的生物学解读：matrix scaffold 与 stromal trophic support

N1 有 203 个细胞和 419 条内部 selected edges。它的命名来自内部 interaction structure，而不是细胞组成的纯度：51.1% 的 selected attention 来自 sfrpEGC→sfrpEGC，24.7% 来自 VLMC↔sfrpEGC。因此更准确的名称是“sfrpEGC–VLMC-associated domain”，不能写成纯 sfrpEGC/VLMC cluster。

N1 的几条 LR 轴共同指向一个相对完整的生物学模型：

- `AGRN–DAG1`、`LAMA2–DAG1` 提示 basement membrane/ECM 与细胞表面锚定。
- `TNC–SDC4` 和 `THBS1–SDC4` 提示动态 ECM 重塑、黏附和潜在的局部运动调控。
- 最有方向性的机制是 `FGF7–FGFR1, VLMC→sfrpEGC`。该方向贡献了这一 LR pair 总得分的 86.2%，相对 composition-matched null 富集 6.45 倍。这提示 VLMC-like stromal cells 可能向 sfrpEGC 提供局部 trophic input。

因此，N1 可以被概括为一个以 sfrpEGC 为中心、由 VLMC 输入支持的 matrix/trophic microenvironment。它可能参与建立或维持适合 injury-responsive EGC 存活、状态转换或局部组织重塑的 extracellular scaffold。不过，这仍然是模型支持的机制假说，不能写成已经验证的 VLMC→sfrpEGC 因果信号。

N1 中的 `LAMA2–DAG1` 主要方向是 reaEGC→reaEGC，并不与区域名称矛盾。N1 本身是异质 component，其中包含 7 个 reaEGC。这个结果提示在较大的 sfrpEGC/VLMC 网络内部还存在一个较小的 reactive-EGC matrix subnetwork。

### N2 的生物学解读：reactive EGC coordination、adhesion 与 neural guidance

N2 有 77 个细胞和 122 条内部 selected edges。虽然区域较小，但它相对于 composition-matched null 的 enrichment 更强。内部 selected attention 的 61.2% 来自 wntEGC→wntEGC，27.6% 来自 reaEGC→reaEGC，11.3% 来自 reaEGC↔wntEGC。因此 N2 是一个由两类 injury-responsive ependymoglial states 主导的局部协调网络。

N2 的候选机制包括：

- `GRN–SORT1, wntEGC→wntEGC`：提示 trophic/stress-response 或蛋白运输相关信号。
- `L1CAM–L1CAM, reaEGC→reaEGC`：提示 homophilic adhesion 和局部细胞/突起的组织。
- `NRXN2–NLGN2, wntEGC→wntEGC`：提示 cell-contact 和 synaptic-organization machinery，但不能据此声称已经形成成熟突触。
- `SEMA3F–NRP2/PLXNA3, wntEGC→wntEGC`：这一方向贡献了 pair 总得分的 94.6%，提示非常集中的 neural-guidance program。它是 N2 最清晰的方向性机制之一。
- `FN1–ITGA5/ITGB1, reaEGC→reaEGC`：提示 matrix–integrin anchoring。图中的 32.05 倍是相对于非常低的 null mean 得到的 fold enrichment。它的 absolute observed score 很低，因此应解释为“高度特异但低幅度”的候选轴，不能称为整个网络中最强的通信。

N2 因而更适合被解释为 reactive adhesion/guidance domain：reaEGC 和 wntEGC 可能通过 homotypic adhesion、matrix engagement 和 guidance cues 协调局部状态，并与损伤后的神经组织重新排列发生联系。

### 两个区域合起来说明什么

最稳妥的整体模型是：5 DPI 的 injury response 不是一个空间均一、只沿着伤口边界变化的程序，而是由至少两个并存的局部微环境组成。N1 更偏向搭建和维持 matrix/trophic scaffold，N2 更偏向协调 reactive EGC 状态、细胞黏附和 neural guidance。这个结果为原始 ARISTA 工作中“local resident EGC 被损伤激活并进入 progenitor/neural transition”的框架增加了空间和 interaction 层面的解释。

目前的数据不能证明 N1→N2 是时间上的前后阶段，也不能证明两个区域分别对应伤口内外。更不能把 high cosine 直接解释成物理 migration。正确的表述应该是：**Figure 5c 的异质场中存在两个 localized, organized, model-supported interaction domains，它们分别具有 matrix/trophic 和 reactive adhesion/guidance 的候选分子程序。**

### 结论强度和投稿时的用词边界

可以较强地说：

- 两个空间 connected components 在固定规则下稳定存在。
- 它们的 selected attention 显著高于相同 cell-type composition 的随机区域。
- 两个区域具有不同的 dominant sender–receiver structure。
- 展示的 pathway 和 LR pairs 均通过多重检验校正。
- FGF7–FGFR1 和 SEMA3F–NRP2/PLXNA3 提供了清晰、可实验验证的方向性机制假说。

不能过度声称：

- 这些 LR axes 已经被证明具有因果作用。
- attention 是真实的生物物理 signaling flux。
- N1 和 N2 是确定的连续时间阶段。
- 这些区域严格沿着伤口轮廓分布。
- 32 倍 enrichment 等价于绝对信号最强。

另外，composition-matched null 保持了 cell-type counts，但没有保持空间几何。因此它能够证明 enrichment 不是简单由细胞组成造成的，却不能完全排除所有形式的 spatial autocorrelation。LR 分析还使用了 axolotl-to-human symbol mapping 和 human CellChatDB，因此 pair-level 结果应作为 biologically informed and model-supported hypotheses，而不是物种内已经验证的分子互作。

---

## Central conclusion

The heterogeneous spatial-velocity pattern in Figure 5c contains two spatially connected, model-supported interaction domains rather than an unstructured collection of high-scoring cells. One domain is dominated by sfrpEGC homotypic and VLMC–sfrpEGC interactions and is associated with extracellular-matrix and trophic signaling. The second is dominated by wntEGC and reaEGC interactions and is associated with reactive adhesion and neural-guidance signaling. These domains provide candidate mechanisms through which local ependymoglial microenvironments may coordinate tissue remodeling during regeneration. The analysis supports organized interaction hypotheses, but it does not by itself establish causal ligand–receptor signaling or direct geometric correspondence with the wound boundary.

---

## 1. Correspondence-ready response to the reviewer

### Reviewer comment

> Figure 5c, right: this is a heterogeneous and interesting pattern that does not follow the contours of what is presumably the site of injury. The conclusion that there are localized ongoing cellular networks of either communication or migration makes sense. These cell populations should be pulled out and analyzed, using gene ontology or network analysis, or even the authors' own method of spatiotemporal cell–cell interactions, which would help demonstrate that these are organized clusters of cells.

### Response

We thank the reviewer for this constructive suggestion. We agree that the heterogeneous pattern in the right panel of Figure 5c should be resolved into explicit cell populations and tested for organized cell–cell interactions rather than interpreted from the cosine map alone. We therefore added a spatially constrained network and ligand–receptor analysis using the corrected CytoBridge model outputs at 5 DPI.

We first defined the spatial domains without using cell-type annotations. For each cell in the frozen 1,454-cell Figure 5c region of interest, the full and interaction spatial-velocity vectors were independently projected into the spatial basis and their cell-wise cosine similarity was calculated. Cells in the upper quartile of this cosine distribution were connected on the physical-radius graph used by the trained interaction model. Connected components containing at least 20 cells and internal selected model edges were retained. This label-blind procedure identified two discrete components containing 203 and 77 cells, respectively. Cell identities were used only after component detection to interpret the internal network structure.

The first domain was dominated by sfrpEGC→sfrpEGC edges, which accounted for 51.1% of selected attention, together with bidirectional VLMC↔sfrpEGC edges, which accounted for a further 24.7%. The second domain was dominated by wntEGC→wntEGC, reaEGC→reaEGC, and reaEGC↔wntEGC edges, contributing 61.2%, 27.6%, and 11.3% of selected attention, respectively. To determine whether these networks could be explained by cell-type composition alone, we compared selected attention per domain cell with 9,999 random cell sets drawn from the same region while preserving the exact number of every cell type. Selected attention was enriched 1.69-fold in the sfrpEGC–VLMC-associated domain and 3.06-fold in the reaEGC–wntEGC-associated domain relative to these composition-matched null distributions. Both empirical P values were 0.0001.

We next integrated the selected CytoBridge edges with ligand and receptor expression. The sfrpEGC–VLMC-associated domain showed significant enrichment of AGRN, LAMININ, TENASCIN, FGF, and THBS programs, consistent with a matrix-remodeling and trophic-support state. The cell-state-resolved axes included AGRN–DAG1 and TNC–SDC4 signaling within sfrpEGCs, THBS1–SDC4 within sfrpEGCs, and a directional FGF7–FGFR1 axis from VLMCs to sfrpEGCs. By contrast, the reaEGC–wntEGC-associated domain showed GRN, L1CAM, NRXN, SEMA3, and FN1 programs. Its representative axes included GRN–SORT1 and SEMA3F–NRP2/PLXNA3 within wntEGCs, together with L1CAM–L1CAM and FN1–ITGA5/ITGB1 within reaEGCs. All displayed pathways passed Benjamini–Hochberg correction at q < 0.05 in 1,999 composition-matched permutations. All displayed ligand–receptor pairs also passed pair-level correction across 531 tested pairs within each domain.

These results demonstrate that the heterogeneous Figure 5c field contains two localized and statistically organized interaction domains with distinct candidate molecular programs. The first is consistent with extracellular-matrix organization and stromal trophic support of an sfrpEGC-associated microenvironment. The second is consistent with reactive ependymoglial adhesion, guidance, and local state coordination. We have added this analysis as Supplementary Figure SXX and revised the corresponding Results and Discussion text. Because a pixel-level wound contour was not available, we do not claim that these domains reproduce the exact wound geometry. We also describe the ligand–receptor axes as model-supported mechanistic hypotheses rather than direct evidence of causal signaling.

---

## 2. Why this analysis answers the reviewer

The response follows one continuous chain of evidence:

1. **The heterogeneous cells are objectively extracted.** Panel a converts the continuous Figure 5c cosine field into connected physical components using a fixed upper-quartile threshold, the trained physical-neighbor radius, and a minimum component size.
2. **The components contain organized model interactions.** Panel b shows that selected attention is higher than expected from cell-type composition alone and identifies the cell states carrying that attention.
3. **The two components are biologically distinct.** Panel c resolves one matrix/trophic program and one reactive adhesion/guidance program.
4. **The pathway labels are converted into testable mechanisms.** Panel d names the ligand, receptor or receptor complex, sender cell state, and receiver cell state supporting each representative program.

The resulting conclusion is not merely that “the algorithm found hotspots.” It is that two spatially separated portions of the same heterogeneous field have different internal interaction architectures and different molecular programs. This directly addresses the reviewer’s request to demonstrate organized clusters and to analyze their populations using the authors’ interaction framework.

The figure does not require the two domains to follow a single wound contour. The reviewer’s observation that the pattern is heterogeneous is instead explained by spatially localized microenvironments. The domains may reflect cell-state-specific repair niches embedded within the broader injured tissue rather than a uniform radial response to the lesion.

---

## 3. How the spatial domains were defined

### 3.1 Starting measurement

The right panel of the accepted Figure 5c reports, for each cell, the cosine similarity between:

- the first two dimensions of the **full spatial velocity**, and
- the first two dimensions of the **interaction spatial velocity**.

The two velocity terms were projected independently into the spatial basis before the cell-wise cosine was calculated. A positive cosine therefore indicates that the interaction contribution points in a direction aligned with the full local spatial dynamics. It does not directly measure ligand–receptor activity, migration speed, or distance from the wound.

### 3.2 Fixed segmentation rule

- Frozen Figure 5c ROI: **1,454 cells**.
- High-cosine threshold: within-ROI upper quartile, **cosine ≥ 0.608**.
- Connectivity: physical-radius graph with the trained interaction cutoff, **0.03154** in normalized spatial coordinates.
- Minimum component size: **20 cells**.
- Retention condition: the component contained at least one internal selected model edge.
- Biological annotation: assigned only after spatial component detection from the internal selected-edge structure.

This rule produced exactly two retained domains:

| Domain | Cells | Internal selected edges | Selected attention | Attention per domain cell |
|---|---:|---:|---:|---:|
| N1, sfrpEGC–VLMC-associated | 203 | 419 | 581.60 | 2.865 |
| N2, reaEGC–wntEGC-associated | 77 | 122 | 247.12 | 3.209 |

The domain names describe their dominant selected-edge structure, not mutually exclusive cell-type composition. N1 also contains neuronal and progenitor populations, including scgnIN, tlNBL, MSN, and WSN cells. N2 also contains dpEX, sstIN, mpEX, and mpIN cells. The terminology therefore should remain “sfrpEGC–VLMC-associated domain” and “reaEGC–wntEGC-associated domain,” rather than implying pure cell-type clusters.

---

## 4. Panel-by-panel biological interpretation

### Panel a: Spatial interaction domains

Panel a preserves the original continuous full-versus-interaction spatial-velocity cosine field and overlays the boundaries of the two connected domains. The map shows that high alignment is locally concentrated but is not restricted to one smooth anatomical contour. This is the pattern that motivated the reviewer’s question.

The important result is the existence of two connected regions after applying one fixed, label-blind spatial rule. The outlines are not hand-drawn anatomical regions and were not defined by sfrpEGC, wntEGC, reaEGC, or VLMC labels. The subsequent cell-state differences therefore interpret the detected components rather than construct them.

The original ARISTA study identified injury-induced ependymoglial cells at the wound site and proposed that they arise from local resident ependymoglial cells before contributing to progenitor and neuronal state transitions. The two domains identified here refine that general framework by suggesting that injury-responsive ependymoglial states participate in more than one local interaction program at 5 DPI ([Wei et al., *Science*, 2022](https://doi.org/10.1126/science.abp9444)).

### Panel b: Organized cell-state interactions

Panel b provides the formal organization test. Its null model samples random cells from the same 1,454-cell ROI while preserving the exact cell-type counts of each observed domain. This is a **cell-type-composition-matched null**: it asks whether a domain carries more selected model attention than would be expected from having the same mixture of annotated cell types elsewhere in the ROI.

| Domain | Observed attention per cell | Null mean ± s.d. | Fold over null | Empirical P |
|---|---:|---:|---:|---:|
| N1 | 2.865 | 1.698 ± 0.164 | 1.69× | 0.0001 |
| N2 | 3.209 | 1.048 ± 0.303 | 3.06× | 0.0001 |

Thus, the elevated interaction signal cannot be attributed solely to the fact that the domains contain particular cell types. N2 shows the larger relative enrichment, whereas N1 contains the larger absolute network and more internal selected edges.

The edge decomposition identifies different modes of organization:

- **N1:** sfrpEGC→sfrpEGC edges account for 51.1% of selected attention. VLMC↔sfrpEGC edges account for 24.7%. The remaining selected edges account for 24.2%. This pattern is consistent with an sfrpEGC-centered network that combines strong local reinforcement with cross-state input from a VLMC-like stromal population.
- **N2:** wntEGC→wntEGC edges account for 61.2% of selected attention. reaEGC→reaEGC edges contribute 27.6%, and reaEGC↔wntEGC edges contribute 11.3%. N2 is therefore dominated by coordinated ependymoglial-state interactions rather than by many unrelated cell-state pairs.

“Homotypic” here means that the dominant selected edges connect different cells carrying the same annotation. It should not be interpreted as a molecular self-loop within one cell.

### Panel c: Domain-specific candidate repair programs

Panel c asks whether the exact spatial components are enriched for specific ligand–receptor programs relative to random cell sets with the same cell-type composition. Scores combine mean ligand expression in the sender state, mean receptor expression in the receiver state, and selected attention per source cell. For heteromeric complexes, all subunits are required and the minimum subunit expression is used. Each domain was compared with 1,999 random sets, and pathway P values were corrected within that domain.

The displayed pathways are representative FDR-significant programs chosen to communicate the distinct biology of the two domains. They are not the complete result. Of 80 tested pathways, 23 were significant in N1 and 18 were significant in N2.

#### N1: matrix/trophic program

| Pathway | Fold over composition-matched null | BH q |
|---|---:|---:|
| AGRN | 2.78× | 0.0050 |
| LAMININ | 2.61× | 0.0050 |
| TENASCIN | 2.54× | 0.0073 |
| FGF | 2.22× | 0.0200 |
| THBS | 1.65× | 0.0274 |

AGRN, LAMININ, TENASCIN, and THBS collectively point to extracellular-matrix assembly, anchoring, and remodeling. The FGF program adds a trophic component. This combination suggests that N1 may represent a local scaffold-and-support environment rather than a purely neuronal signaling field. Laminin α2 and dystroglycan are experimentally linked to glial and cell–matrix organization in the mammalian brain, providing biological plausibility for interpreting the LAMININ signal as an anchoring program ([Menezes et al., *J. Neurosci.*, 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC6608454/)).

#### N2: reactive adhesion/guidance program

| Pathway | Fold over composition-matched null | BH q |
|---|---:|---:|
| GRN | 5.93× | 0.0057 |
| L1CAM | 4.34× | 0.0057 |
| NRXN | 3.66× | 0.0057 |
| SEMA3 | 4.28× | 0.0178 |
| FN1 | 5.17× | 0.0182 |

The N2 program combines secreted trophic or stress-response signaling with homophilic cell adhesion, synaptic adhesion machinery, guidance cues, and matrix–integrin engagement. This mixture is consistent with local organization of reactive ependymoglial states and their interfaces with emerging neural cells. It should be described as a candidate adhesion/guidance program, not as proof of mature synapse formation.

### Panel d: Candidate ligand–receptor mechanisms

Panel d resolves the pathway-level result into cell-state-specific molecular hypotheses. All 531 retained ligand–receptor pairs were tested independently within each domain. The empirical P values were corrected across all 531 pairs. Within each pathway displayed in panel c, the plotted pair was selected using a fixed rule: minimum pair-level BH q, followed by maximum observed pair score and maximum fold enrichment.

#### N1 axes

| Ligand–receptor axis | Dominant cell-state direction | Fold over null | BH q | Biological interpretation |
|---|---|---:|---:|---|
| AGRN–DAG1 | sfrpEGC→sfrpEGC | 2.75× | 0.0156 | ECM/basement-membrane anchoring and coupling to the cell surface |
| LAMA2–DAG1 | reaEGC→reaEGC | 10.82× | 0.0156 | Laminin–dystroglycan anchoring within a small reactive EGC contribution to N1 |
| TNC–SDC4 | sfrpEGC→sfrpEGC | 2.63× | 0.0156 | Dynamic matrix engagement and potential control of adhesion or motility |
| FGF7–FGFR1 | VLMC→sfrpEGC | 6.45× | 0.0204 | Directional stromal-to-EGC trophic-support hypothesis |
| THBS1–SDC4 | sfrpEGC→sfrpEGC | 1.94× | 0.0204 | Matrix remodeling and adhesion-associated signaling |

The most informative directional mechanism in N1 is FGF7–FGFR1. VLMC→sfrpEGC contributes 86.2% of the total score for this pair, making it substantially more cell-state-specific than the other displayed axes. FGF7 has experimentally supported roles in wound repair and progenitor-supporting stromal signaling in other tissues, which makes this direction biologically plausible while not proving conservation in axolotl brain regeneration ([Huang et al., *PLoS Biology*, 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6368328/); [Yamamoto et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9266578/)).

The LAMA2–DAG1 direction is reaEGC→reaEGC even though the domain is named sfrpEGC–VLMC-associated. This is not a labeling inconsistency. The component is heterogeneous and contains seven reaEGCs. Its name reflects the dominant selected-edge architecture, not the exclusive presence of two cell types. The LAMA2 result suggests that a smaller reactive-EGC subnetwork may coexist within the larger N1 matrix domain.

#### N2 axes

| Ligand–receptor axis | Dominant cell-state direction | Fold over null | BH q | Biological interpretation |
|---|---|---:|---:|---|
| GRN–SORT1 | wntEGC→wntEGC | 5.97× | 0.0140 | Candidate trophic, stress-response, or protein-trafficking signal |
| L1CAM–L1CAM | reaEGC→reaEGC | 4.39× | 0.0140 | Homophilic adhesion and local process alignment |
| NRXN2–NLGN2 | wntEGC→wntEGC | 4.79× | 0.0106 | Cell-contact and synaptic-organization machinery |
| SEMA3F–NRP2/PLXNA3 | wntEGC→wntEGC | 10.45× | 0.0140 | Secreted neural-guidance cue with a defined receptor complex |
| FN1–ITGA5/ITGB1 | reaEGC→reaEGC | 32.05× | 0.0275 | Highly specific matrix–integrin engagement candidate |

L1CAM homophilic binding can support neurite extension and axonal organization, which makes the reaEGC→reaEGC L1CAM axis consistent with a coordinated adhesive state ([Lemmon et al., *Neuron*, 1989](https://pubmed.ncbi.nlm.nih.gov/2627381/)). NRXN2 can bind NLGN2 and participate in cell-contact-dependent synaptic organization, although its appearance here should not be equated with a mature functional synapse ([Gauthier et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3204930/)).

The SEMA3F receptor complex is particularly coherent: wntEGC→wntEGC contributes 94.6% of its total domain score. Loss-of-function studies have established SEMA3F–NRP2–PLXNA3 as a neural-patterning and axon-guidance system in vertebrates ([Schwarz et al., 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2814064/)). The present result therefore supports a guidance-related hypothesis for the wntEGC-associated domain, but it does not determine whether SEMA3F is attractive, repulsive, or synapse-modifying in this axolotl context.

FN1–ITGA5/ITGB1 has the largest fold enrichment because its null expectation is extremely low. Its absolute observed pair score is only 0.0248, compared with 17.82 for L1CAM–L1CAM. The FN1 axis should therefore be interpreted as highly domain-specific but low-amplitude, not as the strongest communication mechanism. Fibronectin binding through α5β1 integrin has independent experimental support in neuronal attachment and injured-CNS regeneration ([Tonge et al., 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3989037/); [Deng et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11283980/)).

---

## 5. Integrated biological model

The two domains are best interpreted as coexisting local repair microenvironments at 5 DPI.

### N1: scaffold stabilization and trophic support

N1 contains a large sfrpEGC-centered interaction network with substantial VLMC coupling. Its pathway and pair-level results converge on matrix proteins, cell-surface proteoglycans, dystroglycan, and FGF signaling. A parsimonious model is that this domain helps establish or remodel a permissive local scaffold while providing trophic support to resident or activated sfrpEGCs. In this model:

- sfrpEGCs reinforce a shared matrix-associated state through AGRN–DAG1, TNC–SDC4, and THBS1–SDC4;
- VLMCs provide a directional FGF7–FGFR1 input to sfrpEGCs;
- a smaller reaEGC contribution engages LAMA2–DAG1 signaling within the same spatial component.

This organization could support cell anchoring, tissue remodeling, and maintenance of a progenitor-compatible environment. The data do not establish whether N1 lies directly at the wound margin or whether it precedes N2 temporally.

### N2: reactive-state coordination and neural patterning

N2 is smaller but more strongly enriched over its composition-matched null. Its internal attention is almost entirely carried by wntEGC and reaEGC interactions. GRN–SORT1, L1CAM–L1CAM, NRXN2–NLGN2, SEMA3F–NRP2/PLXNA3, and FN1–ITGA5/ITGB1 converge on local trophic response, cell adhesion, contact organization, neural guidance, and matrix anchoring. A parsimonious model is that N2 coordinates reactive ependymoglial states and their spatial relationship to cells entering or rebuilding neural programs.

This interpretation complements the original ARISTA lineage model. The original study proposed that local resident ependymoglial cells become injury-induced progenitors and subsequently contribute to neuronal replenishment. The present analysis suggests that this response may occur within spatially distinct microenvironments: one emphasizing matrix/trophic support and another emphasizing reactive adhesion and guidance.

### What the model does not imply

- N1 and N2 are not demonstrated temporal stages.
- N1 is not a pure sfrpEGC/VLMC population, and N2 is not a pure reaEGC/wntEGC population.
- High cosine does not prove that a cell is physically migrating.
- Attention is a model influence weight, not a measured signaling flux.
- Ligand and receptor co-expression does not prove biochemical binding in this tissue.
- The analysis does not establish that either domain follows the exact wound contour.

---

## 6. Statistical and analytical strength

### Findings that are well supported

1. **Two connected high-cosine components exist under the fixed segmentation rule.** Their sizes are 203 and 77 cells, well above the minimum size of 20.
2. **The domains are not explained by cell-type composition alone.** Both observed attention values exceeded every one of the 9,999 composition-matched random values, yielding empirical P = 0.0001 after the standard plus-one correction.
3. **The dominant cell-state networks are quantitatively distinct.** N1 is centered on sfrpEGC homotypic and VLMC–sfrpEGC edges. N2 is centered on wntEGC and reaEGC edges.
4. **The displayed pathway and pair-level enrichments survive multiple-testing correction.** The ten displayed pathways have q < 0.05. The ten displayed pairs have pair-level q values from 0.0106 to 0.0275 after testing 531 pairs in each domain.
5. **Several molecular axes have a concentrated cell-state direction.** FGF7–FGFR1 is 86.2% attributable to VLMC→sfrpEGC, and SEMA3F–NRP2/PLXNA3 is 94.6% attributable to wntEGC→wntEGC.

### Findings that should remain hypotheses

1. **Causality.** No ligand, receptor, or interaction was perturbed experimentally.
2. **Spatial independence.** The composition-matched null does not preserve local spatial geometry. It demonstrates enrichment beyond composition, but not beyond every possible form of spatial autocorrelation.
3. **Model independence.** Domain detection and selected-edge analysis both use outputs from the same fitted model. The components are not selected by attention magnitude, but the evidence is not an independent biological validation of the model.
4. **Cross-species LR annotation.** Axolotl features were mapped to human gene symbols and evaluated against human CellChatDB. CellChatDB provides a curated interaction prior and handles heteromeric complexes, but conservation of every pair in axolotl is not guaranteed ([Jin et al., *Nature Communications*, 2021](https://doi.org/10.1038/s41467-021-21246-9)).
5. **Biological replication.** The 5-DPI analysis is derived from one observed section. Cells are sampling units within that section, not biological replicates.
6. **Wound geometry.** A pixel-level wound mask was unavailable, preventing a formal distance-to-wound or boundary-alignment test.

### Recommended strength of language

Use:

> localized, organized, model-supported interaction domains with distinct candidate molecular programs

Avoid:

> proven communication centers that causally drive regeneration

---

## 7. Manuscript-ready Results text

### Local interaction domains resolve the heterogeneous spatial-velocity pattern

The cell-wise alignment between full and interaction spatial velocity at 5 DPI was spatially heterogeneous, suggesting that cell–cell interactions contribute to local dynamics in discrete portions of the injured tissue. We selected cells in the upper quartile of the full-versus-interaction spatial-velocity cosine and connected them on the physical-radius graph used by the trained interaction model. Components containing at least 20 cells identified two localized domains of 203 and 77 cells. Cell-type labels were not used during component detection.

The larger domain was dominated by sfrpEGC→sfrpEGC and VLMC↔sfrpEGC edges, which accounted for 51.1% and 24.7% of selected attention, respectively. The smaller domain was dominated by wntEGC→wntEGC, reaEGC→reaEGC, and reaEGC↔wntEGC edges, which accounted for 61.2%, 27.6%, and 11.3% of selected attention. Relative to 9,999 random cell sets with the same cell-type composition, selected attention per cell was enriched 1.69-fold in the sfrpEGC–VLMC-associated domain and 3.06-fold in the reaEGC–wntEGC-associated domain (both empirical P = 0.0001).

Ligand–receptor analysis further separated the two domains. The sfrpEGC–VLMC-associated domain was enriched for AGRN, LAMININ, TENASCIN, FGF, and THBS programs and contained a directional FGF7–FGFR1 axis from VLMCs to sfrpEGCs. The reaEGC–wntEGC-associated domain was enriched for GRN, L1CAM, NRXN, SEMA3, and FN1 programs, including wntEGC-associated SEMA3F–NRP2/PLXNA3 and reaEGC-associated L1CAM–L1CAM and FN1–ITGA5/ITGB1 axes. All displayed pathways and ligand–receptor pairs remained significant after Benjamini–Hochberg correction. These results resolve the heterogeneous Figure 5c pattern into two localized interaction domains associated with matrix/trophic support and reactive adhesion/guidance, respectively.

---

## 8. Manuscript-ready Discussion text

The localized interaction domains suggest that the regenerative response is organized into distinct microenvironments rather than following a spatially uniform program. The sfrpEGC–VLMC-associated domain combines ECM-receptor interactions with a directional VLMC→sfrpEGC FGF7–FGFR1 axis, supporting a hypothesis in which stromal and ependymoglial states cooperate to establish a matrix-rich, trophically supported environment. The reaEGC–wntEGC-associated domain instead combines homophilic adhesion, neurexin–neuroligin contact, semaphorin guidance, and fibronectin–integrin engagement. This program may help coordinate reactive ependymoglial states and local neural reorganization. Together, the two domains provide a spatially resolved extension of the proposed injury-induced ependymoglial response in the axolotl telencephalon. Because the interaction scores integrate model attention with mapped ligand–receptor expression, these axes should be treated as experimentally testable mechanisms rather than evidence of causal signaling.

---

## 9. Manuscript-ready figure caption

**Supplementary Figure SXX. Local interaction domains underlying the heterogeneous spatial-velocity pattern in the ARISTA dataset.** (a) Cell-wise cosine similarity between full and interaction spatial velocity in the frozen 5-DPI Figure 5c region. Cells in the upper quartile were connected on the trained physical-radius graph. Components containing at least 20 cells defined two spatial domains. (b) Selected attention per domain cell compared with 9,999 random cell sets from the same region that preserved the exact cell-type composition of each domain. Diamonds denote the observed domains. Circles and error bars denote the null mean plus or minus s.d. The sfrpEGC–VLMC- and reaEGC–wntEGC-associated domains showed 1.69-fold and 3.06-fold enrichment, respectively (both empirical P = 0.0001). Bars summarize the cell-state pairs carrying selected attention. (c) Representative ligand–receptor pathway scores at 5 DPI relative to 1,999 cell-type-composition-matched permutations. All displayed pathways passed Benjamini–Hochberg correction at q < 0.05. (d) Candidate cell-state-resolved ligand–receptor axes within the exact spatial domains. Within each pathway shown in panel c, the pair with the smallest pair-level BH q was selected, followed by the largest observed pair score and fold enrichment. All displayed pairs passed BH correction across 531 tested pairs within their domain. Bars show enrichment over the composition-matched null. The axes represent model-supported mechanistic hypotheses rather than causal signaling measurements.

---

## 10. Concise take-home message for the response letter

> Further analysis of the heterogeneous Figure 5c field identified two spatially connected interaction domains that were significantly enriched over cell-type-composition-matched null models. An sfrpEGC–VLMC-associated domain was linked to ECM and trophic signaling, including VLMC→sfrpEGC FGF7–FGFR1, whereas a reaEGC–wntEGC-associated domain was linked to reactive adhesion and neural-guidance programs, including L1CAM, NRXN–NLGN, SEMA3F–NRP2/PLXNA3, and FN1–integrin axes. These findings demonstrate that the heterogeneous field contains organized local interaction networks and provide specific, experimentally testable mechanisms for their potential roles during regeneration.

---

## 11. Analysis provenance

- Accepted corrected package run: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1`
- Spatial-domain and matched-null analysis: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_timecourse_v1`
- Pair-level LR analysis: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_lr_axes_v1`
- Final figure bundle: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_reviewer_figure_clean_v9_mechanism_final`
- Final plotting script: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/scripts/arista_paper_equivalent/plot_figure5c_two_niche_reviewer_figure_clean.py`
- Server run: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/arista-package-native-spatialqc-z50-20260824-r1`

The correspondence, Results text, Discussion text, and caption above all use values recalculated from the metric tables archived with the final figure bundle.

## References used for biological interpretation

1. Wei X, et al. Single-cell Stereo-seq reveals induced progenitor cells involved in axolotl brain regeneration. *Science*. 2022. [doi:10.1126/science.abp9444](https://doi.org/10.1126/science.abp9444)
2. Jin S, et al. Inference and analysis of cell-cell communication using CellChat. *Nature Communications*. 2021. [doi:10.1038/s41467-021-21246-9](https://doi.org/10.1038/s41467-021-21246-9)
3. Menezes MJ, et al. The extracellular matrix protein laminin α2 regulates the maturation and function of the blood–brain barrier. *Journal of Neuroscience*. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6608454/)
4. Huang X, et al. Insulin resistance disrupts epithelial repair and niche-progenitor FGF signaling during chronic liver injury. *PLoS Biology*. 2019. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6368328/)
5. Lemmon V, Farr KL, Lagenaur C. L1-mediated axon outgrowth occurs via a homophilic binding mechanism. *Neuron*. 1989. [PubMed](https://pubmed.ncbi.nlm.nih.gov/2627381/)
6. Gauthier J, et al. Truncating mutations in NRXN2 and NRXN1 in autism spectrum disorders and schizophrenia. 2011. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC3204930/)
7. Schwarz Q, et al. Plexin A3 and plexin A4 convey semaphorin signals during facial nerve development. 2008. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC2814064/)
8. Tonge DA, et al. Fibronectin supports neurite outgrowth and axonal regeneration of adult brain neurons in vitro. 2012. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC3989037/)
9. Deng K, et al. Augmenting fibronectin levels in injured adult CNS promotes robust axon regeneration in vivo. 2024. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11283980/)
