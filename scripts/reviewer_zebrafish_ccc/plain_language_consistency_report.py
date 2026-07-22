#!/usr/bin/env python3
"""Build a self-contained, plain-language guide to the zebrafish CCC results.

This script deliberately leaves the frozen reviewer bundle untouched.  It
turns the same tables into three more direct figures and a Chinese reading
guide that separates positive evidence, limitations, and audit-only panels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


STAGE_LABELS = {
    0.0: "5.25 hpf",
    1.0: "10 hpf",
    2.0: "12 hpf",
    3.0: "18 hpf",
    4.0: "24 hpf",
}
ORIGINAL_FIGURES = [
    "rank_concordance",
    "top_edge_overlap",
    "condition_coverage",
    "directionality_concordance",
    "stage_stability",
    "cytobridge_control_panel",
    "positive_consistency_overview",
    "top_signal_biology",
    "reviewer_validation_axes",
    "spatial_lr_interaction_maps",
    "ccc_circle_comparison",
    "known_lr_temporal_consistency_bubble",
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--bundle-dir", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--overwrite", action="store_true")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def require(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def read_table(bundle: Path, name: str, required: list[str]) -> pd.DataFrame:
    path = bundle / "tables" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    require(frame, required, path)
    return frame


def top_set(group: pd.DataFrame, column: str, requested: int) -> set[int]:
    values = pd.to_numeric(group[column], errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return set()
    boundary = finite.nlargest(min(requested, len(finite))).iloc[-1]
    return set(group.index[np.isfinite(values) & (values >= boundary)])


def prepare_stage_table(scores: pd.DataFrame, reported: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    formal = reported.loc[
        reported["target"].eq("CytoBridge attention")
        & reported["reference"].eq("External native consensus")
    ].set_index("stage")
    for stage, group in scores.groupby("stage", sort=True):
        group = group.copy()
        requested = max(1, int(round(len(group) * 0.20)))
        left = top_set(group, "cytobridge_attention", requested)
        right = top_set(group, "external_native_consensus", requested)
        shared = left & right
        group["attention_top20"] = group.index.isin(left)
        group["external_top20"] = group.index.isin(right)
        group["shared_top20"] = group.index.isin(shared)
        group["external_consensus_rank"] = group["external_native_consensus"].rank(
            method="average", pct=True
        )
        expected = formal.loc[float(stage)]
        observed = (len(left), len(right), len(shared))
        target = (
            int(expected["target_set_size_after_boundary_ties"]),
            int(expected["reference_set_size_after_boundary_ties"]),
            int(expected["intersection"]),
        )
        if observed != target:
            raise AssertionError(
                f"Top-set reconstruction differs at stage {stage}: {observed} != {target}"
            )
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def stage_summary(stage_table: pd.DataFrame, consensus: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
    consensus = consensus.loc[
        consensus["design"].eq("external_only_native_primary")
        & consensus["target"].eq("CytoBridge attention")
    ].copy()
    top = top.loc[
        top["target"].eq("CytoBridge attention")
        & top["reference"].eq("External native consensus")
    ].copy()
    selected = [
        "stage",
        "stage_label",
        "n_directed_pairs",
        "spearman",
    ]
    result = consensus[selected].merge(
        top[
            [
                "stage",
                "top_k_requested",
                "target_set_size_after_boundary_ties",
                "reference_set_size_after_boundary_ties",
                "intersection",
                "overlap_fraction_of_smaller_set",
                "overlap_enrichment_over_random",
                "bh_q_within_target_reference_family",
            ]
        ],
        on="stage",
        validate="one_to_one",
    )
    return result.sort_values("stage").reset_index(drop=True)


def plot_rank_scatter(stage_table: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18.0, 4.4), sharex=True, sharey=True)
    for ax, row in zip(axes, summary.itertuples(index=False)):
        group = stage_table.loc[stage_table["stage"].eq(row.stage)]
        other = group.loc[~group["shared_top20"]]
        shared = group.loc[group["shared_top20"]]
        ax.add_patch(
            Rectangle((0.8, 0.8), 0.2, 0.2, facecolor="#EEE8FF", edgecolor="none", zorder=0)
        )
        ax.scatter(
            other["external_consensus_rank"],
            other["cytobridge_attention_rank"],
            s=18,
            color="#C7CBD1",
            alpha=0.72,
            linewidths=0,
            zorder=2,
        )
        ax.scatter(
            shared["external_consensus_rank"],
            shared["cytobridge_attention_rank"],
            s=48,
            color="#6F42C1",
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.axvline(0.8, color="#8E7CC3", lw=0.9, ls="--")
        ax.axhline(0.8, color="#8E7CC3", lw=0.9, ls="--")
        ax.plot([0, 1], [0, 1], color="#8B9198", lw=0.8, ls=":")
        ax.set_title(str(row.stage_label), fontsize=12, weight="bold")
        ax.text(
            0.04,
            0.96,
            f"rank correlation = {row.spearman:.2f}\nshared top 20% = {int(row.intersection)}/{int(row.target_set_size_after_boundary_ties)}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.2,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#D5D8DC", "alpha": 0.94},
        )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.5, 0.8, 1.0])
        ax.set_yticks([0, 0.5, 0.8, 1.0])
        ax.grid(color="#ECEFF1", lw=0.65)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("CytoBridge attention rank\nweak  →  strong", fontsize=11)
    fig.supxlabel("External-only consensus rank   weak  →  strong", fontsize=11, y=0.025)
    fig.suptitle(
        "Do CytoBridge and independent external methods rank the same cell-type arrows highly?",
        fontsize=14,
        weight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.075,
        "Each dot is one directed sender→receiver cell-type pair. Purple dots are in both top-20% sets.",
        ha="center",
        fontsize=10,
        color="#4A4F55",
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def pair_label(sender: str, receiver: str) -> str:
    if sender == receiver:
        return f"{sender} → itself (self-loop)"
    return f"{sender} → {receiver}"


def make_pair_checklist(stage_table: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for stage in (1.0, 4.0):
        group = stage_table.loc[stage_table["stage"].eq(stage)].copy()
        group["pair"] = [
            pair_label(str(sender), str(receiver))
            for sender, receiver in zip(group["sender_type"], group["receiver_type"])
        ]
        group["combined_rank"] = (
            group["cytobridge_attention_rank"] + group["external_consensus_rank"]
        ) / 2
        shared = group.loc[group["shared_top20"]].nlargest(8, "combined_rank").copy()
        shared["category"] = "shared_top20"
        attention_only = group.loc[
            group["attention_top20"] & ~group["external_top20"]
        ].nlargest(3, "cytobridge_attention_rank").copy()
        attention_only["category"] = "cytobridge_only_top20"
        external_only = group.loc[
            group["external_top20"] & ~group["attention_top20"]
        ].nlargest(3, "external_consensus_rank").copy()
        external_only["category"] = "external_only_top20"
        outputs.extend([shared, attention_only, external_only])
    selected = pd.concat(outputs, ignore_index=True)
    return selected[
        [
            "stage",
            "sender_type",
            "receiver_type",
            "pair",
            "category",
            "cytobridge_attention_rank",
            "external_consensus_rank",
        ]
    ]


def plot_pair_checklist(checklist: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 10.0))
    category_meta = {
        "shared_top20": ("AGREE", "#E8F5E9", "#207245"),
        "cytobridge_only_top20": ("CB ONLY", "#FFF3E0", "#A05A00"),
        "external_only_top20": ("EXT ONLY", "#E3F2FD", "#1C6395"),
    }
    for ax, stage in zip(axes, (1.0, 4.0)):
        ax.set_axis_off()
        row = summary.loc[summary["stage"].eq(stage)].iloc[0]
        local = checklist.loc[checklist["stage"].eq(stage)].copy()
        ax.text(
            0.0,
            1.04,
            STAGE_LABELS[stage],
            transform=ax.transAxes,
            fontsize=17,
            weight="bold",
            va="top",
        )
        ax.text(
            0.0,
            0.99,
            (
                f"Both rank {int(row.intersection)} arrows in their top 20% "
                f"({row.overlap_enrichment_over_random:.2f}× random expectation)."
            ),
            transform=ax.transAxes,
            fontsize=10.5,
            color="#40464D",
            va="top",
        )
        y = 0.92
        for category in ("shared_top20", "cytobridge_only_top20", "external_only_top20"):
            block = local.loc[local["category"].eq(category)]
            badge, background, color = category_meta[category]
            if category == "shared_top20":
                heading = "Concrete arrows both methods call high"
            elif category == "cytobridge_only_top20":
                heading = "Examples high only in CytoBridge"
            else:
                heading = "Examples high only in external consensus"
            ax.text(0.0, y, heading, transform=ax.transAxes, fontsize=11, weight="bold", va="top")
            y -= 0.038
            for item in block.itertuples(index=False):
                height = 0.067 if category == "shared_top20" else 0.061
                ax.add_patch(
                    Rectangle(
                        (0.0, y - height + 0.006),
                        1.0,
                        height,
                        transform=ax.transAxes,
                        facecolor=background,
                        edgecolor="white",
                        linewidth=1.0,
                    )
                )
                ax.text(
                    0.012,
                    y - 0.012,
                    badge,
                    transform=ax.transAxes,
                    fontsize=8.4,
                    weight="bold",
                    color=color,
                    va="top",
                )
                wrapped = textwrap.fill(str(item.pair), width=47)
                ax.text(
                    0.125,
                    y - 0.006,
                    wrapped,
                    transform=ax.transAxes,
                    fontsize=8.7,
                    color="#25292D",
                    va="top",
                )
                ax.text(
                    0.985,
                    y - 0.006,
                    f"CB {item.cytobridge_attention_rank:.2f} | EXT {item.external_consensus_rank:.2f}",
                    transform=ax.transAxes,
                    fontsize=7.8,
                    color="#4F555B",
                    va="top",
                    ha="right",
                )
                y -= height
            y -= 0.026
    fig.suptitle(
        "Which sender→receiver arrows are actually consistent?",
        fontsize=16,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Ranks run from 0 (weak) to 1 (strong). The formal top-20% rule includes self-loops and all boundary ties.",
        ha="center",
        fontsize=9.5,
        color="#4A4F55",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_evidence_map(out: Path) -> None:
    rows = [
        (
            "Independent methods rank cell-type arrows similarly",
            "SUPPORTED",
            "External-only consensus vs attention: mean rank correlation = 0.53; all five stages are positive.",
        ),
        (
            "Direct agreement with a spatial CCC method",
            "SUPPORTED",
            "COMMOT is the strongest direct external comparison: mean stage correlation = 0.566.",
        ),
        (
            "The strongest signals overlap more than random",
            "SUPPORTED",
            "Top-20% overlap averages 1.88× random expectation, but strength varies by stage.",
        ),
        (
            "Known signaling biology appears near the top",
            "SUPPORTED",
            "CXCL, NOTCH and non-canonical WNT pathways are enriched; NicheNet overlap is 50–100% in tested units.",
        ),
        (
            "High signals occupy similar spatial neighborhoods",
            "SUPPORTED",
            "Wnt5b→Fzd7a and Dla→Notch1a show strong CytoBridge→COMMOT spatial co-localization; CXCL is partial.",
        ),
        (
            "The result is not merely 'near cells score higher'",
            "PARTIAL",
            "Raw attention vs inverse distance is near zero, but the graph is already spatially local by construction.",
        ),
        (
            "Attention gives the exact ligand→receptor direction",
            "NOT SHOWN",
            "Forward LR residual association is positive, but reverse is not weaker; direction specificity is not established.",
        ),
        (
            "Virtual removal proves a causal perturbation response",
            "NOT SHOWN",
            "It is a one-model sensitivity analysis, not an experimental perturbation or causal test.",
        ),
    ]
    styles = {
        "SUPPORTED": ("#E7F5EC", "#176B3A"),
        "PARTIAL": ("#FFF4D6", "#8A5A00"),
        "NOT SHOWN": ("#FDE9E7", "#A5322A"),
    }
    fig, ax = plt.subplots(figsize=(15.5, 9.0))
    ax.set_axis_off()
    ax.text(
        0.0,
        1.03,
        "Can the current results answer the reviewer's concern?",
        transform=ax.transAxes,
        fontsize=20,
        weight="bold",
        va="top",
    )
    ax.text(
        0.0,
        0.975,
        "Short answer: yes for communication-relevant organization; no for literal biochemical strength, exact direction, or causality.",
        transform=ax.transAxes,
        fontsize=12,
        color="#3F454B",
        va="top",
    )
    y, height = 0.91, 0.101
    for question, status, evidence in rows:
        background, color = styles[status]
        ax.add_patch(
            Rectangle(
                (0.0, y - height + 0.008),
                1.0,
                height - 0.008,
                transform=ax.transAxes,
                facecolor=background,
                edgecolor="white",
                linewidth=1.0,
            )
        )
        ax.text(
            0.018,
            y - 0.018,
            status,
            transform=ax.transAxes,
            fontsize=10.5,
            weight="bold",
            color=color,
            va="top",
        )
        ax.text(
            0.165,
            y - 0.012,
            question,
            transform=ax.transAxes,
            fontsize=11.2,
            weight="bold",
            color="#202428",
            va="top",
        )
        ax.text(
            0.165,
            y - 0.050,
            evidence,
            transform=ax.transAxes,
            fontsize=9.7,
            color="#454A50",
            va="top",
        )
        y -= height
    ax.text(
        0.0,
        0.012,
        "Recommended wording: attention captures communication-relevant, biologically coherent interaction organization.",
        transform=ax.transAxes,
        fontsize=11,
        weight="bold",
        color="#32205F",
        va="bottom",
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_figure_index() -> pd.DataFrame:
    rows = [
        ("01_reviewer_evidence_map", "新主图", "审稿人的各项疑问目前分别得到什么答案", "结论总览；先看这一张"),
        ("02_external_consensus_rank_scatter", "新主图", "每个细胞类型箭头在双方排名中是否同时靠前", "最直观的跨方法总体一致性"),
        ("03_top_communication_arrows_checklist", "新主图", "具体哪些 sender→receiver 箭头一致，哪些不一致", "把抽象 overlap 还原成具体生物学对象"),
        ("spatial_lr_interaction_maps", "主证据", "三个已知 LR 轴在真实空间中的箭头分布", "WNT/NOTCH 空间一致，CXCL 部分一致"),
        ("known_lr_temporal_consistency_bubble", "主/补充证据", "已知 LR 轴随时间在两种方法中的相对强弱", "同一信号轴是否被双方同时列为高位"),
        ("top_signal_biology", "主/补充证据", "高 attention 信号富集哪些通路，并与 NicheNet 下游结果重合多少", "支持已知生物通路，但不是因果证据"),
        ("ccc_circle_comparison", "补充证据", "10 和 24 hpf 的高位细胞类型箭头网络", "适合展示整体网络；不适合精确数边"),
        ("positive_consistency_overview", "补充摘要", "共识、top overlap、self-inclusion 和距离关系的统计摘要", "信息全但较抽象，建议不作为第一张"),
        ("rank_concordance", "补充/审计", "所有方法两两排序相关", "说明 COMMOT 与 attention 最一致；矩阵本身较密"),
        ("top_edge_overlap", "补充/审计", "各方法 top-k 集合的 Jaccard 重叠", "方法定义差异大，不能单独据此判定成败"),
        ("stage_stability", "生物学审计", "同一方法相邻发育时点是否稳定", "反映发育变化，不是跨方法验证"),
        ("condition_coverage", "质量审计", "每种方法实际评估了多少细胞类型方向对", "只看覆盖范围，不是性能图"),
        ("directionality_concordance", "限制/审计", "A→B 与 B→A 的方向差是否跨方法一致", "稀疏且混合，不支持精确方向主张"),
        ("cytobridge_control_panel", "限制/审计", "距离、状态、度数、初始化和随机化控制", "attention 有部分正证据，但 exact-message 控制混合"),
        ("reviewer_validation_axes", "限制/审计", "内部 LR、方向、空间及虚拟移除测试", "不能把虚拟移除写成实验因果"),
    ]
    return pd.DataFrame(rows, columns=["figure", "recommended_role", "plain_question", "plain_conclusion"])


def report_text(summary: pd.DataFrame, checklist: pd.DataFrame) -> str:
    stage_rows = "\n".join(
        (
            f"| {row.stage_label} | {row.spearman:.3f} | "
            f"{int(row.intersection)}/{int(row.target_set_size_after_boundary_ties)} | "
            f"{row.overlap_enrichment_over_random:.2f}× | {row.bh_q_within_target_reference_family:.3g} |"
        )
        for row in summary.itertuples(index=False)
    )
    shared_10 = checklist.loc[
        checklist["stage"].eq(1.0) & checklist["category"].eq("shared_top20"), "pair"
    ].tolist()
    shared_24 = checklist.loc[
        checklist["stage"].eq(4.0) & checklist["category"].eq("shared_top20"), "pair"
    ].tolist()
    bullets_10 = "\n".join(f"- {item}" for item in shared_10)
    bullets_24 = "\n".join(f"- {item}" for item in shared_24)
    return f"""# 斑马鱼 CCC 结果：一份讲人话的读图说明

## 先看结论：这些结果能不能回复审稿人？

**能，但必须把论点说准。** 现有结果支持的是：

> CytoBridge attention 捕捉到了与细胞通信相关、具有生物学一致性的 interaction organization；它与独立空间 CCC 方法的整体排序、已知 LR/通路以及若干空间定位信号相一致。

现有结果**不支持**把 attention 直接写成：

- 生化意义上的“通信强度”或“通信概率”；
- 精确的 ligand→receptor 方向；
- 实验扰动意义上的因果效应。

这一区分非常重要。审稿人的核心担忧不是“attention 是否完美复刻每一个 CCC 软件”，而是 attention 会不会只反映距离、共同表达或模型拟合。当前结果已经提供了多层正向证据，足以回应这个担忧；但不能越界宣称 attention 就是真实信号通量。

建议先看 [图 1：审稿问题证据地图](figures/01_reviewer_evidence_map.png)，30 秒就能知道哪些结论成立、哪些只成立一部分、哪些还没证明。

![Reviewer evidence map](figures/01_reviewer_evidence_map.png)

## 审稿人究竟问了什么？

审稿人担心：高 attention 可能只是因为细胞靠得近、转录状态相似或者模型恰好这样拟合，不一定是真正的 signaling。因此她希望看到至少一种外部或生物学证据：已知 ligand–receptor、空间局部信号，或 perturbation-sensitive communication program。

我们现在给出的不是单一指标，而是四层相互补充的证据：

1. **总体排序一致：** 不把 CytoBridge 放进共识，三个外部方法仍与 attention 正相关。
2. **具体高位信号一致：** top 20% 的 cell-type arrows 超过随机预期地重叠。
3. **生物学内容一致：** 高位信号富集 CXCL、NOTCH、ncWNT 等已知通路，并与 NicheNet 的下游 ligand 排名重合。
4. **真实空间位置一致：** Wnt5b→Fzd7a 和 Dla→Notch1a 等信号在 CytoBridge 与 COMMOT 中落在相似空间邻域。

## 四个词先讲明白

### 一个“点”或一条“箭头”是什么？

它代表一个有方向的细胞类型对，例如 `Notochord → Spinal Cord Ventral Region`。方向相反的箭头是另一个对象。self-loop 表示同一细胞类型内部的潜在自分泌通信。

### CytoBridge attention 是什么？

它是模型在图边上学习到的权重摘要。数值大表示这类 sender→receiver interaction 对模型更重要。它本身不是某个特定 LR 分子的生化通量。

### External-only consensus 是什么？

先在每个发育时点内，把 COMMOT、CellAgentChat CTPS 和 CellChat triMean 各自的 cell-type arrow 分数转成 0–1 排名，再取三个排名的平均。**这里完全不包含 CytoBridge**，所以不会因为把自己的结果放进共识而人为提高相关性。

### rank correlation 怎么理解？

它只问“双方把同一批箭头排出的先后顺序像不像”：1 表示顺序完全相同，0 表示没有稳定关系，负数表示相反。它不要求两个软件的原始分数单位相同。

## 图 2：最直观的总体一致性

![External-only rank scatter](figures/02_external_consensus_rank_scatter.png)

### 这是什么？

每个小图对应一个时点；每个点是一条 sender→receiver 细胞类型箭头。横轴是三个外部方法的共识排名，纵轴是 CytoBridge attention 排名。

### 怎么画的？

所有分数只在同一时点内转成百分位排名。右上角紫色区域表示双方都把该箭头放进 top 20%；紫点是实际落在双方 top 20% 交集中的箭头。没有把五个时点的原始分数混在一起，也没有平均不同软件的原始单位。

### 怎么看？

- 点云越呈左下到右上的趋势，双方整体排序越一致。
- 紫点越多，双方对“最强的一批箭头”越有具体共识。
- 相关系数是总体排序指标；紫点交集是 top-signal 指标。两者回答不同问题，不能互相替代。

### 结果是什么？

| 时点 | 全部箭头排序相关 rho | 双方 top 20% 交集 | 相对随机预期 | 多重校正 q |
|---|---:|---:|---:|---:|
{stage_rows}

五个时点相关性全部为正，平均约为 **0.53**。10 hpf 和 24 hpf 的 top overlap 最清楚；早期 5.25 hpf 的 top overlap 较弱。因此正确说法是“总体稳定正相关、具体 top signal 的一致性具有阶段差异”，而不是“每个时点都完全一致”。

## 图 3：不要只看数字，具体哪些箭头一致？

![Concrete arrow checklist](figures/03_top_communication_arrows_checklist.png)

### 这是什么？

它把抽象的 overlap 数字还原成具体 sender→receiver 箭头。绿色 `AGREE` 是双方都排进 top 20% 的例子；橙色和蓝色行故意保留“不一致例子”，防止把结果画成只有成功案例。

### 怎么画的？

用与正式统计完全相同的 top-20% 规则：包含 self-loop，并保留落在第 k 名边界上的全部并列项。每行末尾直接写双方 0–1 排名。

### 怎么解读？

你不需要先理解 Jaccard。直接看一条生物学箭头是否同时被双方排高，以及双方排名差多大即可。

10 hpf 的代表性共同高位箭头包括：

{bullets_10}

24 hpf 的代表性共同高位箭头包括：

{bullets_24}

## 图 4：已知 LR 轴是否落在相似空间位置？

![Spatial LR maps](figures/spatial_lr_interaction_maps.png)

### 这是什么？

同一个胚胎空间中，左列画 CytoBridge 的高位细胞边，右列画 COMMOT 的高位细胞流；三行分别是 18 hpf `Wnt5b→Fzd7a`、18 hpf `Cxcl12a→Cxcr4b` 和 24 hpf `Dla→Notch1a`。

### 怎么画的？

每种方法各自从所有正分、非 self 的细胞边中取 top 80。箭头背景是相同的组织坐标和表达背景。方法分数单位不同，所以比较的是**高位箭头落在哪片空间**，不是线长或颜色数值是否完全相等。

### 怎么看？

看两列高密度箭头区域是否重合。定量上，在半个图构建 cutoff 的匹配半径内：

- WNT：92.8% 的 CytoBridge 高位中点附近存在 COMMOT 高位中点；反向为 47.3%。
- CXCL：48.6% / 21.9%，属于部分一致。
- NOTCH：99.1% / 50.0%，空间一致很强。

两个方向不对称，是因为 COMMOT 候选流通常更宽、更密；它不是算错，也不能解读为 ligand→receptor 的方向准确率。

### 结论是什么？

WNT 和 NOTCH 给出了最直观的 spatially localized signaling evidence；CXCL 是较弱但仍为正的例子。建议审稿回复中主打 WNT/NOTCH，把 CXCL 写成 partial consistency。

## 图 5：已知 LR 信号随时间是否被双方同时认为重要？

![Known LR temporal bubble](figures/known_lr_temporal_consistency_bubble.png)

### 这是什么？

每一行是一个已知 LR 轴，每一列是发育时点；圆点和方块分别代表 CytoBridge 与 COMMOT。点越大/越深，表示它在该方法、该时点内的相对排名越高。黑色外框表示双方都进入较高分位。

### 怎么画的？

先按预先登记的已知 zebrafish 轴筛选可识别 LR，然后在每个方法内部转成排名。原始 score 不直接相减，因为 COMMOT 和 attention 的单位没有可比性。

### 结论是什么？

它回答的是“已知信号轴是否在相似发育阶段被双方同时列为高位”，不是“两个方法给出的绝对强度相等”。WNT、NOTCH 等轴提供了可讲清楚的正向生物学例子。

## 图 6：高 attention 中是什么生物学通路？

![Top signal biology](figures/top_signal_biology.png)

### 左图怎么读？

取每个时点 attention×LR 排名前 20 的 LR 轴，和完整 LR 数据库背景比较。fold enrichment = 1 表示与随机背景一样；大于 1 表示某通路在高位信号中出现得更多。显著富集包括：CXCL 21.96×、NOTCH 7.68×、ncWNT 5.24×，均通过多重检验。

### 右图怎么读？

NicheNet 不直接预测空间 cell–cell communication，它问“哪些 ligand 更能解释 receiver 的下游靶基因”。因此这里比较的是**下游生物学一致性**：被 NicheNet 排高的 ligand 有多少也出现在 attention 的高位 LR 轴中。五个可测单元的重合比例为 50%–100%。

### 结论是什么？

高 attention 不是一堆无法解释的边，它集中在已知 signaling programs；NicheNet 从不同目标函数给出下游侧支持。但这不是对空间通信强度的一对一复刻，也不是完全独立于 LR 先验的验证。

## 原来的折线图、barplot 和矩阵到底是什么意思？

下面按文件逐张说明，并标注它在论文叙事中的正确位置。

### `positive_consistency_overview`

- **是什么：** 四块统计摘要：各阶段 external-only correlation、把自身放进共识导致的升高、top-20% overlap enrichment，以及与距离的关系。
- **怎么看：** A 是最重要的；B 告诉读者 self-included consensus 会把均值从 0.53 抬到约 0.78，因此正式结论必须使用 external-only；C 表示 top signals 平均约为随机预期的 1.88 倍；D 表示 attention 原始分数并不是简单“越近越高”。
- **结论：** 有用但过于压缩，适合作为补充统计摘要，不适合作为读者第一张图。

### `rank_concordance`

- **是什么：** 所有方法两两之间的平均 stage-wise Spearman 矩阵。
- **怎么看：** 红色接近 1 表示两种方法把 cell-type arrows 排得相似，0 表示没稳定关系，负值表示相反。对角线永远是自己和自己，信息量为零。
- **结论：** attention 与 COMMOT 的直接平均相关最高，为 0.566；与 CellAgentChat project-LR 为 0.208；CellChat 只有两个 stage 有有限分数，均值 0.082。它说明不同 CCC 方法并不等价，不能要求所有格子都红。
- **建议：** 补充材料或方法审计；正文改用图 2 的点云。

### `top_edge_overlap`

- **是什么：** 每种方法 top-k sender→receiver 集合之间的 Jaccard，即“交集/并集”。
- **怎么看：** 0.1 不是“10 条里相同 1 条”，而是交集占两个集合并集的 10%。零值很多也可能来自阈值、结构零和方法目标不同。
- **结论：** 可显示严格 top set 的差异，但很抽象，也容易被误读。不要单独用它回答审稿人，改看图 3 的具体箭头。

### `condition_coverage`

- **是什么：** 每种方法、每个时点实际有多少 directed type pairs 可比较。
- **结论：** 这是输入覆盖质量检查，不是方法性能。NicheNet 的空格来自它只分析特定 source→target stage/receiver unit，不代表算法失败。

### `directionality_concordance`

- **是什么：** 比较一种方法中 A→B 相对 B→A 的排名差，是否被另一方法重复。
- **结论：** 结果稀疏且混合，不足以证明精确方向。它应该作为限制或审计图，不宜主打。

### `stage_stability`

- **是什么：** 同一种方法在相邻发育时点的排序相关和 top-k 重合。
- **结论：** 它描述发育过程中网络是否变化，不是跨方法一致性。早期变化大、后期较稳定完全可能是合理生物学，不应简单理解成“越高越好”。

### `cytobridge_control_panel`

- **是什么：** 在控制 stage、cell type、距离、细胞状态和图度数后测试 LR association，并与初始化/随机化模型比较。
- **结论：** trained attention 的残差 LR association 为正且随机化后消失，是部分正证据；但初始化也有非零信号，exact-message 某些 control 甚至不更好，所以它不能作为唯一或最强证据。

### `reviewer_validation_axes`

- **是什么：** 把 LR 匹配、方向、空间和 virtual removal 放在一张内部验证图里。
- **结论：** 适合作为完整审计。virtual removal 只能说明模型对删除某些相互作用敏感，不能写成真实 perturbation causality。

### `ccc_circle_comparison`

- **是什么：** 10 和 24 hpf 的高位 cell-type communication 网络。节点是 cell type，箭头是高位 sender→receiver。
- **怎么看：** 用于看网络结构和双方共同出现的箭头，不适合逐条精确对数。
- **注意：** 为了画面可读，这张图去掉 self-loop，并只显示有限条边；因此它的边数不会与正式 top-20% 统计完全相同。正式数字以图 2/图 3 和 CSV 为准。

## 为什么 CellChat、CellAgentChat、NicheNet 的相关性不都很高？

- **COMMOT** 是最接近的外部对照：它也是空间 CCC 方法，并使用当前项目 LR 数据库，所以直接排序相关最高。
- **CellChat** 主要是非空间的群体表达统计；当前 native triMean 还在多个 stage 产生大量并列零，仅两个 stage 有可计算相关，因此均值低不奇怪。
- **CellAgentChat** 使用不同的统计和数据库/orthology 路径，project-LR 条件比 official-default 更接近 CytoBridge，但相关仍只有中低水平。
- **NicheNet** 的目标是 ligand→target gene regulation，而不是当前空间 sender→receiver strength。用它的 raw score 与 attention 做直接排序相关并不公平；它更适合图 6 右侧的 downstream ligand overlap。

因此最有说服力的论证不是“CytoBridge 与所有软件都高度相关”，而是：“与目标最接近的空间方法 COMMOT 有稳定正相关；多个方法组成的 external-only consensus 在五个时点均为正；已知 LR/pathway 和真实空间定位又提供了独立维度的生物学一致性。”

## 哪些证据有一定循环性？

CytoBridge 的 LR edge prior 使用了 LR 信息，所以“高 attention 中出现更多已知 LR”不是完全独立的盲验证。它仍能说明训练后模型把权重组织到了可解释信号上，但不能单独证明真实通信。为降低循环性，正式回复应把重点放在：

1. 不含 CytoBridge 的 external-only consensus；
2. 与 COMMOT 的直接空间比较；
3. 真实空间中的 WNT/NOTCH 局部一致性；
4. NicheNet 下游 ligand consistency；
5. 对方向性与因果性的明确限制。

## 建议给审稿人的英文回复

> We agree that attention weights should not be interpreted as direct biochemical communication strengths. We therefore revised the manuscript to describe them as interaction-associated weights and added external and biological consistency analyses. Using a consensus constructed only from COMMOT, CellAgentChat, and CellChat, thereby excluding CytoBridge itself, CytoBridge attention showed positive stage-wise rank concordance at all five developmental stages (mean Spearman rho = 0.53). The strongest direct spatial comparison was with COMMOT (mean stage-wise rho = 0.566). CytoBridge top-20% cell-type interactions overlapped the external consensus above random expectation on average (1.88-fold), with the strongest evidence at 10 and 24 hpf. High-ranked LR-associated signals were enriched for known zebrafish signaling programs, including CXCL, NOTCH, and non-canonical WNT, and Wnt5b–Fzd7a and Dla–Notch1a signals occupied similar spatial neighborhoods in CytoBridge and COMMOT. NicheNet provided complementary downstream ligand consistency (50–100% overlap in the evaluable receiver units). These results support that the learned weights capture communication-relevant and biologically coherent interaction organization, while we do not claim that attention is a calibrated CCC probability, an exact ligand-to-receptor direction, or causal perturbation evidence.

## 写稿时可以说 / 不要说

可以说：

- `communication-relevant interaction organization`
- `biologically coherent interaction-associated weights`
- `consistent with external spatial CCC rankings and known signaling programs`
- `spatially co-localized high-ranking interactions`

不要说：

- `attention is the true communication strength`
- `attention is a calibrated CCC probability`
- `the analysis proves the ligand-to-receptor direction`
- `virtual ablation proves a causal perturbation response`
- `all external methods agree strongly with CytoBridge`

## 最后给汇报者的一句话

**最强的故事不是“所有方法给出一模一样的网络”，而是“在完全排除自身的外部共识中，CytoBridge 在五个时点都呈正向排序一致；这种一致又能落到具体 cell-type arrows、已知 WNT/NOTCH/CXCL 通路和真实空间位置上，因此 attention 不是纯粹的几何或拟合产物，但它仍不等同于真实生化通信强度。”**
"""


def reviewer_reply_text() -> str:
    return """# Reviewer-response wording (plain and bounded)

We agree that attention weights should not be interpreted as direct biochemical communication strengths. We therefore revised the manuscript to describe them as interaction-associated weights and added external and biological consistency analyses. Using a consensus constructed only from COMMOT, CellAgentChat, and CellChat, thereby excluding CytoBridge itself, CytoBridge attention showed positive stage-wise rank concordance at all five developmental stages (mean Spearman rho = 0.53). The strongest direct spatial comparison was with COMMOT (mean stage-wise rho = 0.566). CytoBridge top-20% cell-type interactions overlapped the external consensus above random expectation on average (1.88-fold), with the strongest evidence at 10 and 24 hpf. High-ranked LR-associated signals were enriched for known zebrafish signaling programs, including CXCL, NOTCH, and non-canonical WNT, and Wnt5b–Fzd7a and Dla–Notch1a signals occupied similar spatial neighborhoods in CytoBridge and COMMOT. NicheNet provided complementary downstream ligand consistency (50–100% overlap in the evaluable receiver units). These results support that the learned weights capture communication-relevant and biologically coherent interaction organization, while we do not claim that attention is a calibrated CCC probability, an exact ligand-to-receptor direction, or causal perturbation evidence.
"""


def main() -> None:
    args = parser().parse_args()
    bundle = args.bundle_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not (bundle / "bundle_manifest.json").is_file():
        raise FileNotFoundError(f"Not a reviewer bundle: {bundle}")
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}")
        shutil.rmtree(output)
    figures = output / "figures"
    tables = output / "tables"
    originals = figures / "original"
    figures.mkdir(parents=True)
    tables.mkdir(parents=True)
    originals.mkdir(parents=True)

    scores = read_table(
        bundle,
        "harmonized_type_pair_scores.csv.gz",
        [
            "stage",
            "sender_type",
            "receiver_type",
            "cytobridge_attention",
            "cytobridge_attention_rank",
            "external_native_consensus",
        ],
    )
    consensus = read_table(
        bundle,
        "consensus_by_stage.csv",
        ["design", "target", "stage", "stage_label", "spearman"],
    )
    top = read_table(
        bundle,
        "top_signal_overlap_by_stage.csv",
        [
            "target",
            "reference",
            "stage",
            "intersection",
            "target_set_size_after_boundary_ties",
            "reference_set_size_after_boundary_ties",
            "overlap_enrichment_over_random",
            "bh_q_within_target_reference_family",
        ],
    )
    stage_table = prepare_stage_table(scores, top)
    summary = stage_summary(stage_table, consensus, top)
    checklist = make_pair_checklist(stage_table)

    summary.to_csv(tables / "plain_language_stage_summary.csv", index=False)
    checklist.to_csv(tables / "top_pair_checklist.csv", index=False)
    build_figure_index().to_csv(tables / "figure_reading_index.csv", index=False)

    plot_evidence_map(figures / "01_reviewer_evidence_map")
    plot_rank_scatter(stage_table, summary, figures / "02_external_consensus_rank_scatter")
    plot_pair_checklist(checklist, summary, figures / "03_top_communication_arrows_checklist")

    for name in ORIGINAL_FIGURES:
        for suffix in ("png", "pdf"):
            source = bundle / "figures" / f"{name}.{suffix}"
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, originals / source.name)
            if name in {
                "spatial_lr_interaction_maps",
                "known_lr_temporal_consistency_bubble",
                "top_signal_biology",
                "ccc_circle_comparison",
                "positive_consistency_overview",
            }:
                shutil.copy2(source, figures / source.name)

    guide = output / "START_HERE_CN.md"
    guide.write_text(report_text(summary, checklist), encoding="utf-8")
    (output / "REVIEWER_RESPONSE_PLAIN_EN.md").write_text(
        reviewer_reply_text(), encoding="utf-8"
    )

    artifacts = [path for path in output.rglob("*") if path.is_file()]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_bundle": str(bundle),
        "source_bundle_manifest_sha256": sha256(bundle / "bundle_manifest.json"),
        "notes": [
            "The frozen source reviewer bundle was not modified.",
            "SUPPORTED/PARTIAL/NOT SHOWN are narrative evidence labels, not statistical scores.",
            "Top-20% sets reproduce the source tie-inclusive formal analysis exactly.",
        ],
        "artifacts": [record(path, output) for path in sorted(artifacts)],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "n_artifacts": len(artifacts) + 1}, indent=2))


if __name__ == "__main__":
    main()
