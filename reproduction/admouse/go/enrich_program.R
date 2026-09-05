
suppressPackageStartupMessages({
  library(clusterProfiler)
  library(enrichplot)
  library(ggplot2)
  library(org.Mm.eg.db)
})

rm(list = ls())
set.seed(1)

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg) == 1) {
  dirname(normalizePath(sub("^--file=", "", script_arg)))
} else {
  getwd()
}
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop('Usage: Rscript enrich_program.R gene_list.csv output_directory')
f <- normalizePath(args[1])
out_dir <- args[2]
plots_dir <- out_dir

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

pvalue_cutoff <- 0.05
qvalue_cutoff <- 0.2


# 绘图主题
theme_nature <- function(base_size = 8) {
  theme_bw(base_size = base_size, base_family = "Arial") +
    theme(
      panel.grid = element_blank(),
      panel.border = element_rect(
        color = "black",
        linewidth = 0.6
      ),
      axis.line = element_line(
        color = "black",
        linewidth = 0.6
      ),
      axis.ticks = element_line(
        color = "black",
        linewidth = 0.5
      ),
      axis.title = element_text(size = base_size + 1),
      axis.text = element_text(size = base_size),
      legend.title = element_text(size = base_size),
      legend.text = element_text(size = base_size - 1),
      plot.title = element_text(
        size = base_size + 2,
        face = "bold"
      ),
      strip.background = element_blank()
    )
}


palette_main <- c(
  "#3B6FB6",
  "#6AAED6",
  "#BFD3E6",
  "#FEE090",
  "#F46D43",
  "#D73027"
)


safe_save <- function(
    plot_fun,
    filename,
    width = 6,
    height = 5,
    apply_theme = TRUE
) {
  cairo_pdf(
    filename,
    width = width,
    height = height,
    family = "Arial"
  )

  p <- plot_fun()

  if (inherits(p, "ggplot") && apply_theme) {
    p <- p + theme_nature()
  }

  print(p)
  dev.off()
}


# 读取基因
filedata <- read.csv(
  f,
  stringsAsFactors = FALSE
)

genes_raw <- unique(na.omit(filedata$gene))


# 转换为ENTREZ ID
mapping <- bitr(
  genes_raw,
  fromType = "SYMBOL",
  toType = "ENTREZID",
  OrgDb = org.Mm.eg.db
)

entrez <- unique(mapping$ENTREZID)

if (length(entrez) == 0) {
  stop("No genes could be mapped to ENTREZID.")
}


# GO富集分析
ego <- enrichGO(
  gene = entrez,
  OrgDb = org.Mm.eg.db,
  keyType = "ENTREZID",
  ont = "ALL",
  pvalueCutoff = pvalue_cutoff,
  qvalueCutoff = qvalue_cutoff,
  readable = TRUE
)

res <- list(
  mapping = mapping,
  go = ego
)
write.csv(mapping, file.path(out_dir, 'gene_id_mapping.csv'), row.names=FALSE)
write.csv(as.data.frame(ego), file.path(out_dir, 'go_enrichment.csv'), row.names=FALSE)
saveRDS(ego, file.path(out_dir, 'go_enrichment.rds'))
writeLines(capture.output(sessionInfo()), file.path(out_dir, 'R_packages.txt'))


plot_set <- function(res, prefix, title_base) {

  if (is.null(res) || nrow(as.data.frame(res)) == 0) {
    return(invisible(NULL))
  }


  # Dotplot
  safe_save(
    function() {
      dotplot(
        res,
        showCategory = 15,
        font.size = 8,
        label_format = 1000
      ) +
        ggtitle(paste0(title_base, " Dotplot")) +
        scale_color_gradientn(colors = palette_main) +
        theme(
          axis.text.y = element_text(size = 8),
          plot.margin = margin(10, 20, 10, 10)
        )
    },
    paste0(prefix, "_dot.pdf"),
    width = 12,
    height = 7
  )


  # Barplot
  safe_save(
    function() {
      barplot(
        res,
        showCategory = 20,
        font.size = 8,
        label_format = 1000
      ) +
        ggtitle(paste0(title_base, " Barplot")) +
        scale_fill_gradientn(colors = palette_main) +
        theme(
          axis.text.y = element_text(size = 8),
          plot.margin = margin(10, 20, 10, 10)
        )
    },
    paste0(prefix, "_bar.pdf"),
    width = 13,
    height = 8
  )


  # Cnetplot
  safe_save(
    function() {
      cnetplot(
        res,
        showCategory = 10,
        layout = "circle",
        node_label = "all"
      ) +
        ggtitle(paste0(title_base, " Cnetplot"))
    },
    paste0(prefix, "_cnet.pdf"),
    width = 14,
    height = 14
  )


  # UpSet plot
  safe_save(
    function() {
      upsetplot(
        res,
        n = 10
      ) +
        ggtitle(paste0(title_base, " UpSet")) +
        theme_nature(base_size = 8) +
        ggupset::theme_combmatrix(
          combmatrix.label.height =
            grid::unit(3, "cm"),

          combmatrix.label.extra_spacing = 0,

          combmatrix.label.total_extra_spacing =
            grid::unit(1, "pt"),

          combmatrix.panel.margin =
            grid::unit(c(0.2, 0.2), "pt"),

          combmatrix.panel.point.size = 3.2,
          combmatrix.panel.line.size = 0.8,
          combmatrix.panel.line.color = "black",

          combmatrix.panel.point.color.fill = "black",
          combmatrix.panel.point.color.empty = "#D9D9D9",

          combmatrix.panel.striped_background = TRUE,

          combmatrix.panel.striped_background.color.one =
            "white",

          combmatrix.panel.striped_background.color.two =
            "#F4F4F4"
        ) +
        theme(
          plot.margin = margin(8, 8, 8, 8)
        )
    },
    paste0(prefix, "_upset.pdf"),
    width = 7.5,
    height = 7.5,
    apply_theme = FALSE
  )


  # Enrichment map
  if (nrow(as.data.frame(res)) >= 3) {

    safe_save(
      function() {

        res_ts <- pairwise_termsim(res)

        emapplot(
          res_ts,
          showCategory = 20,

          layout.params = list(layout = "fr"),
          edge.params = list(min = 0.25),

          size_edge = 0.4,
          node_label_size = 3.5,

          cluster.params = list(label_format = 1000)
        ) +
          ggtitle(
            paste0(
              title_base,
              " Enrichment Map"
            )
          ) +
          theme(
            plot.margin = margin(
              25, 25, 25, 25
            )
          )
      },
      paste0(prefix, "_emap.pdf"),

      # 增大画布
      width = 20,
      height = 16
    )
  }
}


prefix <- file.path(
  out_dir,
  tools::file_path_sans_ext(
    basename(f)
  )
)

if (
  !is.null(res$go) &&
  nrow(as.data.frame(res$go)) > 0
) {

  plot_set(
    res$go,
    file.path(
      plots_dir,
      paste0(
        basename(prefix),
        "_go"
      )
    ),
    paste0(
      basename(f),
      " - GO (ALL)"
    )
  )

} else {

  message(
    "No GO enrichment found for ",
    f
  )
}
# End of GO enrichment script.
