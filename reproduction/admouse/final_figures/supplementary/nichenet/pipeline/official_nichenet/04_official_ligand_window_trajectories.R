#!/usr/bin/env Rscript
# Non-heatmap view of official NicheNet ligand scoring over dense adjacent windows.
# Inputs are read-only CSV outputs from the completed official NicheNet scoring run.

suppressPackageStartupMessages(library(ggplot2))

script_file <- sub('^--file=', '', grep('^--file=', commandArgs(trailingOnly=FALSE), value=TRUE)[1])
script_dir <- dirname(normalizePath(script_file)); final_root <- normalizePath(file.path(script_dir, '..', '..', '..', '..')); data_root <- dirname(final_root)
root <- file.path(data_root, 'nichenet', 'new_interpolation', 'latest_nichenet')
activity_path <- file.path(root, "results", "official_ligand_activity_all_50_windows.csv")
figure_dir <- file.path(root, "figures")
data_dir <- file.path(root, "data")

activity <- read.csv(activity_path, check.names = FALSE, stringsAsFactors = FALSE)
activity <- activity[activity$response_model_time <= 2.0 + 1e-10, ]

# Formal mapping used for the 51-state interpolation grid.  It is defined only
# on the observed interval: model 0, 1, 2 = 2.5, 5.7, 17.9 months.
map_months <- function(t) ifelse(t <= 1,
  2.5 + (5.7 - 2.5) * t,
  5.7 + (17.9 - 5.7) * (t - 1))
activity$midpoint_model_time <- (activity$baseline_model_time + activity$response_model_time) / 2
activity$midpoint_months <- map_months(activity$midpoint_model_time)

means <- aggregate(aupr_corrected ~ test_ligand, activity, mean)
means <- means[order(-means$aupr_corrected, means$test_ligand), ]
selected <- head(means$test_ligand, 8)
plot_data <- activity[activity$test_ligand %in% selected, ]
plot_data$test_ligand <- factor(plot_data$test_ligand, levels = selected)
palette <- c("#B2182B", "#2166AC", "#1B7837", "#762A83", "#E08214", "#4D4D4D", "#00A6A6", "#CC6677")
names(palette) <- selected

anchors <- data.frame(age = c(2.5, 5.7, 17.9), label = c("observed: 2.5", "observed: 5.7", "observed: 17.9"))
base_theme <- theme_classic(base_size = 12) + theme(
  legend.position = "bottom", legend.title = element_blank(),
  plot.title = element_text(face = "bold"), plot.subtitle = element_text(size = 10),
  axis.title = element_text(face = "bold"))

rank_plot <- ggplot(plot_data, aes(midpoint_months, activity_rank, colour = test_ligand)) +
  geom_vline(data = anchors, aes(xintercept = age), inherit.aes = FALSE, colour = "grey55", linetype = "dashed", linewidth = 0.4) +
  geom_line(linewidth = 0.7) + geom_point(size = 1.3) +
  scale_colour_manual(values = palette) + scale_y_reverse(breaks = c(1, 5, 10, 15, 20, 23), limits = c(23.5, 0.5)) +
  labs(title = "Official NicheNet ligand rank across dense transition windows",
       subtitle = "Top 8 ligands by mean corrected AUPR; rank 1 = strongest within each window",
       x = "Transition-window midpoint (months)", y = "Ligand activity rank") + base_theme

activity_plot <- ggplot(plot_data, aes(midpoint_months, aupr_corrected, colour = test_ligand)) +
  geom_vline(data = anchors, aes(xintercept = age), inherit.aes = FALSE, colour = "grey55", linetype = "dashed", linewidth = 0.4) +
  geom_hline(yintercept = 0, colour = "grey70", linewidth = 0.35) +
  geom_line(linewidth = 0.7) + geom_point(size = 1.3) +
  scale_colour_manual(values = palette) +
  labs(title = "Official NicheNet ligand activity across dense transition windows",
       subtitle = "Corrected AUPR from predict_ligand_activities(); higher = stronger target-set prioritization",
       x = "Transition-window midpoint (months)", y = "Corrected AUPR") + base_theme

top3 <- activity[activity$activity_rank <= 3, ]
top3$rank_label <- factor(paste0("rank ", top3$activity_rank), levels = c("rank 3", "rank 2", "rank 1"))
top3$test_ligand <- factor(top3$test_ligand, levels = unique(c(selected, sort(setdiff(unique(top3$test_ligand), selected)))))
all_ligands <- levels(top3$test_ligand)
extra <- setdiff(all_ligands, names(palette))
extra_cols <- setNames(rep("#999999", length(extra)), extra)
turnover_plot <- ggplot(top3, aes(midpoint_months, rank_label, colour = test_ligand)) +
  geom_vline(data = anchors, aes(xintercept = age), inherit.aes = FALSE, colour = "grey55", linetype = "dashed", linewidth = 0.4) +
  geom_line(aes(group = interaction(test_ligand, activity_rank)), alpha = 0.35, linewidth = 0.45) +
  geom_point(size = 2.0) +
  scale_colour_manual(values = c(palette, extra_cols)) +
  labs(title = "Top-ligand succession across dense transition windows",
       subtitle = "Each point is one ligand ranked among the top three in its adjacent window",
       x = "Transition-window midpoint (months)", y = NULL) + base_theme

save_plot <- function(plot, stem, width, height) {
  ggsave(file.path(figure_dir, paste0(stem, ".png")), plot, width = width, height = height, dpi = 300)
  ggsave(file.path(figure_dir, paste0(stem, ".pdf")), plot, width = width, height = height)
}
save_plot(rank_plot, "08a_official_ligand_rank_40_observed_windows", 9, 5.3)
save_plot(activity_plot, "08b_official_ligand_activity_40_observed_windows", 9, 5.3)
save_plot(turnover_plot, "08c_official_top3_ligand_turnover_40_observed_windows", 9, 3.2)

write.csv(activity, file.path(data_dir, "08_official_ligand_activity_40_observed_windows_with_month_mapping.csv"), row.names = FALSE)
write.csv(plot_data, file.path(data_dir, "08_top8_official_ligand_rank_activity_plot_data.csv"), row.names = FALSE)
write.csv(top3, file.path(data_dir, "08_top3_official_ligand_turnover_plot_data.csv"), row.names = FALSE)
