# Figure 4e arrow annotations

The author adjusted the blue and pink arrows in Illustrator on 13 September
2026. `Figure4e.pdf` is the supplied panel. `Figure4.pdf` inserts that panel into
the current figure, retaining panels a-d, including the 27.8% label without a
white background.

The editable arrow paths are recorded in `brain_arrow_paths.json`, with panel
coordinates in PDF points. The plotting function
`reproduction.mosta.brain_arrows.draw_author_arrows` draws those paths over the
four newly calculated velocity fields. No model settings or velocity estimates
were changed. The Figure 4 tutorial still calculates the fields from the model
before plotting them. It does not load this archived figure as its calculation.

The numerical input, field plotting code and layout remain in
`paper_resources/mosta/reproduction/main_fig4_panels/fig4e`.
`reproduction.mosta.main_figure.draw_brain_velocity` calls the updated assembly.
