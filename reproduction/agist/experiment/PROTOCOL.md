# AGIST experiment

The generator is a MOSTA model fitted to the four developmental stages.
Generation starts from 5,000 Brain cells sampled from E12.5. Each state has
two spatial coordinates and 50 gene-expression principal components.

The generator simulates times 0, 1, 2 and 3 using diffusion 0.03 and an
integration step of 0.01. Growth determines particle division and loss.
Interaction is evaluated in groups of 1,024 cells. All descendants of the
initial cells are included in the observations.

CytoBridge, stVCR and STORIES are fitted to these four observations.
CytoBridge and stVCR each produce ten stochastic predictions from the same
initial population. STORIES produces a deterministic gene-expression map.
Velocity, growth and attention are compared with the generator. Distribution
agreement is measured by Wasserstein distance.

The scripts and configuration files specify the remaining training settings.
