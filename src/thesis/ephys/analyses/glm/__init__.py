"""V1 Poisson GLM: design construction, fitting, and figures.

`dmat` builds the stimulus windows and the design matrix, `video_svd` fits the
motion-energy PCs, `fit` runs selection/cross-validation/refit and writes that
fit's figures, `unique_dev` scores each block and writes its figure, and `cli`
holds the `glm` command.
Run the pipeline through the `glm` command (entry point
`thesis.ephys.analyses.glm.cli:main`).
"""
