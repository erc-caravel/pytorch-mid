# Results

This folder holds the values behind three figures of the paper. Each file is a CSV table with a header row. All values are taken from the figures of the paper. Their precision is that of the figure coordinates.

## `real_model_mid_vs_dice.csv`

Paper figure: real segmentation models, panel a. One row is one group of model, dataset and foreground class. The file holds 56 groups for each projection.

| Column | Content |
| --- | --- |
| `projection` | `global`: one PCA basis for all groups. `local`: one PCA basis per dataset and organ, with k ≤ 128. |
| `model` | `SR-50`, `SR-150`, `SR-300` (SegResNet after 50, 150 and 300 training epochs), `SwinUNETR`, `MedSAM`, `nnUNet`, `A2V`, `SPOCKMIP` |
| `group_type` | `CT` (MSD Task03 Liver, Task06 Lung, Task07 Pancreas, Task09 Spleen, Task10 Colon), `ACDC` (cardiac MRI), `IXI MRA` (cerebral arteries) |
| `mean_dice` | Mean Dice over the cases of the group |
| `mid` | MID of the group |

The paper figure shows the `global` rows. The `local` rows come from an earlier version of the figure.

## `ixi_mra_pca2.csv`

Paper figure: real segmentation models, panel b. The file holds the IXI MRA cases in the first two principal components of the ground-truth reference. The PCA is fitted on the ground-truth set. PC 1 holds 25.4% of the variance. PC 2 holds 19.9%.

| Column | Content |
| --- | --- |
| `set` | `ground_truth`, `nnUNet`, `A2V`, `SPOCKMIP` |
| `point_index` | Plotting order within the set, from 0 to 22. The index carries no subject identity. |
| `pc1`, `pc2` | Coordinates on PC 1 and PC 2 |

Each set holds 23 cases. The MID values of the three methods are in `real_model_mid_vs_dice.csv`, in the rows with `group_type` equal to `IXI MRA`.

## `anatomy_reference_index.csv`

Paper figure: per-anatomy reference index (appendix). The file gives the clean-versus-clean MID of each anatomy. For each anatomy, the clean ground-truth pairs are split at random into two halves. MID is computed between the two halves. Repeated splits give the mean and the standard deviation. Use these values as the reference range of each anatomy.

| Column | Content |
| --- | --- |
| `region` | Body region |
| `organ` | Anatomy |
| `modality` | `CT` or `MRI` (the MRI rows include the IXI MRA cerebral arteries) |
| `mid_mean` | Mean clean-versus-clean MID |
| `mid_std` | Standard deviation over repeated splits |
