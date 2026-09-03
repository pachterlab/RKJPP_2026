# attainable_information_figures

Analysis code, figures, and manuscript source for "Image-Identifiable Genomic Subspaces: A Linear-Gaussian Model of Radiogenomic Recoverability". The estimator itself lives in the standalone [attainable-information](https://github.com/pachterlab/attainable_information) package; this repository holds the cohort pipeline, the scripts that produce every figure and number in `main.tex`, the generated figures under `notebooks/figures/`, and the Lean proofs in `rgit_lean/`.

A statistical analysis framework for characterizing fundamental information-theoretic limits on the relationship between radiology imaging phenotypes and genomic data.

## Overview

This project investigates the theoretical upper and lower bounds on how much predictive information flows between:

- **Imaging phenotypes** — features derived from CT and MRI studies (radiomic descriptors, deep features, or structured reads)
- **Genomic alterations** — somatic mutations, copy number variants (CNV), and germline SNPs

Rather than optimizing a specific predictor, the goal is to quantify the *fundamental* limits imposed by information theory: how much mutual information exists between these modalities, what rate-distortion trade-offs apply when compressing one modality to predict the other, and where the predictive ceiling lies regardless of model choice.

## Research Questions

1. What is the mutual information between imaging-derived features and genomic alteration profiles?
2. What are the theoretical upper bounds on genomics → imaging (and imaging → genomics) prediction accuracy?
3. How do data quantity, feature dimensionality, and noise affect the achievable information transfer?

## Repository Structure

```
.
├── data/           # Raw and processed datasets (not committed)
├── notebooks/      # Exploratory analysis and figure generation
├── rgit/           # Core Python package
├── scripts/        # Standalone analysis scripts
└── pyproject.toml  # Python project configuration
```

## Methods

The analysis draws on:

- **Mutual information estimation** — non-parametric (KSG, MINE) and parametric estimators for continuous and mixed-type variables
- **Rate-distortion theory** — characterizing the minimum description length of one modality needed to predict the other at a given fidelity
- **Data processing inequality** — bounding information loss through feature extraction pipelines
- **Finite-sample corrections** — bias correction and bootstrap confidence intervals for MI estimates in high-dimensional settings

## The attainable-information ceiling

The estimator and bounds are packaged separately as [attainable-information](https://github.com/pachterlab/attainable_information); `rgit.model` and `rgit.bounds` re-export it.

The headline deliverable is an **upper bound on how much genomic information any
model trained on `n` patients can extract from imaging** — not the
infinite-data channel value. For a genomic direction with channel
recoverability `rho^2` and a `d*`-dimensional imaging representation,

```
R_n = rho^2 * n / (n + nu),        nu = d* * (1 - rho^2) / rho^2
I_n = -0.5 * sum_i log2(1 - R_n,i)     # bits per patient
```

`nu` is the *learning cost*: the cohort size at which half the channel is
attainable, equal to the feature count divided by the channel SNR, and equal to
the Bayes-optimal ridge penalty. Because `R_n` decays like `rho^4` for weak
channels, marginal genomic directions are doubly penalized — the formal reason
not to widen a gene panel at fixed `n`.

```python
import rgit
rgit.attainable_recoverability(0.1, n=190, d_star=20)   # 0.051
rgit.attainable_information([0.1, 0.05], n=190, d_star=20)  # bits/patient
rgit.learning_cost(0.1, 20)              # 180 patients
rgit.sample_size_for_fraction(0.1, 20, 0.9)   # 1620 patients for 90%
rgit.auc_ceiling(rgit.attainable_recoverability(0.1, 190, 20))  # AUC <= 0.565

# total retention, not just the best axis: at fixed total signal T the
# information is maximized by CONCENTRATION, so this is the knapsack optimum
rgit.max_total_information(T=0.37, n=190, d_star=20, rho2_max=0.15)  # 0.164 bits

# spectrum-free: a fourth-moment budget S = sum_i rho_i^4 converts exactly,
# valid for EVERY spectrum shape (here S is bounded from data by 3a + 3b)
rgit.max_information_given_fourth_moment(0.20, n=190, d_star=20)   # 0.761 bits
```

Reproducing the analysis (each writes figures + JSON under `notebooks/figures/`):

```bash
# 1. simulation: six model families approach the ceiling, none crosses it
python scripts/attainable_bound_simulation.py

# 2. per-cohort learning curves, fitted channel value and learning cost
python scripts/attainable_bound_cohorts.py kirc nsclc adni

# 3a. model-free 95% upper limit on the LEADING axis (rank-one planting)
#     (also validates the closed form against real covariance structure)
python scripts/channel_ucl.py kirc nsclc adni

# 3b. limit on TOTAL retention across all directions (rank-swept planting).
#     The leading-axis figure from 3a bounds only the best canonical direction
#     and is a LOWER bound on the total.
python scripts/total_information_ucl.py kirc nsclc adni

# 3c. spectrum-free bound: sweeps decay shapes, combines the 3a cap with the 3b
#     statistic to bound the fourth moment S, then converts exactly.
#     THIS is the number to quote -- valid for every spectrum shape.
python scripts/shape_sensitivity.py kirc nsclc adni

# 4. chain-rule decomposition against hand-specified anchor gene panels
python scripts/anchor_gene_saturation.py nsclc kirc adni

# 4b. STRESS TEST (the headline figure): sweeps data size x genomics set x
#     algorithm and checks every combination against the spectrum-free ceiling.
#     Add --replot to redraw from saved JSON without recomputing.
python scripts/bound_stress_test.py kirc nsclc adni

# 5. cross-cohort table + summary figure (headline numbers)
python scripts/attainable_summary.py

# TCGA-KIRC target classes and CT representations (Section "On TCGA-KIRC, CT decodes stage
# and sex but not driver mutations", the sex-confound paragraph, Appendix figure and table).
# Needs the four KIRC imaging matrices (tumor/organ radiomics, tumor/whole-volume RadImageNet),
# gene_expression.h5ad, mutated_genes.h5ad, clinical_tcga.tsv and the TCIA series digest.
python scripts/build_kirc_representation_notebook.py
jupyter nbconvert --execute --inplace notebooks/kirc_representation_bound.ipynb   # -> notebooks/figures/kirc_representations/summary.json
python scripts/kirc_channel_ucl_by_dimension.py                                   # -> notebooks/figures/kirc_representations/invert2.json
python scripts/kirc_per_gene_budget.py                                            # -> notebooks/figures/kirc_representations/budget.json
python scripts/kirc_target_classes.py                                             # -> notebooks/figures/kirc_target_classes.json
python scripts/kirc_target_classes_figure.py                                      # -> kirc_target_classes.pdf, kirc_representations.pdf, kirc_target_classes_table.tex
```

Run 2–4 before 5; step 5 reads their JSON outputs (3b requires 3a; 3c requires both).

## Setup

```bash
conda create -n rgit -y python=3.10 && conda activate rgit
pip install -e .[processing,notebooks,dev]
```

Python 3.10+ is recommended. Dependencies are declared in `pyproject.toml`.

## Checking math
``bash
cd rgit_lean && ./check.sh
```

## Usage

Analysis notebooks are in `notebooks/`. Reusable estimation utilities live in the `rgit/` package. Batch scripts for large-scale runs are in `scripts/`.

### Reproducing the analysis without the notebook

The full recoverability pipeline that `notebooks/radiogenomic_recoverability.ipynb`
orchestrates is also packaged as importable code, so it can be reproduced after a
plain `pip install rgit` — no notebook, and (for the synthetic case) no data on disk.

With no `.h5ad` files it runs the synthetic probabilistic-CCA dataset with known
ground truth, writing `stats.json` and figures to the output directory:

```bash
# console script (installed with the package)
rgit-recoverability --output-dir out/synthetic

# real data
rgit-recoverability \
    --genomics data/tcga_kirc/genomics/mutated_genes.h5ad \
    --imaging  data/tcga_kirc/imaging/tumor_radimagenet.h5ad \
    --genomics-data-type variant \
    --output-dir out/kirc
```

Or from Python:

```python
import rgit

# synthetic ground-truth run (nothing needed on disk)
report = rgit.run_recoverability_analysis(
    rgit.RecoverabilityConfig(output_dir="out/synthetic")
)
print(report.stats["effective_identifiable_rank"]["rank"])

# real cohort
report = rgit.run_recoverability_analysis(
    genomics_h5ad="data/.../mutated_genes.h5ad",
    imaging_h5ad="data/.../tumor_radimagenet.h5ad",
    genomics_data_type="variant",
    output_dir="out/kirc",
)
```

Every knob the notebook exposes lives on `rgit.RecoverabilityConfig`; the resolved
config is written next to `stats.json` for provenance. Only the core dependencies
are required — `scanpy` (HVG selection) falls back to variance ranking, and the
RadImageNet feature extractor (`torch`) is loaded lazily, so neither is needed to
run the recoverability analysis itself.

### Synthetic data
papermill notebooks/radiogenomic_recoverability.ipynb notebooks/out/radiogenomic_recoverability_output_synthetic.ipynb  # synthetic data

### Real data (TCGA-KIRC example)
#### Imaging data processing
python scripts/process_imaging_tcga_kirc.py -d data/tcga_kirc/imaging
<!-- python scripts/make_imaging_matrix.py -o data/tcga_kirc/imaging/whole_radiomics.h5ad -m data/tcga_kirc/imaging/metadata.csv --embedder pyradiomics -->
python scripts/make_imaging_matrix.py -o data/tcga_kirc/imaging/organ_radiomics.h5ad -m data/tcga_kirc/imaging/metadata.csv --mask_col organ_mask --embedder pyradiomics
python scripts/make_imaging_matrix.py -o data/tcga_kirc/imaging/tumor_radiomics.h5ad -m data/tcga_kirc/imaging/metadata.csv --mask_col tumor_mask --embedder pyradiomics --label 2
python scripts/make_imaging_matrix.py -o data/tcga_kirc/imaging/whole_radimagenet.h5ad -m data/tcga_kirc/imaging/metadata.csv --embedder radimagenet --model_path data/models/RadImageNet_pytorch/ResNet50.pt --clip_min -200 --clip_max 300 --resample_spacing 0.8,0.8,3.0 --apply_mask --crop_size 625,625,200
python scripts/make_imaging_matrix.py -o data/tcga_kirc/imaging/organ_radimagenet.h5ad -m data/tcga_kirc/imaging/metadata.csv --mask_col organ_mask --embedder radimagenet --model_path data/models/RadImageNet_pytorch/ResNet50.pt --clip_min -200 --clip_max 300 --resample_spacing 0.8,0.8,3.0 --apply_mask --crop_size 185,185,75
python scripts/make_imaging_matrix.py -o data/tcga_kirc/imaging/tumor_radimagenet.h5ad -m data/tcga_kirc/imaging/metadata.csv --mask_col tumor_mask --embedder radimagenet --model_path data/models/RadImageNet_pytorch/ResNet50.pt --clip_min -200 --clip_max 300 --resample_spacing 0.8,0.8,3.0 --apply_mask --crop_size 185,185,75 --label 2

#### Genomics data processing
wget -O data/tcga_kirc/genomics/mc3.v0.2.8.PUBLIC.maf.gz https://api.gdc.cancer.gov/data/1c8cfe5f-e52d-41ba-94da-f15ea1337efc
python scripts/make_genomics_matrix.py -o data/tcga_kirc/genomics/mutated_genes.h5ad --dataset tcga --feature gene_symbol --patient_ids data/tcga_kirc/imaging/metadata.csv data/tcga_kirc/genomics/mc3.v0.2.8.PUBLIC.maf.gz
python scripts/make_genomics_matrix.py -o data/tcga_kirc/genomics/mutated_pathways.h5ad --dataset tcga --feature pathway --patient_ids data/tcga_kirc/imaging/metadata.csv data/tcga_kirc/genomics/mc3.v0.2.8.PUBLIC.maf.gz

gdc-client download -m data/tcga_kirc/genomics/gene_expression_manifest.txt -d data/tcga_kirc/genomics
tar -xzvf data/tcga_kirc/genomics/gene_expression.tar.gz -C data/tcga_kirc/genomics/gene_expression
python scripts/make_genomics_matrix.py -o data/tcga_kirc/genomics/gene_expression.h5ad --dataset tcga --feature gene_expression --patient_ids data/tcga_kirc/imaging/metadata.csv --filename_to_patientid data/tcga_kirc/genomics/gene_expression_filename_to_patientid.csv data/tcga_kirc/genomics/gene_expression

#### Run notebooks
for genomics_h5ad in data/tcga_kirc/genomics/*.h5ad; do
    for imaging_h5ad in data/tcga_kirc/imaging/*_radimagenet.h5ad; do
        echo "Running recoverability analysis for genomics: $genomics_h5ad and imaging: $imaging_h5ad"
        output_notebook="notebooks/out/radiogenomic_recoverability_output_tcga_kirc_genomics_$(basename ${genomics_h5ad%.*})_imaging_$(basename ${imaging_h5ad%.*}).ipynb"
        papermill notebooks/radiogenomic_recoverability.ipynb "$output_notebook" -p GENOMICS_H5AD "$genomics_h5ad" -p IMAGING_H5AD "$imaging_h5ad"
    done
done

### NSCLC
wget -O data/nsclc/imaging/manifest.tcia https://www.cancerimagingarchive.net/wp-content/uploads/NSCLC_Radiogenomics-6-1-21-Version-4.tcia
wget -O data/nsclc/imaging/metadata.xlsx https://www.cancerimagingarchive.net/wp-content/uploads/NSCLC_Radiogenomics-6-1-21-Version-4-nbia-digest.xlsx
wget -O data/nsclc/genomics/gene_expression.txt.gz https://ftp.ncbi.nlm.nih.gov/geo/series/GSE103nnn/GSE103584/suppl/GSE103584%5FR01%5FNSCLC%5FRNAseq%2Etxt%2Egz
nbia-data-retriever --cli data/nsclc/imaging/manifest.tcia -d data/nsclc/imaging/dicom -v -f
python scripts/make_imaging_matrix.py -o data/nsclc/imaging/organ_radiomics.h5ad -m data/nsclc/imaging/metadata.csv --mask_col organ_mask --embedder pyradiomics --label 1
python scripts/make_genomics_matrix.py -o data/nsclc/genomics/gene_expression.h5ad --dataset nsclc --feature gene_expression data/nsclc/genomics/gene_expression.txt.gz

### ADNI
python scripts/process_imaging_adni.py
python scripts/make_genomics_matrix.py -o data/adni/genomics/gene_expression.h5ad --dataset adni --feature gene_expression data/adni/genomics/ADNI_Gene_Expression_Profile.csv


## Manuscript

```bash
latexmk -pdf main.tex
```
