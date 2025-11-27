# ADHD_connectome_biomarkers_transfer_research
This research contains some stages in order to investigate whether specific connectome biomarkers of ADHD is transferred between adolescent and adult age cohorts.

# Model Card — ADHD Connectome (rs-fMRI)

**Intended use.** Research on ADHD vs control classification using rs-fMRI time series and functional connectivity.

## Training data
- ADHD200 (cohorts 12–15 and 16–21), QC for motion/metadata.
- (Optional) UCLA CNP LA5c for external checks.

## Metrics
- Primary: ROC-AUC, Accuracy (subject-level).
- Secondary: Precision/Recall/F1, PR-AUC, site-wise stratification.

## Ethical considerations
- Research-only; not a clinical device.
- Potential site/protocol biases; use within evaluated domain.

## Limitations
- Small N; multi-site heterogeneity; motion confounds.
- Results depend on parcellation, TR, preprocessing (Athena/fMRIPrep).

# Contributing

1. Open an issue to discuss changes.
2. Create a feature branch from `main`.
3. Follow PEP8 for Python; add tests.
4. Your contributions are under Apache-2.0.

# Data Usage and Redistribution

This repository **does not redistribute** original imaging data.

Data sources used in this project include:
- ADHD200 / INDI (Athena derivatives and/or BIDS-converted datasets).
- UCLA CNP LA5c (subset of the 1000 Functional Connectomes Project).

Obtain data directly from providers under their terms. This repo provides only **manifests, scripts, and instructions**.

**Important**
- Respect each dataset's license/DUA and citation requirements.
- Do not upload PHI/PII or any de-anonymized data.
- Prefer sharing derived, aggregate metrics that comply with the source terms.

# Licensing

- **Code:** Apache-2.0 (`LICENSE`).
- **Documentation:** CC BY 4.0 (`LICENSE-CC-BY-4.0`).
- **Models:** CC BY 4.0
- **Data:** Not redistributed. See `Data Usage and Redistribution` section.
