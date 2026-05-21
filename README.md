# Tehran GSMP Statistical Analysis

Python code for the statistical analysis of district-level ecological indicators in the Tehran GSMP study.

## Overview

This repository contains the code used to analyze district-level values of:

- BAF (Biotope Area Factor)
- CCI (Canopy Cover Index)

The script computes:

- Gini coefficients for distributional inequality
- Wilcoxon signed-rank tests for paired comparisons
- Pearson correlations with a density proxy
- Lorenz curves for visual comparison of distributions

## Repository contents

- `ucc.py` — main analysis script
- `requirements.txt` — Python dependencies
- `output/` — optional folder for generated figures and results

## Input data

The script uses district-level summary values transcribed from the study tables/supplementary material.

Variables included in the script:

- BAF under LULC-current condition
- BAF under GSMP scenario
- CCI under LULC-current condition
- CCI under GSMP scenario
- Urban density proxy (neighborhoods per hectare)

## Requirements

Install dependencies with:

```bash
python -m pip install -r requirements.txt
