# Quantifying CO2 Leakage Risk — Importance Sampling Methodology

This repository reproduces the workflow described in the paper  
**_"Quantifying CO₂ Leakage Risk when Planning Safe Geological Carbon Storage using Importance Sampling of Failure Probabilities."_**

It provides a reproducible Jupyter-based implementation of the statistical and computational framework developed to evaluate rare CO₂ leakage events in geological carbon storage (GCS).  
The workflow integrates Naïve Monte Carlo (NMC) simulations, dimensionality reduction (PCA/DGSA), and Importance Sampling (IS) to achieve accurate leakage-risk estimates with dramatically fewer forward simulations.

---

## Purpose

This project is designed to make the paper’s methodology **transparent and replicable**.  
It implements all steps:

1. Naïve Monte Carlo baseline — generation of reservoir realizations ($m_1$, $m_2$) for top-surface geometry and porosity.  
2. Dimensionality reduction and sensitivity screening (PCA + DGSA).  
3. Construction of IS alternative distribution $g(\mathbf{m})$ via Multivariate Kernel Density Estimation (MKDE).  
4. Reconstruction of IS realizations $m′^{(l)}$ for forward MRST flow simulations.  
5. Post-simulation weighting and analysis — computation of importance weights, effective sample size (ESS), and Chebyshev confidence bounds.  

The final output quantifies how many fewer simulations are required by IS to reach the same confidence level as the full Monte Carlo ensemble.

---

## Repository Structure


