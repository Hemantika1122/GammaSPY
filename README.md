# GammaSPY: Gamma Spectroscopy analysis in PYthon

> **Warning**
This package is in active development and may introduce breaking changes.

A suite of open-source tools for automated γ-ray spectroscopy analysis of fast
neutron-induced nuclear reactions, developed for data collected with the EXOGAM
clover detector array, but can be use for any HPGe detector arrays.

---

## Overview

This repository contains the analysis framework developed for processing and
interpreting γ–γ coincidence data from the EXOGAM detector array at GANIL. The
tools integrate experimental data processing, nuclear database
cross-referencing, and reaction simulation predictions into a unified automated
workflow.

The suite includes:

- **2D background subtraction** — Native ROOT implementation of the
  Palameta–Waddington background subtraction algorithm for γ–γ matrices,
  eliminating the need to convert to legacy RadWare formats
- **Coincident γ finder** — Automated coincidence identification tool with
  recursive cascade mapping against ENSDF and XUNDL databases, to arbitrary
  cascade order
- **Expected spectrum plotter** — Quantitative prediction of gated γ-ray spectra
  based on TALYS cross-sections, detector efficiency, and experimental geometry
- **DCO and polarization asymmetry extraction** — Angular correlation analysis
  tools for spin-parity assignments from oriented nuclear ensembles
- **Custom TALYS modifications** — Source code patches extending discrete level
  tracking, memory allocation, and output formatting in TALYS for level-by-level
  population cross-sections

---

## Requirements

### Python

- Python ≥ 3.12
- ROOT with PyROOT bindings (≥ 6.24 recommended)
- NumPy, Matplotlib

### C++ / ROOT

- ROOT ≥ 6.24 (for TSpectrum)
- GCC or Clang with C++17 support

### TALYS (optional, for expected spectrum plotter)

- TALYS ≥ 1.96 with the patches in `talys_patches/` applied
- gfortran with 64-bit memory allocation support

### Databases

The tools query ENSDF and XUNDL locally. Database files should be placed in
`data/ensdf/` and `data/xundl/` respectively. See `data/README.md` for
formatting instructions.

---

## Installation

Using `pixi` ([installation instructions](https://pixi.prefix.dev/latest/installation/)) is highly recommended since it can also install `root`.
Then simply:

```bash
git clone https://gitlab.com/Hemantika1122/GammaSPY.git
pixi install
```

---

## Quick Start

### Get projection with background subtraction

```python
from gammaspy.hist2d import Hist2D

bkg = Hist2D("output_gg_addback.root", "hgg_sym")
projection = bkg.get_projection(gate_energy=1454, gate_width=3)
```

### Add coincidence finder

```python
cf = CoincidenceFinder(
    matrix_file="output_bkgsub.root",
    isotopes=["57Ni", "58Ni", "56Co", "57Co"],
    databases=["ENSDF", "XUNDL"],
)
cf.gate(energy=768.5, width=3.0, order=3)
cf.plot(save="768_gate_projection.svg")
```
