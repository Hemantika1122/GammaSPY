# GammaSPY: Gamma Spectroscopy analysis in PYthon

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A suite of open-source tools for automated γ-ray spectroscopy analysis of fast neutron-induced nuclear reactions, developed for data collected with the EXOGAM clover detector array, but can be use for any HPGe detector arrays.

---

## Overview

This repository contains the analysis framework developed for processing and interpreting γ–γ coincidence data from the EXOGAM detector array at GANIL. The tools integrate experimental data processing, nuclear database cross-referencing, and reaction simulation predictions into a unified automated workflow.

The suite includes:

- **2D background subtraction** — Native ROOT implementation of the Palameta–Waddington background subtraction algorithm for γ–γ matrices, eliminating the need to convert to legacy RadWare formats
- **Coincident γ finder** — Automated coincidence identification tool with recursive cascade mapping against ENSDF and XUNDL databases, to arbitrary cascade order
- **Expected spectrum plotter** — Quantitative prediction of gated γ-ray spectra based on TALYS cross-sections, detector efficiency, and experimental geometry
- **DCO and polarization asymmetry extraction** — Angular correlation analysis tools for spin-parity assignments from oriented nuclear ensembles
- **Custom TALYS modifications** — Source code patches extending discrete level tracking, memory allocation, and output formatting in TALYS for level-by-level population cross-sections

---

## Requirements

### Python
- Python ≥ 3.8
- ROOT with PyROOT bindings (≥ 6.24 recommended)
- NumPy, Matplotlib

### C++ / ROOT
- ROOT ≥ 6.24 (for TSpectrum)
- GCC or Clang with C++17 support

### TALYS (optional, for expected spectrum plotter)
- TALYS ≥ 1.96 with the patches in `talys_patches/` applied
- gfortran with 64-bit memory allocation support

### Databases
The tools query ENSDF and XUNDL locally. Database files should be placed in `data/ensdf/` and `data/xundl/` respectively. See `data/README.md` for formatting instructions.

---

## Installation

```bash
git clone https://gitlab.com/[your-username]/exogam-analysis.git
cd exogam-analysis
pip install -r requirements.txt
```

---

## Quick Start

### Run background subtraction
```python
from bkg_subtraction import RadwareBkg2D
bkg = RadwareBkg2D("output_gg_addback.root", "hgg_sym")
bkg.compute()
bkg.save("output_bkgsub.root")
```

### Launch the coincidence finder
```python
from coincidence_finder import CoincidenceFinder
cf = CoincidenceFinder(
    matrix_file="output_bkgsub.root",
    isotopes=["57Ni", "58Ni", "56Co", "57Co"],
    databases=["ENSDF", "XUNDL"]
)
cf.gate(energy=768.5, width=3.0, order=3)
cf.plot(save="768_gate_projection.svg")
```

---

## Repository Structure

```
exogam-analysis/
├── proof/               # PROOF-based parallel processing code (C++)
├── bkg_subtraction/     # 2D Radware-style background subtraction (Python/ROOT)
├── coincidence_finder/  # Automated coincidence identification (Python)
├── expected_spectrum/   # Expected yield calculator and spectrum plotter (Python)
├── angular_correlation/ # DCO ratio and polarization asymmetry tools (Python)
├── talys_patches/       # TALYS source code modifications and apply script
├── data/                # Database files (ENSDF, XUNDL) — not included, see data/README.md
├── examples/            # Example scripts and notebooks
└── tests/               # Unit tests
```

---

## Citation

If you use this software, please cite the associated thesis and this repository:

```
[Author], "[Thesis title]", PhD Thesis, [University], [Year].
Repository: https://gitlab.com/[your-username]/exogam-analysis
```

BibTeX entry:
```bibtex
@phdthesis{[citekey],
  author  = {[Author]},
  title   = {[Thesis title]},
  school  = {[University]},
  year    = {[Year]},
  note    = {Software: \url{https://gitlab.com/[your-username]/exogam-analysis}}
}
```

---

## License

Copyright © [Year] [Author]

Licensed under the **Apache License, Version 2.0**. You may use, distribute, and modify this software freely provided that you retain the original copyright notice and attribution. See [LICENSE](LICENSE) for the full terms.

---

## Acknowledgements

- GANIL facility and the EXOGAM collaboration
- ENSDF and XUNDL databases (National Nuclear Data Center, BNL)
- TALYS nuclear reaction code (NRG Petten)
- ROOT data analysis framework (CERN)
