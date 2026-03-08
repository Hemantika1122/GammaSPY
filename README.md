# GammaSPY: Gamma Spectroscopy analysis in PYthon

> **Warning** This package is in active development and may introduce breaking
> changes.

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

The suite currently includes:

- **2D background subtraction** — Native ROOT implementation of the
  Palameta–Waddington background subtraction algorithm for γ–γ matrices,
  eliminating the need to convert to legacy RadWare formats
- **Coincident γ finder** — Automated coincidence identification tool with
  recursive cascade mapping against ENSDF and XUNDL databases, to arbitrary
  cascade order

Planned features include:

- **Expected spectrum plotter** — Quantitative prediction of gated γ-ray spectra
  based on TALYS cross-sections, detector efficiency, and experimental geometry
- **DCO and polarization asymmetry extraction** — Angular correlation analysis
  tools for spin-parity assignments from oriented nuclear ensembles
- **Custom TALYS modifications** — Source code patches extending discrete level
  tracking, memory allocation, and output formatting in TALYS for level-by-level
  population cross-sections

---

## Installation

GammaSPY requires ROOT with PyROOT bindings. If you already have a setup with
that, you can install simply with:

```bash
pip install gammaspy
```

Otherwise using `pixi`
([installation instructions](https://pixi.prefix.dev/latest/installation/)) is
highly recommended since it can also install `root`. Then simply:

```bash
git clone https://gitlab.com/Hemantika1122/GammaSPY.git
pixi install
```

---

## Quick Start

### Get projection with background subtraction

If running in jupyter notebook use `%jsroot` for interactable TCanvases

```python
from gammaspy.hist2d import Hist2D

hgg = Hist2D("output_gg_addback.root", "hgg")
canvas = hgg.draw_projection(gate_energy=1454, gate_width=3, subtract_background=True)
```

### Add coincidence finder

First download the adaptedLevels csv files for relevant isotopes from the nudat
website.

```python
from gammaspy.nudat import LevelSchemes

level_schemes = LevelSchemes(
    isotopes=["57Ni", "58Ni", "56Co", "57Co"],
)
canvas = hgg.draw_projection(
    gate_energy=1454, gate_width=3, level_schemes=level_schemes, coincidence_order=2
)
```
