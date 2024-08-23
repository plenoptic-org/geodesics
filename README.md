# geodesics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/plenoptic-org/geodesics/blob/main/LICENSE)
![Python version](https://img.shields.io/badge/python-3.10|3.11|3.12-blue.svg)
[![Build Status](https://github.com/plenoptic-org/geodesics/workflows/build/badge.svg)](https://github.com/plenoptic-org/geodesics/actions?query=workflow%3Abuild)
[![Project Status: Suspended – Initial development has started, but there has not yet been a stable, usable release; work has been stopped for the time being but the author(s) intend on resuming work.](https://www.repostatus.org/badges/latest/suspended.svg)](https://www.repostatus.org/#suspended)

Compute geodesic between two images, according to a model

The code in this repository used to be part of [plenoptic](https://github.com/plenoptic-org/plenoptic/), but was deprecated in version 1.0.3, as the geodesic code is less robust than the other included synthesis methods. More research needs to be done on the algorithms involved, as well as the specific implementations. If the implementation is improved to the point where we feel it is robust and stable enough, we will add it back to plenoptic.

This repository contains the geodesic code, along with the jupyter notebooks used to demonstrate their usage, and the tests. You are free to use this code for your own purposes and, if you do so, see the [citation](#citation) section for how to credit the code.

## Installation

This package is not on PyPI, but can be installed using `pip`, either directly from GitHub:

```bash
pip install git+https://github.com/plenoptic-org/geodesics.git
```
 
or from a local copy:

```bash
git clone https://github.com/plenoptic-org/geodesics.git
cd geodesics
pip install -e .
```

If you have issues with installation, see the [plenoptic documentation](https://plenoptic.readthedocs.io/en/latest/install.html)

## Citation

If you use the code in this repo in a published academic article or presentation, please cite the geodesic paper along with plenoptic's JoV paper, as below, and include language to the effect of "we made use of the geodesic code that is part of the plenoptic project".

```bibtex
  @article{duong2023plenoptic,
    title={Plenoptic: A platform for synthesizing model-optimized visual stimuli},
    author={Duong, Lyndon and Bonnen, Kathryn and Broderick, William and Fiquet, Pierre-{\'E}tienne and Parthasarathy, Nikhil and Yerxa, Thomas and Zhao, Xinyuan and Simoncelli, Eero},
    journal={Journal of Vision},
    volume={23},
    number={9},
    pages={5822--5822},
    year={2023},
    publisher={The Association for Research in Vision and Ophthalmology}
  }
```

```bibtex
@INPROCEEDINGS{Henaff16,
     TITLE= "Geodesics of learned representations",
     AUTHOR= "O J H\'{e}naff and E P Simoncelli",
     BOOKTITLE= "Int'l Conf on Learning Representations (ICLR)",
     ADDRESS= "San Juan, Puerto Rico",
     MONTH= "May",
     YEAR= 2016,
     PDF-URL= "http://www.cns.nyu.edu/pub/lcv/henaff16b-reprint.pdf",
     URL= "http://arxiv.org/abs/1511.06394",
     NOTE= "Available at http://arxiv.org/abs/1511.06394"
}
```
