# geodesics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/plenoptic-org/geodesics/blob/main/LICENSE)
![Python version](https://img.shields.io/badge/python-3.10|3.11|3.12-blue.svg)
[![Build Status](https://github.com/plenoptic-org/geodesics/workflows/build/badge.svg)](https://github.com/plenoptic-org/geodesics/actions?query=workflow%3Abuild)
[![Project Status: Suspended – Initial development has started, but there has not yet been a stable, usable release; work has been stopped for the time being but the author(s) intend on resuming work.](https://www.repostatus.org/badges/latest/suspended.svg)](https://www.repostatus.org/#suspended)

Compute geodesic between two images, according to a model

The code in this repository used to be part of [plenoptic](https://github.com/plenoptic-org/plenoptic/), but was deprecated in version 1.0.3, as the geodesic code is less robust than the other included synthesis methods. More research needs to be done on the algorithms involved, as well as the specific implementations. If the implementation is improved to the point where we feel it is robust and stable enough, we will add it back to plenoptic.

This repository contains the geodesic code, along with the jupyter notebooks used to demonstrate their usage, and the tests. You are free to use this code for your own purposes and, if you do so, see the [citation](#citation) section for how to credit the code.

> [!CAUTION]
> This repo is far from stable and will need lots of experimenting before it's ready.

## Status

> [!NOTE]
> The GPU tests are not currently running for this repository. It *should* work on the GPU, but not guaranteed.

As of August 2024, the `Geodesic` object works under basic conditions: see the top of the notebook, where it succeeds for very simple measures of distance in a two-dimensional space. However, it's unclear whether this will work for more complicated models, as can be seen in the rest of that notebook, which just kind of fizzles out.

The first goal would be to reproduce the results of Henaff and Simoncelli, 2016 [^1], which requires implementing the gradient projection step. Currently, we simply minimize the energy (i.e., the squared L2 norm) of the path in representational space. This is different than the paper's algorithm, which seeks to minimize the energy in the image-domain, conditioned on staying in the set of representational geodesics. That is, they first minimize the representational energy, as we do, then:

 - Compute the gradient required to minimize the image-domain geodesic
 - Compute the gradient required to minimize the representational geodesic
 - Project the representational geodesic out of the image-domain geodesic
 - Take the resulting step
 - Repeat until convergence
 
This algorithm thus has an inner and outer loop, so that the representation-space optimization is run until convergence, then the image-space optimization, and back and forth until there's no movement from either.
 
Using this algorithm, the authors are able to show that the VGG 16 network using L2 pooling has geodesics that more closely align with human perception than the same network with max pooling. Then the goal would be to take this object and use it on a variety of models and images and validate that it behaves reasonably well.

Should probably start simpler than that though, so: implement the above algorithm and use it with a relatively simple model, such as the `FactorizedPyramid`, which computes the amplitude and phase of the [steerable pyramid](https://plenoptic.readthedocs.io/en/latest/tutorials/models/03_Steerable_Pyramid.html) coefficients. Using two locally different images as the endpoints (e.g., smiling and frowning views of the same face), rather than globally different ones (e.g., different semantic categories, global translation or rotation), will probably be more meaningful here.

The long-term goal of this research project would be to construct a series of representational geodesics in a hierarchical model: as we do now, start with a pixelfade, i.e., a geodesic in the image domain. Then iteratively construct geodesic at each layer conditioned on the previous layer (with the first layer conditioned on the image-domain geodesic). This will allow us to visualize the gradual building up of invariances in the model. Would probably be most informative if we culminated in a metamer for the model's final layer.

There are two main concerns here:

- Any given layer of a model will have many possible geodesics, because of the amount of information discarded. The goal of conditioning the geodesic on another domain is to restrict the possible solutions, but this may not be sufficient.
- More importantly, we do not have a way of checking that a given solution is reasonable, i.e., that we're not stuck in a local optimum. With metamers, we know that there is at least one image that has 0 loss (the target image), and thus we can reason about how far we are from it. With MAD Competition, local optima are still informative, even if they're not *as* informative. But for representational geodesics, we really need the global optimum; local optima will not be informative. We are worried that we cannot come up with a robust enough algorithm here: one that works in many cases and that we can at least check whether the current solution is good and, if not, can take steps to improve it.

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

[^1]: Hénaff, O. J. and Simoncelli, E. P., “Geodesics of learned representations”, ICLR, 2016. doi:10.48550/arXiv.1511.06394.
