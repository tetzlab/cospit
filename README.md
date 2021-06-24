# Introduction

[![cospit](https://github.com/tetzlab/cospit/actions/workflows/basic.yml/badge.svg)](https://github.com/tetzlab/cospit/actions/workflows/basic.yml)

`cospit` implements the mixture method of "Generation of Correlated
Spike Trains" (Brette 2009, https://doi.org/10.1162/neco.2009.12-07-657)
to generate correlated spike trains.

For an alternative implementations see https://github.com/gdetor/CorrSpikeTrains.
For the original C++ implementation see http://romainbrette.fr/publications.

# Example

```console
scripts/example.py
Target Pearson correlation coefficients:        [0.209 0.028 0.293 0.228 0.236 0.038 0.135 0.111 0.278 0.193]
Generated correlations:                         [0.201 0.027 0.281 0.219 0.22  0.035 0.127 0.105 0.264 0.183]
Target spike train rates:       [ 63. 166. 148. 115. 114.]
Generated rates (rounded):      [ 63. 166. 147. 115. 114.]
```