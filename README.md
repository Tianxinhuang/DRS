# RFNet
The codes for Deep Residual Surrogate Model published in INFORMATION SCIENCES.

## Environment
* Python 3.6.9
* numpy 1.14.5
* [SMTorg package](https://github.com/SMTorg/smt)

## Dataset
The Test problems are intergrated in the `tf_util`, which mainly come from [Ackley](http://www.sfu.ca/~ssurjano/ackley.html) and [SMT](https://smt.readthedocs.io/en/latest/_src_docs/problems.html).

## Usage

1. Optimize different models

```
Python3 test_surro.py
```
Different models will be optimized and tested on the benmark problems required in file `functions2.txt`, the results will be saved in file `result.xls`.

2. Evaluate the performances

```
Python3 draw_pics.py
```
The performances could be compared through histograms.


## Citation
If you find our work useful for your research, please cite:
```
@article{huang2022deep,
  title={Deep residual surrogate model},
  author={Huang, Tianxin and Liu, Yong and Pan, Zaisheng},
  journal={Information Sciences},
  volume={605},
  pages={86--98},
  year={2022},
  publisher={Elsevier}
}
```
