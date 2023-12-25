# Paper in 2023 Information Processing & Management

Pytorch code for 2023 IPM paper: "Not all fake news is semantically similar: Contextual semantic representation learning for multimodal fake news detection"

# Overview

This directory contains code necessary to run the CSFND. CSFND is a multimodal fake news detection model. See our paper for details on the code.


# Dataset

The meta-data of the Weibo and Twitter datasets used in our experiments are available in their papers.

Besides, the meta-data can be downloaded from the following project:

[dataset in MRML](https://github.com/plw-study/MRML)

# Requirements

It is recommended to create a conda virtual environment to run the code.
The python version is python-3.7.9. The detailed version of some packages is available in requirements.txt. You can install all the required packages using the following command:

``` 
$ conda install --yes --file requirements.txt
```

# Running the code

The run.py is the main file for running the code.

``` 
$ python run.py
```

# Reference

Detailed data analysis and method are in our paper. If you are insterested in this work, and want to use the dataset or codes in this repository, please star this repository and cite by:
```
@article{PENG2024103564,
title = {Not all fake news is semantically similar: Contextual semantic representation learning for multimodal fake news detection},
journal = {Information Processing & Management},
volume = {61},
number = {1},
pages = {103564},
year = {2024},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2023.103564},
url = {https://www.sciencedirect.com/science/article/pii/S0306457323003011},
author = {Liwen Peng and Songlei Jian and Zhigang Kan and Linbo Qiao and Dongsheng Li},
keywords = {Fake news detection, Multimodal learning, Social network, Representation learning, Deep learning}
}
```
