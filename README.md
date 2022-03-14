# Learning Motion-Appearance Co-Attention for Zero-Shot Video Object Segmentation (AMC-Net)

![GitHub License](https://img.shields.io/github/license/isyangshu/DABNet?style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/isyangshu/DABNet?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/isyangshu/DABNet?style=flat-square)


> Official implementation of 'Learning Motion-Appearance Co-Attention for Zero-Shot Video Object Segmentation', ICCV-2021 
> 
> in Pytorch


![image](Figure&Table/Fig7.png)

## Installation


### Enviroment
### Datasets

## Training


## Testing

Please use the files in the `test` folder to generate results.

## Metric

Please use the files in the `EVALVOS` folder to measure metrics.

Taking `test_for_davis.py` as an example:

`Line 13`: Setup `db_info.yml`

`Line 14`: Set the folder of groundtruth

`Line 254`: Set the folder of images

`Line 255 & 256`: Whether to discard the first frame and the last frame

`Line 257`: Save output in `.pkl` format

## Tools
