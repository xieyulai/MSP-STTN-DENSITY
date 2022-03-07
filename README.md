# MSP-STTN

Code and data for the paper [Multi-Size Patched Spatial-Temporal Transformer Network for Short- and Long-Term Grid-based Crowd Flow Prediction]()

Please cite the following paper if you use this repository in your research.
```
Under construction
```

This repo is for **CrowdDensityBJ**, more information can be found in [MSP-STTN](https://github.com/xieyulai/MSP-STTN). 

## CrowdDensityBJ

### Package
```
PyTorch > 1.07
```
Please refer to `requirements.txt`

### Data Preparation
- Processing data according to [MSP-STTN-DATA](https://github.com/xieyulai/MSP-STTN-DATA).
- The `data\` should be like this:
```bash
data
___ DENSITY
```
- Or the processed data can be downloaded from [BAIDU_PAN](https://pan.baidu.com/s/1tLvQx-f-HijwCObUX_AZdA),PW:`woqt`.


### Pre-trained Models
- Several pre-trained models can be downloaded from [BAIDU_PAN](https://pan.baidu.com/s/1g7Ymmdx2FlQn-XlLd7MNhQ), PW:`gy7c`.
- The `model\` should be like this:
```bash
model
___ Imp_1712
___ ___ pre_model_94.pth
___ Imp_3711
___ ___ pre_model_194.pth
___ Imp_5712
___ ___ pre_model_149.pth
___ Imp_5720
___ ___ pre_model_193.pth
___ Imp_5721
    ___ pre_model_1.pth
```
- Use `sh BEST_1.sh` for *scheme1*.
- Use `sh BEST_2.sh` for *scheme2*.

### Train and Test
- Use `sh TRAIN.sh` for short-term prediction.

### Repo Structure
```bash
___ BEST_1.sh
___ BEST_2.sh
___ data # Data
___ dataset
___ model # Store the training weights
___ net # Network struture
___ pre_main_short.py # Main function for shot-term prediction
___ pre_setting_density.yaml # Configuration for short-term prediction
___ README.md
___ record # Recording the training and the test
___ TRAIN.sh
___ util
```
