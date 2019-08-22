# Image Segmentation Task

![](docs/overview.png)
<br>

## Installation
Use Python 3.6, please install the required libraries as follows;
```
pip install -r requirements.txt
```

## Algorithm
I finished this project inspired by **Maximum Classifier Discrepancy for Domain Adaptation**: [[Paper (arxiv)]](https://arxiv.org/abs/1712.02560).

## Train
- Dataset
    - **Source**: train set and val set, **Target**: test set
- Network
    - ResFCN

```
python adapt_trainer.py --res 50 --epochs 20 --lr 0.001 --adjust_lr --train_img_shape (600, 800) --source_list data/source.txt --source_label_list data/source_label.txt --target_list data/target.txt
```
Trained models will be saved as "snapshot/model/fcn-res50-EPOCH.pth.tar"

## Test
```
python adapt_tester.py --test_list data/test.txt --test_img_shape (600, 800) --trained_checkpoint snapshot/model/fcn-res50-test.pth.tar
```

Results will be saved under "snapshot/test/"

## Eval
For convenience, I just merged train set and val set as source domain and random selected a trained model for test.

