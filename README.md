# EfficientDet-console-trainer

Script for training the EfficientDet model from command line. \
Based on https://github.com/rwightman/efficientdet-pytorch implementation of EfficientDet.

To run training, place images and labels into the Data folder and run the command:
```
python train.py --accelerator 'gpu' --devices 4 --data data/data.yaml --batch-size 32 --lr 1e-4  --img 512 --max_epochs 100 --precision 16 --name effdet_trainer
```
