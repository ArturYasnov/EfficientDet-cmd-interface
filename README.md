# EfficientDet-console-trainer

Script for training the EfficientDet model. \
To run training, place images and labels in the data folder and run the command:
```
python train.py --accelerator 'gpu'  --devices 4  --max_epochs 100  --precision 16
```
