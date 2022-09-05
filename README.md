# EfficientDet-console-trainer

Script for training the EfficientDet model. \
To run training, place images and labels into the <data> folder and run the command:
```
python train.py --accelerator 'gpu' --devices 4 --data data/data.yaml --batch-size 32 --img 512 --max_epochs 100 --precision 16 --weights '' --name effdet_trainer
```
