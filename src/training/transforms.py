import albumentations as A
import albumentations.augmentations.transforms as T
from albumentations.pytorch.transforms import ToTensorV2

from src.utils.config import TRAIN_CFG


def get_train_transform():
    return A.Compose([

        A.OneOf(
            [
                A.RandomSizedBBoxSafeCrop(width=TRAIN_CFG.img_size, height=TRAIN_CFG.img_size, erosion_rate=0),
                A.Resize(height=TRAIN_CFG.img_size, width=TRAIN_CFG.img_size, p=1.0),
            ],
            p=1,
        ),
        A.OneOf(
            [
                A.Downscale(scale_min=0.80, scale_max=0.99, p=1),
                A.ImageCompression(quality_lower=80, quality_upper=99, p=1),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.GaussNoise(var_limit=(10.0, 100.0), p=1),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1),
            ],
            p=0.95,
        ),

        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.10, sat_shift_limit=0.10,
                                 val_shift_limit=0.10, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.10,
                                       contrast_limit=0.10, p=0.9),
            A.RandomGamma(gamma_limit=(95, 105), p=1),
        ], p=0.7),

        A.OneOf([
            A.Blur(blur_limit=(3, 3), p=1),
            A.MotionBlur(blur_limit=(3, 5), p=0)
        ], p=0.2),

        A.OneOf([
            A.Cutout(num_holes=8, max_h_size=6, max_w_size=6, fill_value=0, p=1),
            A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, p=1),
        ], p=0.5),

        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})



def get_valid_transform():
    return A.Compose([
        A.Resize(height=TRAIN_CFG.img_size, width=TRAIN_CFG.img_size, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

