import glob
import os
import numpy as np
import pytorch_lightning as pl
import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.config.model_config import efficientdet_model_param_dict
from effdet.efficientdet import HeadNet
from torch.optim import lr_scheduler
from torchmetrics.detection.map import MeanAveragePrecision

from src.data_scripts.dataset import DatasetEffdet
from src.data_scripts.dataset import get_train_valid_loader
from src.training.transforms import get_train_transform, get_valid_transform
from src.utils.config import TRAIN_CFG

from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def get_effdet_model(num_classes=12, image_size=512, architecture="tf_efficientnetv2_b0"):
    efficientdet_model_param_dict[architecture] = dict(
        name=architecture,
        backbone_name=architecture,
        backbone_args=dict(drop_path_rate=0.2),
        num_classes=num_classes,
        url='', )

    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})

    print(config)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, image_size=512, batch_size=16, lr=0.1, num_classes=12, data_train="data/train", data_valid="data/valid"):
        super().__init__()
        self.losses_list_train = []
        self.losses_list_valid = []
        self.acc_list_train = []
        self.acc_list_valid = []

        self.model = get_effdet_model(num_classes=num_classes, image_size=image_size)
        self.map = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        self.id2label = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12}

        self.image_size=image_size
        self.batch_size=batch_size
        self.data_train=data_train
        self.data_valid=data_valid
        self.lr=lr


    def forward(self, x):
        boxes = self.model(x)
        return boxes

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=False)
        if TRAIN_CFG.step:
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=TRAIN_CFG.step, gamma=TRAIN_CFG.gamma)
        else:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=TRAIN_CFG.step_size, gamma=TRAIN_CFG.gamma)

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        images, annotations, targets = train_batch
        annotations = {"bbox": [target['boxes'].float() for target in targets],
                       "cls": [target['labels'].float() for target in targets]}
        losses = self.model(images, annotations)
        loss = losses['loss']

        self.losses_list_train.append(loss.item())
        self.log(
            "train_loss",
            losses["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_class_loss",
            losses["class_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_box_loss",
            losses["box_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def training_epoch_end(self, outs):
        self.log('train_epoch_loss', np.mean(self.losses_list_train))
        self.losses_list_train = []

    def validation_step(self, val_batch, batch_idx):
        images, annotations, _ = val_batch

        outputs = self.model(images, annotations)
        detections = outputs["detections"]

        self.log(
            "val_loss",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_class_loss",
            outputs["class_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            sync_dist=True,
        )
        self.log(
            "val_box_loss",
            outputs["box_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            sync_dist=True,
        )

        batch_size = images.shape[0]
        preds = []
        for i in range(batch_size):
            # detections: detection results in a tensor with shape [max_det_per_image, 6],
            #  each row representing [x_min, y_min, x_max, y_max, score, class]
            scores = detections[i, ..., 4]
            non_zero_indices = scores.nonzero()
            boxes = detections[i, non_zero_indices, 0:4]
            labels = detections[i, non_zero_indices, 5]
            # non_zero_indices retrieval adds extra dimension into dim=1
            #  so needs to squeeze it out
            preds.append(
                dict(
                    boxes=boxes.squeeze(dim=1),
                    scores=scores[non_zero_indices].squeeze(dim=1),
                    labels=labels.squeeze(dim=1),
                )
            )

        # target needs conversion from y1,x1,y2,x2 to x1,y1,x2,y2
        targets = []
        for i in range(batch_size):
            targets.append(
                dict(
                    boxes=annotations["bbox"][i][:, [1, 0, 3, 2]],
                    labels=annotations["cls"][i],
                )
            )
        self.map.update(preds=preds, target=targets)

    def validation_epoch_end(self, outs):
        mAPs = {"val_" + k: v for k, v in self.map.compute().items()}
        # print(mAPs)

        mAPs_per_class = mAPs.pop("val_map_per_class")
        mARs_per_class = mAPs.pop("val_mar_100_per_class")

        # f1_score = 2 * ((mAP * mAR) / (mAP + mAR))
        f1_score = 2 * ((mAPs['val_map'] * mAPs['val_mar_100']) / max(1e-9, (mAPs['val_map'] + mAPs['val_mar_100'])))

        mAPs["F1 score"] = f1_score

        self.log_dict(mAPs, sync_dist=True)
        self.log(
            "maps_val_map",
            mAPs["val_map"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        print(f"\n mAP: {mAPs['val_map']}, mAR: {mAPs['val_mar_100']}, F1: {f1_score}")

        self.map.reset()

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    ####################
    # DATA RELATED HOOKS
    ####################

    @staticmethod
    def collate_fn(batch):
        batch = [x for x in batch if x is not None]
        images, targets, _ = tuple(zip(*batch))
        images = torch.stack(images).float()

        boxes = [target['boxes'].float() for target in targets]
        labels = [target['labels'].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }
        return images, annotations, targets


    def setup(self, stage=None):

        train_images_paths = glob.glob(f'{self.data_train}/images/*jpg', recursive=False)
        train_images_names = [os.path.basename(x) for x in train_images_paths]

        valid_images_paths = glob.glob(f'{self.data_valid}/images/*jpg', recursive=False)
        valid_images_names = [os.path.basename(x) for x in valid_images_paths]

        train_dataset = DatasetEffdet(self.data_train, train_images_names, get_train_transform(img_size=self.image_size))
        valid_dataset = DatasetEffdet(self.data_valid, valid_images_names, get_valid_transform(img_size=self.image_size))

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            # pin_memory=True,
            collate_fn=self.collate_fn)

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            # pin_memory=True,
            collate_fn=self.collate_fn)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.tr_loader = train_loader
        self.v_loader = valid_loader


    def train_dataloader(self):
        return self.tr_loader

    def val_dataloader(self):
        return self.v_loader

    def test_dataloader(self):
        return self.v_loader

