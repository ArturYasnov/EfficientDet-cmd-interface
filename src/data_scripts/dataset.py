import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.training.transforms import get_train_transform, get_valid_transform
from src.utils.config import CFG, TRAIN_CFG
from src.utils.helpers import collate_fn_base

data_path = CFG.DATA_PATH


class DatasetEffdet(Dataset):
    def __init__(self, folder_path, images_paths, transforms=None):
        super().__init__()
        self.folder_path = folder_path
        self.images_paths = images_paths
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_path = f"{self.folder_path}/images/{self.images_paths[index]}"

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        dh, dw, _ = image.shape

        yolo_boxes_path = (
            f"{self.folder_path}/labels/{self.images_paths[index][:-4]}.txt"
        )
        boxes_classes = yolo_to_voc_bbox(yolo_boxes_path, dh=dh, dw=dw)

        boxes = [[int(b) for b in box[:4]] for box in boxes_classes]
        classes = [int(box[4]) for box in boxes_classes]

        boxes = np.array(boxes).astype(int)
        classes = np.array(classes).astype(int)

        boxes = torch.as_tensor(boxes, dtype=torch.float32).contiguous()
        classes = torch.as_tensor(classes, dtype=torch.int64).contiguous()

        target = {}
        target["boxes"] = boxes.contiguous()
        target["labels"] = classes.contiguous()

        if self.transforms:
            for i in range(10):
                sample = self.transforms(
                    **{"image": image, "bboxes": target["boxes"], "labels": classes}
                )
                if len(sample["bboxes"]) > 0:
                    image = sample["image"]
                    target["boxes"] = torch.stack(
                        tuple(map(torch.tensor, zip(*sample["bboxes"])))
                    ).permute(1, 0)
                    target["labels"] = torch.stack(sample["labels"])
                    break

        target["boxes"][:, [0, 1, 2, 3]] = target["boxes"][:, [1, 0, 3, 2]].contiguous()

        target["img_size"] = image.shape[-2:]
        target["img_scale"] = 1.0

        return image, target, index

    def __len__(self) -> int:
        return len(self.images_paths)


def get_train_val_dataset(df_path):
    df = pd.read_csv(f"{CFG.DATA_PATH}/{df_path}")

    df.images_paths = df["images_paths"]
    df = df[df.bboxes != "0"].reset_index(drop=True)

    image_ids = df["images_names"].unique()
    train_ids, valid_ids = train_test_split(image_ids, test_size=0.1, random_state=42)

    train_df = df[df["images_names"].isin(train_ids)].reset_index(drop=True)
    valid_df = df[df["images_names"].isin(valid_ids)].reset_index(drop=True)

    return train_df, valid_df


def get_train_val_pytorch_dataset(df_path):
    df = pd.read_csv(f"{CFG.DATA_PATH}/{df_path}")

    df.images_paths = df["images_paths"]
    df = df[df.bboxes != "0"].reset_index(drop=True)

    image_ids = df["images_names"].unique()
    train_ids, valid_ids = train_test_split(image_ids, test_size=0.1, random_state=42)

    train_df = df[df["images_names"].isin(train_ids)].reset_index(drop=True)
    valid_df = df[df["images_names"].isin(valid_ids)].reset_index(drop=True)

    train_dataset = DatasetEffdet(train_df, get_train_transform())
    valid_dataset = DatasetEffdet(valid_df, get_valid_transform())

    return train_dataset, valid_dataset


def get_train_valid_loader(train_dataset, valid_dataset, collate_fn=collate_fn_base):
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CFG.train_bs,
        shuffle=True,
        num_workers=4,
        # pin_memory=True,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=TRAIN_CFG.valid_bs,
        shuffle=False,
        num_workers=4,
        # pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader


def voc_to_yolo_bbox(bbox, w=1024, h=1024):
    # xmin, ymin, xmax, ymax
    x_center = ((int(bbox[2]) + int(bbox[0])) / 2) / w
    y_center = ((int(bbox[3]) + int(bbox[1])) / 2) / h
    width = (int(bbox[2]) - int(bbox[0])) / w
    height = (int(bbox[3]) - int(bbox[1])) / h
    return [x_center, y_center, width, height]


def yolo_to_voc_bbox(yolo_file_path, dw=1024, dh=1024):
    file = open(yolo_file_path, "r")
    boxes = file.readlines()
    file.close()

    bboxes = []

    for bx in boxes:
        # Split string to float
        c, x, y, w, h = map(float, bx.split(" "))

        x1 = int((x - w / 2) * dw)
        x2 = int((x + w / 2) * dw)
        y1 = int((y - h / 2) * dh)
        y2 = int((y + h / 2) * dh)

        x1 = max(0, x1)
        x2 = max(x1 + 1, min(dw - 1, x2))
        y1 = max(0, y1)
        y2 = max(y1 + 1, min(dh - 1, y2))

        bboxes.append([x1, y1, x2, y2, c])

    return bboxes
