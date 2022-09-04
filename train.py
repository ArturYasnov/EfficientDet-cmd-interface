import gc
import warnings

import numpy
import argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, Timer

from src.training.trainer_lightning import LitAutoEncoder
from src.utils.config import CFG, TRAIN_CFG

from argparse import ArgumentParser

warnings.filterwarnings("ignore")
data_path = CFG.DATA_PATH

logger = pl.loggers.TensorBoardLogger(TRAIN_CFG.log_dir, name=TRAIN_CFG.exp_name)


def lightning_train_callbacks(swa_lr=0.1, swa_epoch_start=100):
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        save_last=True,
        monitor="maps_val_map",
        mode="max",
        dirpath=f"{CFG.MODEL_SAVE_DIR}/{TRAIN_CFG.exp_name}",
        filename="model-{epoch:02d}-{maps_val_map:.5f}",
        save_weights_only=True
    )
    timer_callback = Timer(duration="02:23:59:59")
    swa = StochasticWeightAveraging(swa_lrs=swa_lr, swa_epoch_start=swa_epoch_start, annealing_strategy="cos", annealing_epochs=5)

    return [
        checkpoint_callback,
        swa,
        timer_callback,
    ]


def pl_trainer_arg_parser(parser):
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--weights", default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--gradient_clip_val", type=float, default=1e-8)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    parser.add_argument("--profiler", default=None)  # "simple", "advanced", pytorch
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--swa_lr", type=float, default=0.1)
    parser.add_argument("--swa_epoch_start", type=int, default=100)
    return parser


def run_pl_main(hparams):
    if hparams.weights:
        model = LitAutoEncoder().load_from_checkpoint(hparams.weights)
    else:
        model = LitAutoEncoder()

    callbacks=lightning_train_callbacks(swa_lr=hparams.swa_lr, swa_epoch_start=hparams.swa_epoch_start)

    trainer = pl.Trainer(
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        max_epochs=hparams.max_epochs,
        precision=hparams.precision,
        gradient_clip_val=hparams.gradient_clip_val,
        fast_dev_run=hparams.fast_dev_run,
        profiler=hparams.profiler,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        callbacks=callbacks,
    )

    trainer.fit(model)
    trainer.validate(model)


if __name__ == "__main__":

    pl.seed_everything(42)

    parser = ArgumentParser()
    parser = pl_trainer_arg_parser(parser)
    args = parser.parse_args()
    run_pl_main(args)
    # safe model.model


"""
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--data", type=str, default="data/data.yaml")
parser.add_argument("--img", type=int, default=512)
parser.add_argument("--weights", type=str, default="")
parser.add_argument("--name", type=str, default="experiment_run")
"""