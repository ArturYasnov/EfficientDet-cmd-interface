import gc
import warnings

import numpy
import argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, Timer

from src.training.trainer_lightning import LitAutoEncoder
from src.utils.config import CFG, TRAIN_CFG

warnings.filterwarnings("ignore")

data_path = CFG.DATA_PATH

logger = pl.loggers.TensorBoardLogger(TRAIN_CFG.log_dir, name=TRAIN_CFG.exp_name)


if __name__ == "__main__":

    pl.seed_everything(42)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        save_last=True,
        monitor="maps_val_map",
        mode="max",
        dirpath=f"{CFG.MODEL_SAVE_DIR}/{TRAIN_CFG.exp_name}",
        filename="model-{epoch:02d}-{maps_val_map:.5f}",
        save_weights_only=True
    )
    timer_callback = Timer(duration="00:12:00:00")

    model = LitAutoEncoder()  # .load_from_checkpoint(path_to_ckpt)

    trainer = pl.Trainer(gpus=1,
                         precision=16,
                         gradient_clip_val=1e-9,
                         callbacks=[
                             checkpoint_callback,
                             # StochasticWeightAveraging(swa_lrs=TRAIN_CFG.lr, swa_epoch_start=2, annealing_strategy="cos", annealing_epochs=5),
                             timer_callback,
                             ],
                         fast_dev_run=False,
                         # overfit_batches=1,


                         profiler=None,  # "simple", "advanced"
                         logger=logger,
                         max_epochs=TRAIN_CFG.epochs,
                         accumulate_grad_batches=TRAIN_CFG.accumulate_bs,
                         )

    trainer.fit(model)
    trainer.validate(model)

    # safe model.model
