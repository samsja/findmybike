# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# ## Import

from typing import Optional

# +
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

torch.__version__
# -

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from vision_transformers.data.utils.utils_plot import imshow
from vision_transformers.transformers.vit.vit_lightning import ViTModule
from vision_transformers.utils import ProgressiveImageResizing

from vision_transformers.data.utils.augmentation import (
    get_transforms,
    get_transforms_val,
)

from web2dataset.torch_dataset import TorchDataset

# ## Param

## PARAMS
batch_size = 4
num_workers = 4
patience = 10
model_path = "../data/bike_dataset"
epochs = 100


# ## Dataset

class CustomW2D(TorchDataset):
    def __init__(self, path: str, transform=None):
        super().__init__(path, transform)

        self.classes = []

        for doc in self.docs:
            if (class_ := doc.tags["tag"]["origin"]) not in self.classes:
                self.classes.append(class_)

        self.classes = {class_: i for i, class_ in enumerate(self.classes)}

    def __getitem__(self, key):
        tensor, tags = super().__getitem__(key)
        return tensor, self.classes[tags["tag"]["origin"]]


# +
_IMAGE_SHAPE_VAL = (224, 224)
_IMAGE_SHAPE = (224, 224)


class BikeData(pl.LightningDataModule):
    _VAL_PERCENTAGE = 0.1

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        self.train_dataset = CustomW2D(
            self.data_path,
            transform=get_transforms(_IMAGE_SHAPE),
        )

        self.classes = self.train_dataset.classes

        self.val_dataset_base = CustomW2D(
            self.data_path, transform=get_transforms_val(_IMAGE_SHAPE_VAL)
        )

        val_len = int(self._VAL_PERCENTAGE * len(self.train_dataset))

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset,
            [
                len(self.train_dataset) - val_len,
                val_len,
            ],
            generator=torch.Generator().manual_seed(42),
        )

        self.val_dataset.dataset = self.val_dataset_base

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


# -

data = BikeData(model_path, batch_size, num_workers)

data.setup()

# ## Model

import timm
from vision_transformers.transformers.lightning_module import TransformersModule


class EfficientNet(TransformersModule):
    def __init__(self, num_classes: int, pretrained=True, *args, **kwargs):
        model = timm.create_model("efficientnet_b0", pretrained=True)
        model.classifier = nn.Linear(
            in_features=model.classifier.in_features, out_features=num_classes
        )
        super().__init__(model, *args, **kwargs)


model = EfficientNet(
    num_classes=len(data.classes),
    lr=1e-3,
    one_cycle_scheduler={
        "max_lr": 0.5,
        "steps_per_epoch": len(data.train_dataloader()),
        "epochs": epochs,
    },
)

# + [markdown] tags=[]
#
# ## Training
# -

increase_image_shape = ProgressiveImageResizing(
    data, epoch_final_size=5, n_step=5, init_size=30, final_size=224
)

callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=patience, strict=False),
    increase_image_shape,
    LearningRateMonitor(logging_interval="step"),
    ModelCheckpoint(
        dirpath="../data/models_checkpoint",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=3,
        save_last=True,
        monitor='val_loss',
    ),
]

trainer = pl.Trainer(
    gpus=1,
    max_epochs=epochs,
    check_val_every_n_epoch=1,
    log_every_n_steps=3,
    callbacks=callbacks,
    gradient_clip_val=1.0,
    precision=16,
)

trainer.fit(model, data)
