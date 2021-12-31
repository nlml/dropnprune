import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

from resnet import resnet32
from dropnprune import Pruner

seed_everything(7)


EXP_NAME = "prune0.4-cosineWarm50-lmult10"
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
PATH_DATASETS = "/home/liam/woven-cifar10-challenge-master/data"
BATCH_SIZE = 128
NUM_WORKERS = int(os.cpu_count() / 2)

train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)


class LitResnet(LightningModule):
    def __init__(self, lr=0.1, create_model_fn=resnet32, planes_per_layer=None):
        super().__init__()

        self.save_hyperparameters()
        if planes_per_layer is None:
            self.model = create_model_fn()
        else:
            self.model = create_model_fn(planes_per_layer=planes_per_layer)
        self.pruner = Pruner(self.model)
        self.num_pruned_so_far = 0

    def forward(self, x):
        return self.model(x)

    def training_epoch_end(self, outputs):
        if hasattr(self.pruner, "_last_scores"):
            self.logger.experiment.add_histogram(
                "scores", self.pruner._last_scores, self.current_epoch
            )

    def training_step(self, batch, batch_idx):
        self.pruner.maybe_run_pruning(batch_idx, self.current_epoch)
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, reduction="none")
        self.pruner.step(loss)
        loss = loss.mean()
        self.log("train_loss", loss)
        self.log("num_pruned_so_far", self.pruner._num_pruned_so_far)
        self.log("num_remaining_params", self.pruner._num_remaining_params)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=200
            ),
            "interval": "epoch",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


if __name__ == "__main__":
    model = LitResnet()

    trainset = torchvision.datasets.CIFAR10(
        root=PATH_DATASETS, train=True, download=True, transform=train_transforms
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
    )
    testset = torchvision.datasets.CIFAR10(
        root=PATH_DATASETS, train=False, download=True, transform=test_transforms
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=200,
        gpus=1,
        logger=TensorBoardLogger(f"lightning_logs", name=EXP_NAME),
        callbacks=[LearningRateMonitor(logging_interval="epoch")],
        # precision=16,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
    )

    trainer.fit(model, trainloader, testloader)
