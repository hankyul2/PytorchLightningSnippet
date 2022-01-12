# reference 1 (loss): https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#automatic-logging
# reference 2 (metric): https://torchmetrics.readthedocs.io/en/latest/references/modules.html#base-class

import os
from copy import deepcopy

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from torchvision import models, transforms
from torchvision.datasets import CIFAR10

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from torchmetrics import Accuracy, MetricCollection

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


class CIFAR(LightningDataModule):
    def __init__(self, img_size=32, batch_size=32):
        super().__init__()
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.batch_size = batch_size
        self.train_transforms = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def prepare_data(self) -> None:
        CIFAR10(root='data', train=True, download=True)
        CIFAR10(root='data', train=False, download=True)
    
    def setup(self, stage=None):
        self.train_ds = CIFAR10(root='data', train=True, download=False, transform=self.train_transforms)
        self.valid_ds = CIFAR10(root='data', train=False, download=False, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_ds, num_workers=4, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, num_workers=4, batch_size=self.batch_size, shuffle=False)

class BasicModule(LightningModule):
    def __init__(self, lr=0.01):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        
        metric = MetricCollection({'top@1': Accuracy(top_k=1), 'top@5': Accuracy(top_k=5)})
        self.train_metric = metric.clone(prefix='train_')
        self.valid_metric = metric.clone(prefix='valid_')
    
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self.shared_step(*batch, self.train_metric)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(*batch, self.valid_metric)

    def shared_step(self, x, y, metric):
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        if metric == self.valid_metric:
            self.log('val_loss', loss, on_step=True, sync_dist=True)
        self.log_dict(metric(y_hat, y), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return SGD(self.model.parameters(), lr=self.lr)

if __name__ == '__main__':
    data = CIFAR(batch_size=512)
    model = BasicModule(lr=0.01)
    trainer = Trainer(max_epochs=2, gpus='0,1', strategy="ddp", precision=16)
    trainer.fit(model, data)
