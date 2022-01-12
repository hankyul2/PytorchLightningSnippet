import os

import torch
from torch.utils.data import DataLoader

from torchvision import models, transforms
from torchvision.datasets import CIFAR10

from pytorch_lightning import LightningModule, LightningDataModule, Trainer

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


class CIFAR(LightningDataModule):
    def __init__(self, img_size=32, batch_size=32):
        super().__init__()
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.batch_size = batch_size

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
        self.test_ds = CIFAR10(root='data', train=False, download=False, transform=self.test_transforms)

    def test_dataloader(self):
        return DataLoader(self.test_ds, num_workers=4, batch_size=self.batch_size, shuffle=False)


class BasicModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(num_classes=10, pretrained=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return y, y_hat.argmax(dim=-1)

    def test_epoch_end(self, outputs):
        results = torch.zeros((10, 10)).to(self.device)
        for output in outputs:
            for label, prediction in zip(*output):
                results[int(label), int(prediction)] += 1
        torch.distributed.reduce(results, 0, torch.distributed.ReduceOp.SUM)
        acc = results.diag().sum() / results.sum()
        if self.trainer.is_global_zero:
            self.log("test_metric", acc, rank_zero_only=True)
            self.trainer.results = results
        
    
if __name__ == '__main__':
    data = CIFAR(batch_size=512)
    model = BasicModule()
    trainer = Trainer(max_epochs=2, gpus='0,1', strategy="ddp", precision=16)
    test_results = trainer.test(model, data)
    if trainer.is_global_zero:
        print(test_results)
        print(trainer.results)
