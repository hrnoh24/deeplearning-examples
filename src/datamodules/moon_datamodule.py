from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from sklearn import datasets

class MoonDataset(Dataset):
    def __init__(self, n_steps, n_samples, noise=0.05):
        super().__init__()
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.noise = noise

    def __getitem__(self, idx):
        X, y = datasets.make_moons(n_samples=self.n_samples, noise=self.noise)
        return X, y
    
    def __len__(self):
        return self.n_steps

class MoonDataModule(LightningDataModule):
    def __init__(self, 
                 n_train_steps,
                 n_val_steps,
                 n_train_samples,
                 n_val_samples,
                 batch_size, 
                 num_workers,
                 pin_memory,
                 noise):
        self.save_hyperparameters(logger=False)
        print(self.hparams)

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.data_train = MoonDataset(n_steps=self.hparams.n_train_steps,
                                      n_samples=self.hparams.n_train_samples,
                                      noise=self.hparams.noise)
        self.data_val = MoonDataset(n_steps=self.hparams.n_val_steps,
                                      n_samples=self.hparams.n_val_samples,
                                      noise=self.hparams.noise)
        self.data_val = MoonDataset(n_steps=self.hparams.n_val_steps,
                                      n_samples=self.hparams.n_val_samples,
                                      noise=self.hparams.noise)
        
    def train_dataloader(self):
        return DataLoader(self.data_train,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.data_val,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "moon.yaml")
    dm = hydra.utils.instantiate(cfg)
    train_loader = dm.train_dataloader()
    for x, y in enumerate(train_loader):
        print(x, y)