from lightning.EnformerModule import EnformerModule
from model import enformer
import os
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from IPython import embed
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint

class BasenjiDataset(Dataset):

    def __init__(self, data_dir, num_regions_per_file=256, file_filter="train"):
        
        self.data_dir = data_dir
        self.num_regions_per_file = num_regions_per_file
        
        self._file_filter = file_filter
        self.region_files = self._load_region_files(file_filter)
        
        self._current_file_idx = None
        self._current_file     = None
        self.current_data      = None

        
    def _load_region_files(self, file_filter):
        return sorted([ os.path.join(self.data_dir, x) for x in os.listdir(self.data_dir) if file_filter in x ]) 


    def _get_region_data(self, file_idx):
        
        if self._current_file_idx == file_idx:
            # self._current_file
            return self._current_file, self.current_data
        else:
            self._current_file_idx = file_idx
            self._current_file     = self.region_files[file_idx]
            return self._current_file, torch.load(self._current_file)
            
    
    def __len__(self):
        
        return self.num_regions_per_file * self.region_files

    
    def __getitem__(self, idx):
        
        file_idx = idx // self.num_regions_per_file         
        point_idx = idx % self.num_regions_per_file

        previous_file = self._current_file         
        self._current_file, self.current_data = self._get_region_data(file_idx)
        
        if previous_file != self._current_file:
            print(self._current_file)

        # print(point_idx)
        
        return {
            "sequence": self.current_data["sequence"][point_idx],
            "target":   self.current_data["target"][point_idx]
        }


class ContiguousSampler(Sampler):
    
    def __init__(self, dataset, batch_size):
        
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.num_regions_per_file = dataset.num_regions_per_file
        self.file_indices = list(range(len(dataset.region_files)))
    
    
    def __iter__(self):
        
        np.random.shuffle(self.file_indices)
        indices = []
        
        for file_idx in self.file_indices:
            start_idx = file_idx * self.num_regions_per_file
            for i in range(0, self.num_regions_per_file, self.batch_size):
                batch_indices = list(range(
                    start_idx + i, 
                    min(start_idx + i + self.batch_size, start_idx + self.num_regions_per_file)
                ))
                indices.extend(batch_indices)
        
        return iter(indices)
    
    
    def __len__(self):
        return len(self.dataset)


class Datamodule(pl.LightningDataModule):

    def __init__(self, dataset_train, dataset_val, dataset_test, batch_size):

        self.dataset_train = dataset_train
        self.dataset_val   = dataset_val
        self.dataset_test  = dataset_test

        self.batch_size = batch_size
        
    
    def setup(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train, 
            batch_size=self.batch_size, 
            sampler=ContiguousSampler(self.dataset_train, self.batch_size)
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,   
            batch_size=self.batch_size, 
            sampler=ContiguousSampler(self.dataset_val, self.batch_size)
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,  
            batch_size=self.batch_size, 
            sampler=ContiguousSampler(self.dataset_test, self.batch_size)
        )



def main(args):

    data_dir = "/grand/TFXcan/imlab/data/enformer_training_data/delete/basenji_data_pt/human"

    dataset_train = BasenjiDataset(data_dir=data_dir, file_filter="train")
    dataset_val   = BasenjiDataset(data_dir=data_dir, file_filter="val")
    dataset_test  = BasenjiDataset(data_dir=data_dir, file_filter="test")
    
    dm = Datamodule(dataset_train, dataset_val, dataset_test, batch_size=args.batch_size)
        
    # model = enformer.Enformer(channels=1536, num_heads=8, num_transformer_layers=11)
    model = enformer.Enformer(channels=2, num_heads=2, num_transformer_layers=2)
    enformer_module = EnformerModule(model)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='my_model_checkpoints',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    trainer = pl.Trainer(
        accelerator='gpu', #'gpu' if torch.cuda.is_available() else 'cpu',
        devices = 4 if torch.cuda.is_available() else None,
        max_epochs=2,
        # progress_bar_refresh_rate=20,
        # logger=pl.loggers.TensorBoardLogger("lightning_logs/", name="my_model"),
        callbacks=[checkpoint_callback]
    )

    trainer.fit(enformer_module, dm)

    trainer.test(enformer_module, dm)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", "--batch-size", type=int, default=32)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--mlflow_experiment", "--expname", dest="mlflow_expname", type=str)       
    parser.add_argument("--mlflow_run_name", "--run_name", dest="mlflow_run_name", type=str)       
    parser.add_argument("--comments", type=str, help="Comments to be added to the MLflow run as a tag.")
    # parser.add_argument()

    args = parser.parse_args()

    main(args)