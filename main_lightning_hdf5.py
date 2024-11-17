from mpi4py import MPI

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from lightning.EnformerModule import EnformerModule

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import random_split, Dataset, DataLoader
from IPython import embed

import argparse

import wandb
import os

from model import enformer

# import mlflow
# import mlflow.pytorch

wandb.login(key='5273f52036ac324f6b4cf895810445ede54a92d3')

torch.set_float32_matmul_precision('medium')


fromd dataset import HDF5Dataset, get_dataloaders


class Datamodule(pl.LightningDataModule):

    def __init__(self, dataset_train, dataset_val, dataset_test, batch_size):
        
        super().__init__()
        self.dataset_train = dataset_train
        self.dataset_val   = dataset_val
        self.dataset_test  = dataset_test

        self.batch_size = batch_size
        
    
    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val,   batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

class SaveEveryNBatches(Callback):
    def __init__(self, save_every_n_batches, model_save_path):
        self.save_every_n_batches = save_every_n_batches
        self.model_save_path = model_save_path

    def on_batch_end(self, trainer, pl_module):
        # Check if we need to save the model
        if trainer.global_step % self.save_every_n_batches == 0:
            model_path = f"{self.model_save_path}/model_step_{trainer.global_step}.ckpt"
            trainer.save_checkpoint(model_path)


def main_lightning(args):
       
    # mlflow.pytorch.autolog(checkpoint_save_weights_only=True, log_every_n_step=1, log_models=True, checkpoint_save_freq=5, checkpoint_save_best_only=False )
    # mlflow.pytorch.autolog()

    # mlflow.set_experiment(experiment_name=args.mlflow_expname)
    # # mlflow.set_experiment()# experiment_name=args.mlflow_expname)
    # with mlflow.start_run(run_name=args.mlflow_run_name):
    #     
    #     # Log basic parameters
    #     mlflow.log_param("batch_size", args.batch_size)
    #     mlflow.log_param("n_gpus", args.n_gpus)
    #     mlflow.log_param("precision", args.precision)
    #     mlflow.log_param("patience", args.patience)
    #     if args.comments:
    #         mlflow.set_tag("comments", args.comments)
    
    from pytorch_lightning.loggers import WandbLogger

    wandb_logger = WandbLogger(
        project="enformer_test", 
        entity="hakyimlab",
        log_model="all"
    )

    dataset_train = HDF5Dataset(
        hdf5_file_human="/grand/TFXcan/imlab/data/enformer_training_data/larger_window/train_human.hdf5",
        hdf5_file_mouse=None
    )
    
    dataset_train, dataset_val = random_split(dataset_train, [32000, 1992])
    
    dataset_test = HDF5Dataset(
        hdf5_file_human="/grand/TFXcan/imlab/data/enformer_training_data/larger_window/test_human.hdf5",
        hdf5_file_mouse=None
    )


    dm = Datamodule(dataset_train, dataset_val, dataset_test, batch_size=args.batch_size)
    
    OPTIMIZER_PARAMS = {
        "algorithm": "Adam",
        "parameters": {
            "lr": 0.001, 
            "betas": (0.9, 0.999), 
            "eps": 1e-8, 
            "weight_decay": 1e-2
        }
    }
    
    model = enformer.Enformer(channels=3*2**10, num_heads=8, num_transformer_layers=11)
    # model = enformer.Enformer(channels=3*2**10, num_heads=1, num_transformer_layers=1)
    # model = enformer.Enformer(channels=3*2**9, num_heads=2, num_transformer_layers=2)
    # model = enformer.Enformer(channels=384, num_heads=2, num_transformer_layers=2)

    enformer_module = EnformerModule(model, optimizer_params=OPTIMIZER_PARAMS)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='/grand/TFXcan/imlab/data/enformer_pytorch_training/checkpoints/my_model_checkpoints',
        filename='{epoch}-{val_loss:.2f}',
        # filename='best-checkpoint',
        every_n_epochs=5,
        # save_top_k=1,
        # mode='min'
    )


    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices = args.n_gpus if torch.cuda.is_available() else None,
        max_epochs=100,
        limit_train_batches=10,
        # progress_bar_refresh_rate=20,
        # logger=pl.loggers.TensorBoardLogger("lightning_logs/", name="my_model"),
        callbacks=[checkpoint_callback],
        strategy='ddp_find_unused_parameters_true'
    )

    trainer.fit(enformer_module, dm)

    #train_metrics = trainer.callback_metrics
    # for metric_name, metric_value in train_metrics.items():
        # mlflow.log_metric(metric_name, metric_value)

    # Log the trained model
    # mlflow.pytorch.log_model(enformer_module, "model")
    
    trainer.test(enformer_module, dm)
    # mlflow.end_run()


def get_dataloaders():
         
    dataset_train = HDF5Dataset(
        hdf5_file_human="/grand/TFXcan/imlab/data/enformer_training_data/larger_window/train_human.hdf5",
        hdf5_file_mouse=None
    )

    dataset_test = HDF5Dataset(
        hdf5_file_human="/grand/TFXcan/imlab/data/enformer_training_data/larger_window/test_human.hdf5",
        hdf5_file_mouse=None
    )

    dataset_train, dataset_val = random_split(dataset_train, [32000, 1992])

    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size)
    val_dataloader   = DataLoader(dataset_val,   batch_size=args.batch_size)
    test_dataloader  = DataLoader(dataset_test,  batch_size=args.batch_size)

    return train_dataloader, val_dataloader, test_dataloader


class Trainer():

    def __init__(self, model, dataloaders, optimizer, device="cuda"):

        self.model = model
        self.train_dataloader, self.val_dataloaders, self.test_dataloaders = dataloaders
        self.optimizer = optim.Adam(self.model.parameters())


    def initialize(self):

        selt.iter = iter(self.train_dataloader)
        self.criterion = nn.PoissonNLLLoss(log_input=False, reduction="none")
        


    def train_step(self):

        try:
            batch     = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data)
            batch     = next(self.iter)

        sequences = batch["sequence"]
        targets   = batch["target"].to(self.device)

        # for head in ["human", "mouse"]:
        for head in ["human"]:

            self.optimizer.zero_grad()
            pred = self.model(sequences)
            loss = self.criterion(pred[head], targets)
            loss.mean().backward()

            print(f"head: {head}; loss: {loss.mean().item():03f}")

            self.optimizer.step()



def configure_optimizer(model, optim_parameters):

    algorithm = optimizer_params["algorithm"]
    algorithm = torch.optim.__dict__[algorithm]
    optimizer = algorithm(model.parameters(), **optimizer_params['parameters'])
    return optimizer


def main_plain_torch(args):

    OPTIMIZER_PARAMS = {
        "algorithm": "Adam",
        "parameters": {
            "lr": 0.001, 
            "betas": (0.9, 0.999), 
            "eps": 1e-8, 
            "weight_decay": 1e-2
        }
    }

    device = "cuda"
    model = enformer.Enformer(channels=3*2**10, num_heads=8, num_transformer_layers=11)
    dataloaders = get_dataloaders()

    optimizer = configure_optimizer(model, OPTIMIZER_PARAMS)
    trainer = Trainer(model, dataloaders, optimizer, device=device)

    for _ in range(20):
        trainer.train_step()
   


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", "--batch-size", type=int, default=1)    
    parser.add_argument("--n_gpus", type=int, default=-1)    
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--mlflow_experiment", "--expname", dest="mlflow_expname", type=str, default="Enformer - test")       
    parser.add_argument("--mlflow_run_name", "--run_name", dest="mlflow_run_name", type=str)       
    parser.add_argument("--comments", type=str, help="Comments to be added to the MLflow run as a tag.")

    args = parser.parse_args()

    main(args)