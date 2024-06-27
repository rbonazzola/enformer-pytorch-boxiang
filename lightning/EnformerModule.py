import pytorch_lightning as pl
import torch
import torch.nn.functional as F

OPTIMIZER_PARAMS = {
    "algorithm": "Adam"
}

class EnformerModule(pl.LightningModule):

    def __init__(self, model, optimizer_params=OPTIMIZER_PARAMS):

        super(EnformerModule, self).__init__()

        self.model = model

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # self.loss_params = loss_params
        self.criterion = torch.nn.PoissonNLLLoss(log_input=False, reduction="none")
        self.optimizer_params = optimizer_params


    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)
        

    def _shared_step(self, batch):
        
        pred_human = self.model(batch["human"]["sequence"])
        loss_human = self.criterion(pred_human, batch["human"]["target"])

        # pred_mouse = self.model(batch["mouse"]["sequence"])
        # loss_mouse = self.criterion(pred_mouse, batch["mouse"]["target"])

        loss = loss_human # + loss_mouse
 
        loss_dict = { 
            "loss": loss,
            "loss_human": loss_human
            # "loss_mouse": loss_mouse,
        }  
        
        return loss_dict

    
    def training_step(self, batch, batch_idx):
              
        loss_dict = self._shared_step(batch)
        self.training_step_outputs.append(loss_dict)
        self.log_dict(loss_dict)
        
        return loss_dict


    def on_train_epoch_end(self):
        
        outputs = self.training_step_outputs

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean() 

        self.log_dict(
          { "training_loss": avg_loss },
          on_epoch=True,
          prog_bar=True,
          logger=True,
        )
        
        self.training_step_outputs.clear()
    


    def validation_step(self):
        
        loss_dict = self._shared_step(batch)
        self.validation_step_outputs.append(loss_dict)
        self.log_dict(loss_dict)
        
        return loss_dict


    def on_validation_epoch_end(self):
        
        outputs = self.validation_step_outputs
        
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean() 

        self.log_dict(
            { "val_loss": avg_loss },
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )


    def test_step(self, batch, batch_idx):
        
        loss_dict = self._shared_step(batch)
        self.test_step_outputs.append(loss_dict)
        self.log_dict(loss_dict)
        
        return loss_dict


    def on_test_epoch_end(self):

        outputs = self.test_step_outputs
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        
        self.log_dict(
            { "test_loss": avg_loss },
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.test_step_outputs.clear()


    def configure_optimizers(self):

        algorithm = self.optimizer_params.algorithm
        algorithm = torch.optim.__dict__[algorithm]
        parameters = vars(self.optimizer_params.parameters)
        optimizer = algorithm(self.model.parameters(), **parameters)
        return optimizer