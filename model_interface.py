# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import torch
import numpy as np
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl


def rho(x, y, weight=None):
    return np.corrcoef(x, y)[0, 1]


def covaraince_loss(pred, label):
    pred_mean = torch.mean(pred)
    label_mean = torch.mean(label)
    pred_centered = pred - pred_mean
    label_centered = label - label_mean
    return -torch.mean(pred_centered * label_centered)

def correlation_loss(pred, label):
    pred_mean = torch.mean(pred)
    label_mean = torch.mean(label)
    pred_centered = pred - pred_mean
    label_centered = label - label_mean
    cov = torch.mean(pred_centered * label_centered)
    pred_std = torch.std(pred)
    label_std = torch.std(label)
    return -cov / (pred_std * label_std)

def balanced_mse_loss(pred, label):
    pred_mean = torch.mean(pred)
    label_mean = torch.mean(label)
    pred_centered = pred - pred_mean
    label_centered = label - label_mean
    pred_var = torch.mean(pred_centered ** 2)
    cov = torch.mean(pred_centered * label_centered)
    return pred_var * 0.1 - cov 
    
    

class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.config_eval_metric()
        self.validation_step_outputs = []

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        y, indices = batch[-2], batch[-1]
        out = self(*batch[:-2])
        if isinstance(out, tuple):
            pred = out[0]
        else:
            pred = out 
        pred = pred.squeeze()
        loss = self.loss_function(pred, y)      
        if isinstance(out, tuple):
            loss += out[1]
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        out_mean = torch.mean(out)
        y_mean = torch.mean(y)
        self.log('train_pred_mean', out_mean, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_label_mean', y_mean, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y, indices = batch[-2], batch[-1]
        out = self(*batch[:-2])
        if isinstance(out, tuple):
            pred = out[0]
        else:
            pred = out 
        pred = pred.squeeze()
        loss = self.loss_function(pred, y)      
        if isinstance(out, tuple):
            loss += out[1]
        self.validation_step_outputs.append({'preds': pred.detach().cpu().numpy(), 'labels': y.detach().cpu().numpy()})
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Here we calculate the evaluation metric
        preds = []
        labels = []
        for ddict in self.validation_step_outputs:
            pred = ddict['preds']
            label = ddict['labels']
            preds.append(pred)
            labels.append(label)
            
        preds = self.all_gather(torch.tensor(preds)).view(-1).detach().cpu().numpy()
        labels = self.all_gather(torch.tensor(labels)).view(-1).detach().cpu().numpy()
        
        
        eval_res = self.eval_metric(preds, labels)
        
        pred_std = np.std(preds)
        label_std = np.std(labels)
        pred_mean = np.mean(preds)
        label_mean = np.mean(labels)
        
        self.print(preds.shape, labels.shape, pred_std, label_std, eval_res)
        
        self.log(f'val_{self.hparams.eval_metric}', eval_res, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_pred_std', pred_std, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_label_std', label_std, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_pred_mean', pred_mean, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_label_mean', label_mean, on_step=False, on_epoch=True, prog_bar=True)
        
        # Make the Progress Bar leave there
        self.print('')
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        if hasattr(self.hparams, 'momentum'):
            momentum = self.hparams.momentum
        else:
            momentum = 0.9
        if self.hparams.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr,
                weight_decay=weight_decay)
        elif self.hparams.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.lr,
                weight_decay=weight_decay, momentum=momentum, nesterov=self.hparams.nesterov)
        else:
            raise ValueError(f'Invalid Optimizer Type {self.hparams.optimizer}!')

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif (loss == 'l1') or (loss == 'mae'):
            self.loss_function = F.l1_loss
        elif loss == 'covariance':
            self.loss_function = covaraince_loss
        elif loss == 'correlation':
            self.loss_function = correlation_loss
        elif loss == 'balanced_mse':
            self.loss_function = balanced_mse_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        else:
            raise ValueError(f"Invalid Loss Type {loss}!")
        
    def config_eval_metric(self):
        eval_metric = self.hparams.eval_metric.lower()
        if eval_metric == 'rho':
            self.eval_metric = rho
        else:
            raise ValueError(f"Invalid Metric Type {eval_metric}!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
