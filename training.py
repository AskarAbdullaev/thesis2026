import time
import os
import json
import gc

from dataset import CustomDataSet
from gnn import GNNModel

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.loader.dataloader import DataLoader
from tqdm import tqdm
from sklearn.metrics import (confusion_matrix,
                             f1_score,
                             average_precision_score,
                             accuracy_score,
                             roc_auc_score,
                             balanced_accuracy_score,
                             recall_score,
                             precision_score,
                             jaccard_score)


class TrainingWrapper():

    def __init__(self, logs_dir: str = 'GNNLogs'):

        self._logs_dir = logs_dir
        self._model = None
        self._dataset = None
        self._optimizer = None
        self._scheduler = None
        self._loss_function = None

        self._train_set_indices = None
        self._valid_set_indices = None
        self._test_set_indices = None

        self._train_loader = None
        self._valid_loader = None
        self._test_loader = None

        self._current_epoch = 0

        self._logs = {}

        self._device = 'cpu'
        self._show_progress = True
        self._patience = 5
        self._epochs = 100
        self._verbose = 2

        self._main_settings = {'device': self._device, 'progress': self._show_progress, 'patience': self._patience, 'epochs': self._epochs}
        self._model_settings = {}
        self._data_settings = {}
        self._optimizer_settings = {}
        self._scheduler_settings = {}
        self._loss_settings = {}
        self._loader_settings = {}

        self.logs = {}
        pass

    def _reset(self):
   
        self._model = None
        self._dataset = None
        self._optimizer = None
        self._scheduler = None
        self._loss_function = None

        self._train_set_indices = None
        self._valid_set_indices = None
        self._test_set_indices = None

        self._train_loader = None
        self._valid_loader = None
        self._test_loader = None

        self._current_epoch = 0

        self._logs = {}

        self._model_settings = {}
        self._data_settings = {}
        self._optimizer_settings = {}
        self._scheduler_settings = {}
        self._loss_settings = {}
        self._loader_settings = {}

        self.logs = {}


    def set_settings(self, device: str = None,
                     show_progress: bool = None,
                     patience: int = None,
                     epochs: int = None,
                     verbose: int = None):
        
        if self._verbose == 3:
            print('Updating basic settings')

        if device is not None:
            if self._verbose == 3:
                print(f'\tdevice: {self._device} -> {device}')
            self._device = device
            self._main_settings.update({'device': device})
        if show_progress is not None:
            if self._verbose == 3:
                print(f'\tshow progress: {self._show_progress} -> {show_progress}')
            self._show_progress = show_progress
            self._main_settings.update({'progress': show_progress})
        if patience is not None:
            if self._verbose == 3:
                print(f'\tpatience: {self._patience} -> {patience}')
            self._patience = patience
            self._main_settings.update({'patience': patience})
        if epochs is not None:
            if self._verbose == 3:
                print(f'\tepochs: {self._epochs} -> {epochs}')
            self._epochs = epochs
            self._main_settings.update({'epochs': epochs})
        if verbose is not None:
            if self._verbose == 3:
                print(f'\tverbose: {self._verbose} -> {verbose}')
            self._verbose = verbose
            self._main_settings.update({'verbose': verbose})
        

    @property
    def settings(self):
        return (self._main_settings | self._model_settings |
                self._data_settings | self._optimizer_settings | self._scheduler_settings | 
                self._loss_settings | self._loader_settings)

    def clear_logs(self):
        self._logs = {}

    @property
    def model(self):
        return self._model
    
    @property
    def dataset(self):
        return self._dataset
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def scheduler(self):
        return self._scheduler
    
    @property
    def loss_function(self):
        return self._loss_function

        
    def initialize_model(self, load: str | None = None, **model_kwargs):
        """
        Create or renew the stored model

        Can be tuned:

        geometric_dim: int = 5,
        encoder_hidden_dim: int = 16,
        encoder_depth: int = 1,
        gnn_hidden_dim: int = 128,
        gnn_depth: int = 2,
        gnn_activation: str = 'silu',
        protein_dim: int = 38,
        dense_activation: str = 'elu',
        dense_dim: int = 128,
        dense_depth: int = 1,
        dropout: float = 0.1,
        batch_norm: bool = False,
        seed: int = 1234
        """

        if isinstance(load, str):
            with open(load, 'rb') as file:
                self._model: GNNModel = torch.load(file)
            if self._verbose >= 1:
                print(f'Model is Loaded from {load}: {self._model.total_params} parameters')
        else:
            if 'seed' not in model_kwargs:
                model_kwargs.update({'seed': 1234})

            torch.manual_seed(model_kwargs['seed'])

            self._model = GNNModel(**model_kwargs)
            
            if self._verbose >= 1:
                print(f'Model is Created: {self._model.total_params} parameters')
        

        self._model_settings = {
            'geometric_dim': self._model.geometric_dim,
            'encoder_hidden_dim': self._model.encoder_hidden_dim,
            'encoder_depth': self._model.encoder_depth,
            'dropout': self._model.dropout,
            'batch_norm': self._model.batch_norm,
            'dense_activation': self._model.dense_activation,
            'gnn_activation': self._model.gnn_activation,
            'gnn_depth': self._model.gnn_depth,
            'dense_depth': self._model.dense_depth,
            'dense_dim': self._model.dense_dim,
            'gnn_hidden_dim': self._model.gnn_hidden_dim,
            'model_seed': self._model.seed,
            'total_params': self._model.total_params
        }

        if self._verbose >= 2:
            for key, value, in self._model_settings.items():
                print('\t' + key.ljust(25) + f': {value}')
        if self._verbose >= 3:
            print('Architecture:')
            print(self._model)

    def save_model(self, model_path: str):
        """
        Save the current model to a path
        """
        if self.model is None:
            raise RuntimeError('There is no model to save')
        
        torch.save(self.model, model_path)
        if self._verbose >= 1:
            print(f'Model is saved to {model_path}')

    def initialize_dataset(self,
                           **dataset_kwargs):
        """
        Create or Renew the stored dataset
        """
        if self._verbose >= 2:
            print('Collecting Samples...')
        
        self._dataset = CustomDataSet(
            **(dataset_kwargs | {'show_progress': self._verbose >= 1})
            )
        self._data_settings = ({
            'reduction': self._dataset.reduction,
            'total_samples': len(self._dataset),
            'use_folds': self._dataset.use_folds,
        })
        if self._verbose >= 3:
            for key, value, in self._data_settings.items():
                print('\t' + key.ljust(15) + f': {value}')

        return
    
    def split_dataset(self,
                      train_folds: list[int | str] | None = None,
                      valid_folds: list[int | str] | None = None,
                      test_folds: list[int | str] | None  = None):
        """
        Create or Renew the split of the dataset

        Args:
            train_folds (list[int | str], optional): folds to use for training. Defaults to [101].
            valid_folds (list[int | str], optional): folds to use for validation. Defaults to [102].
            test_folds (list[int | str], optional): folds to use for testing. Defaults to [103].
        """
        if self._verbose >= 2:
            print('Splitting Samples...')

        if self.dataset is None:
            raise RuntimeError('Initialize Dataset before the Split')
        
        if train_folds:
            self._train_set_indices = []
            for i in train_folds:
                self._train_set_indices.extend(self._dataset.fold_indices[i])
            self._data_settings.update({'train_samples': len(self._train_set_indices),
                                        'train_folds': train_folds})
            if self._verbose >= 3:
                print(f'\tTrain Subset      : fold(s) {train_folds}, {len(self._train_set_indices)} samples')
        else:
            self._data_settings.update({'train_samples': 0,
                                        'train_folds': None})
            if self._verbose >= 3:
                print('\tTrain Subset      : Empty')

        if valid_folds:
            self._valid_set_indices = []
            for i in valid_folds:
                self._valid_set_indices.extend(self._dataset.fold_indices[i])
            self._data_settings.update({'valid_samples': len(self._valid_set_indices),
                                        'valid_folds': valid_folds})
            if self._verbose >= 3:
                print(f'\tValidation Subset : fold(s) {valid_folds}, {len(self._valid_set_indices)} samples')
        else:
            self._data_settings.update({'valid_samples': 0,
                                        'valid_folds': None})
            if self._verbose >= 3:
                print('\tValidation Subset : Empty')

        if test_folds:
            self._test_set_indices = []
            for i in test_folds:
                self._test_set_indices.extend(self._dataset.fold_indices[i])
            self._data_settings.update({'test_samples': len(self._test_set_indices),
                                        'test_folds': test_folds})
            if self._verbose >= 3:
                print(f'\tTest Subset       : fold(s) {test_folds}, {len(self._test_set_indices)} samples')
        else:
            self._data_settings.update({'test_samples': 0,
                                        'test_folds': None})
            if self._verbose >= 3:
                print('\tTest Subset       : Empty')
        
        return
    
    def initialize_optimizer(self, optimizer=torch.optim.Adam, **optimizer_kwargs):
        """
        Create or Renew the optimizer

        Args:
            optimizer (_type_, optional): class of optimizer. Defaults to torch.optim.Adam.
        """
        if self.model is None:
            raise RuntimeError('Initialize Model before the Optimizer')

        self._optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)
        self._optimizer_settings = {'optimizer': optimizer} | optimizer_kwargs
        if self._verbose >= 1:
            print(f'Optimizer is created    : {self.optimizer.__class__}')
        if self._verbose >= 2:
            for key, value, in self.optimizer.__dict__['defaults'].items():
                if key[0] != '_' and not isinstance(value, list):
                    print('\t' + key.ljust(15) + f': {value}')

    def initialize_scheduler(self, scheduler=torch.optim.lr_scheduler.StepLR, **scheduler_kwargs):
        """
        Create or Renew the learning rate scheduler

        Args:
            optimizer (_type_, optional): class of optimizer. Defaults to torch.optim.Adam.
        """
        if self.optimizer is None:
            raise RuntimeError('Initialize Optimizer before the Scheduler')

        if scheduler_kwargs is None:
            scheduler_kwargs = {'step_size': 1, 'gamma': 0.8}
        self._scheduler = scheduler(self.optimizer, **scheduler_kwargs)
        self._scheduler_settings = {'scheduler': self._scheduler.__class__,
                                    'scheduler_gamma': self._scheduler.gamma,
                                    'scheduler_step': self._scheduler.step_size}
        if self._verbose >= 1:
            print(f'LR Scheduler is created    : {self.scheduler.__class__}')
        if self._verbose >= 2:
            for key, value, in self._scheduler_settings.items():
                if key[0] != '_' and not isinstance(value, list):
                    print('\t' + key.ljust(17) + f': {value}')


    def initialize_loss_function(self, loss_fn = nn.BCEWithLogitsLoss, **loss_fn_kwargs):
        """
        Create or Renew the loss function

        Args:
            loss_fn (nn.Module, optional): class of loss function. Defaults to nn.BCEWithLogitsLoss.
        """

        self._loss_settings = {'loss_fn': str(loss_fn)}

        if 'pos_weight' in loss_fn_kwargs:
            loss_fn_kwargs['pos_weight'] = torch.tensor(loss_fn_kwargs['pos_weight'], dtype=torch.float32).to(self._device)
    
        self._loss_function = loss_fn(**loss_fn_kwargs)

        self._loss_settings.update({
            'pos_weight': self._loss_function.pos_weight,
            'loss_reduction': self._loss_function.reduction
        })

        if self._verbose >= 1:
            print(f'Loss function is created: {self.loss_function}')
        if self._verbose >= 2:
            for key, value in self._loss_settings.items():
                print('\t' + key.ljust(10) + ': ' + str(value))
    
    class DatasetWrapper(Dataset):

        def __init__(self, dataset: CustomDataSet, use_indices: list[int]):

            self.dataset = dataset
            self.valid_indices = use_indices
            self.secondary_mapping = np.arange(len(use_indices))

        def __len__(self):
            return len(self.valid_indices)
        
        def __getitem__(self, index):

            primary_index = self.valid_indices[index]
            return self.dataset[primary_index]

    def initialize_loaders(self,
                           train_loader_kwargs: dict = None,
                           valid_loader_kwargs: dict = None,
                           test_loader_kwargs: dict = None):
        
        self._loader_settings = {}

        if self.dataset is None:
            raise RuntimeError('Initialize Dataset before the Loaders')
        
        if self._train_set_indices is None and self._valid_set_indices is None and self._test_set_indices:
            raise RuntimeError('Initialize Split before the Loaders')
        
        if train_loader_kwargs is None:
            train_loader_kwargs = {'batch_size': 64, 'shuffle': True}
        if valid_loader_kwargs is None:
            valid_loader_kwargs = {'batch_size': 64, 'shuffle': False}
        if test_loader_kwargs is None:
            test_loader_kwargs = {'batch_size': 64, 'shuffle': False}

        if self._verbose >= 2:
            print('Getting Loaders...')

        torch.manual_seed(1234)

        if self._train_set_indices:
            train_set_wrapped = self.DatasetWrapper(self.dataset, self._train_set_indices)
            self._train_loader = DataLoader(train_set_wrapped, **train_loader_kwargs)
            self._loader_settings.update({'train_batch_size': self._train_loader.batch_size,
                                          'train_shuffle': train_loader_kwargs.get('shuffle', False),
                                          'train_num_workers': self._train_loader.num_workers})
            if self._verbose >= 3:
                print('\tTrain Loader:')
                for key, value, in self._loader_settings.items():
                    if 'train' in key:
                        print('\t\t' + key.ljust(20) + f': {value}')
        elif self._verbose >= 3:
            print('\tTrain Loader:\n\t\tEmpty')

        if self._valid_set_indices:
            valid_set_wrapped = self.DatasetWrapper(self.dataset, self._valid_set_indices)
            self._valid_loader = DataLoader(valid_set_wrapped, **valid_loader_kwargs)
            self._loader_settings.update({'valid_batch_size': self._valid_loader.batch_size,
                                          'valid_shuffle': valid_loader_kwargs.get('shuffle', False),
                                          'valid_num_workers': self._valid_loader.num_workers})
            if self._verbose >= 3:
                print('\tValid Loader:')
                for key, value, in self._loader_settings.items():
                    if 'valid' in key:
                        print('\t\t' + key.ljust(20) + f': {value}')
        elif self._verbose >= 3:
            print('\tValid Loader:\n\t\tEmpty')
            
        if self._test_set_indices:
            test_set_wrapped = self.DatasetWrapper(self.dataset, self._test_set_indices)
            self._test_loader = DataLoader(test_set_wrapped, **test_loader_kwargs)
            self._loader_settings.update({'test_batch_size': self._test_loader.batch_size,
                                          'test_shuffle': test_loader_kwargs.get('shuffle', False),
                                          'test_num_workers': self._test_loader.num_workers})
            if self._verbose >= 3:
                print('\tTest Loader:')
                for key, value, in self._loader_settings.items():
                    if 'test' in key:
                        print('\t\t' + key.ljust(20) + f': {value}')
        elif self._verbose >= 3:
            print('\tTest Loader:\n\t\tEmpty')
            

    @property
    def train_loader(self):
        return self._train_loader
    
    @property
    def valid_loader(self):
        return self._valid_loader
    
    @property
    def test_loader(self):
        return self._test_loader
    
    def _metrics(self,
                 true_labels,
                 logits,
                 mask: np.ndarray = None,
                 prefix: str = '') -> dict:
        """
        Returns a dictionary of basic metrics
        """

        if mask is None:
            mask = np.ones(len(true_labels), dtype=bool)

        _true_labels = true_labels[mask]
        _logits = logits[mask]

        ap = average_precision_score(_true_labels, _logits)
        auc = roc_auc_score(_true_labels, _logits)

        _predictions = (_logits >= 0)

        [[tp, fn], [fp, tn]] = confusion_matrix(_true_labels, _predictions, labels=[1, 0])
        f1 = f1_score(_true_labels, _predictions)
        acc = accuracy_score(_true_labels, _predictions)
        bacc = balanced_accuracy_score(_true_labels, _predictions)
        recall = recall_score(_true_labels, _predictions)
        if tp + fp == 0:
            prec = 0
        else:
            prec = precision_score(_true_labels, _predictions)

        # jac = np.sum(true_labels & predictions) / np.sum(true_labels | predictions)
        jac = jaccard_score(_true_labels, _predictions)

        logs =  {'average_precision': ap, 'roc_auc': auc, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'f1': f1,
                'accuracy': acc, 'balanced_accuracy': bacc, 'recall': recall, 'precision': prec, 'jaccard': jac}
        logs = {prefix + key: value for key, value in logs.items()}
        return logs
    
    def _get_x_y(self, batch):

        return batch.to(self._device), batch.y.reshape(-1, 1).to(self._device)

    def train_one_epoch(self):
        """
        One epoch of training with the current settings
        """

        if self.train_loader is None:
            raise RuntimeError('Create Loaders before Training')
        
        if self.model is None:
            raise RuntimeError('Create Model before Training')
        
        if self.optimizer is None:
            raise RuntimeError('Create Optimizer before Training')
        
        if self.loss_function is None:
            raise RuntimeError('Create Loss Function before Training')

        cumulated_loss = torch.tensor(0, device=self._device, dtype=torch.float32)
        counter = 0
        epoch = self._current_epoch
        progress = self._show_progress or (self._verbose >= 3)
        loader = self.train_loader
        optimizer = self.optimizer
        loss_fn = self.loss_function
        model = self.model
        device = self._device

        if self._verbose >= 2:
            print(f'Training Epoch {epoch}'.ljust(20), end=': ' if (not progress) else '\n')

        model.to(device)
        model.train()
        start = time.perf_counter()

        with tqdm(total=len(loader), disable=not progress) as pbar:

            for batch in loader:

                x, y = self._get_x_y(batch=batch)

                # Skip broken batches
                # if torch.sum(y) == 0:
                #     pbar.update(1)
                #     continue

                optimizer.zero_grad()
                y_hat = model(x)
                loss = loss_fn(y_hat, y)

                if torch.isnan(loss):
                    print('NaN in y_hat: ', y_hat.isnan().any())
                    print('1 in y: ', (y == 1).sum())
                    print('nan in loss!')
                    continue

                loss.backward()
                optimizer.step()

                cumulated_loss += loss.detach()
                counter += 1
                pbar.update(1)
                pbar.set_description(f'Training Epoch {epoch} (mean loss = {cumulated_loss / counter:.3g}):')

        cumulated_loss = cumulated_loss.cpu().numpy()

        del batch

        if self._verbose >= 2:
            print(f'Mean Loss: {cumulated_loss / counter:.5g}'.ljust(70) + f'| {time.perf_counter() - start:.4g} s')

        if self._verbose == 1:
            print(f'T{epoch}'.ljust(4) + f' | {cumulated_loss / counter:.5g}'.ljust(10) + f'| {time.perf_counter() - start:.4g} s')

        return {'cumulated_loss': cumulated_loss,
                'mean_loss': cumulated_loss / counter}
    
    def validate(self, return_logits: bool = False):
        """
        Validate the model with the current settings
        """

        if self.valid_loader is None:
            raise RuntimeError('Create Loaders before Validation')
        
        if self.model is None:
            raise RuntimeError('Create Model before Validation')
        
        if self.loss_function is None:
            raise RuntimeError('Create Loss Function before Validation')

        cumulated_loss = torch.tensor(0, device=self._device, dtype=torch.float32)
        counter = 0
        epoch = self._current_epoch
        progress = self._show_progress or (self._verbose >= 3)
        loader = self.valid_loader
        loss_fn = self.loss_function
        model = self.model
        device = self._device
        labels, logits, entries = [], [], []

        model.to(device)
        model.eval()
        start = time.perf_counter()

        if self._verbose >= 2:
            print(f'Validating Epoch {epoch}'.ljust(20), end=': ' if not progress else '\n')

        with torch.no_grad():
            with tqdm(total=len(loader), disable=not progress) as pbar:
                for batch in loader:

                    x, y = self._get_x_y(batch=batch)

                    y_hat = model(x)
                    loss = loss_fn(y_hat, y)

                    if torch.isnan(loss):
                        print('NaN in y_hat: ', y_hat.isnan().any())
                        print('1 in y: ', (y == 1).sum())
                        print('nan in loss!')
                        continue    

                    cumulated_loss += loss.detach()
                    counter += 1
                    pbar.update(1)
                    pbar.set_description(f'Validating Epoch {epoch} (mean loss = {cumulated_loss / counter:.3g}):')

                    labels.append(y.squeeze())
                    logits.append(y_hat.squeeze().detach())
                    entries.append(batch.entry.detach())

        labels = torch.concatenate(labels).cpu().numpy()
        logits = torch.concatenate(logits).cpu().numpy()
        entries = torch.concatenate(entries).cpu().numpy()

        del batch

        cumulated_loss = cumulated_loss.cpu().numpy()

        metrics = self._metrics(labels, logits)

        if self._verbose >= 2:
            print(f'Mean Loss: {cumulated_loss / counter:.5g}'.ljust(30) + f' | Jaccard: {metrics["jaccard"]:.5g}'.ljust(20)
                  + f' | F1: {metrics["f1"]:.5g}'.ljust(20) + f'| {time.perf_counter() - start:.4g} s')
            
        if self._verbose == 1:
            print(f'V{epoch}'.ljust(4) + f' | {cumulated_loss / counter:.5g}'.ljust(10) + f'| {time.perf_counter() - start:.4g} s'.ljust(10) + f' | Jaccard: {metrics["jaccard"]:.5g}')

        if not return_logits:
            return {'cumulated_loss': cumulated_loss,
                    'mean_loss': cumulated_loss / counter,
                    **metrics}
        else:
            return {'cumulated_loss': cumulated_loss,
                    'mean_loss': cumulated_loss / counter,
                    **metrics}, {'logits': logits, 'labels': labels, 'entries': entries}

    def test(self, silent: bool = False):
        """
        Test the model with the current settings
        """

        if self.test_loader is None:
            raise RuntimeError('Create Loaders before Testing')
        
        if self.model is None:
            raise RuntimeError('Create Model before Testing')
        
        if self.loss_function is None:
            raise RuntimeError('Create Loss Function before Testing')

        cumulated_loss = torch.tensor(0, device=self._device, dtype=torch.float32)
        counter = 0
        progress = self._show_progress or (self._verbose >= 3)
        loader = self.test_loader
        loss_fn = self.loss_function
        model = self.model
        device = self._device
        labels, logits = [], []

        model.to(device)
        model.eval()
        start = time.perf_counter()

        if not silent and self._verbose >= 2:
            print('Testing'.ljust(20), end=': ' if not progress else '\n')

        with torch.no_grad():
            with tqdm(total=len(loader), disable=(not progress) or silent) as pbar:
                for batch in loader:

                    x, y = self._get_x_y(batch=batch)
                    y_hat = model(x)
                    loss = loss_fn(y_hat, y)

                    if torch.isnan(loss):
                        print('nan in loss!')
                        continue

                    cumulated_loss += loss.detach()
                    counter += 1
                    pbar.update(1)
                    pbar.set_description(f'Testing (mean loss = {cumulated_loss / counter:.3g}):')


                    labels.append(y.squeeze())
                    logits.append(y_hat.squeeze().detach())

        labels = torch.concatenate(labels).cpu().numpy()
        logits = torch.concatenate(logits).cpu().numpy()

        del batch

        cumulated_loss = cumulated_loss.cpu().numpy()

        metrics = self._metrics(labels, logits)
    
        if not silent and self._verbose >= 2:
            print(f'Mean Loss: {cumulated_loss / counter:.5g}'.ljust(30) + f' | Jaccard: {metrics["jaccard"]:.5g}'.ljust(20)
                  + f' | F1: {metrics["f1"]:.5g}'.ljust(20) + f'| {time.perf_counter() - start:.4g} s')
            
        if not silent and self._verbose == 1:
            print('T '.ljust(4) + f' | {cumulated_loss / counter:.5g}'.ljust(10) + f'| {time.perf_counter() - start:.4g} s'.ljust(10) + f' | Jaccard: {metrics["jaccard"]:.5g}')


        return {'cumulated_loss': cumulated_loss,
                'mean_loss': cumulated_loss / counter,
                **metrics}

    def train(self, fold_tag: str | None = None, save_best_logits: str | None = None):
        """
        Train the model with current settings
        """

        os.makedirs(self._logs_dir, exist_ok=True)
        already_existing = os.listdir(self._logs_dir)
        already_existing = [file.split('.')[0].replace('log_', '') for file in already_existing]
        already_existing = [int(file) for file in already_existing if file.isnumeric()]
        next_index = (max(already_existing) + 1) if already_existing else 0

        epoch_losses = []
        epoch_f1 = []
        best_logits = None

        best_validation_loss = np.inf
        best_validation_f1 = 0
        best_epoch = -1

        self.logs = {}
        start = time.perf_counter()
        self.model.to(self._device)

        if self._verbose >= 1:
            print('Starting a training loop.')

        # Loop
        for epoch in range(self._epochs):

            self._current_epoch = epoch + 1
            self.logs.update({epoch+1: self.settings})

            # Train
            train_logs = self.train_one_epoch()
            self.logs[epoch+1].update({'train': train_logs})

            # Validation
            if save_best_logits:
                valid_logs, logits = self.validate(return_logits=True)
            else:
                valid_logs = self.validate()
            self.logs[epoch+1].update({'valid': valid_logs})

            # Collect loss and F1 for patience
            current_validation_loss = valid_logs['mean_loss']
            epoch_losses.append(current_validation_loss)
            current_validation_f1 = valid_logs['f1']
            epoch_f1.append(current_validation_f1)

            if self.scheduler is not None:
                self.scheduler.step()

            # Check if new best vaidation result
            if (current_validation_loss * 1.001 < best_validation_loss or
                current_validation_f1 > best_validation_f1  * 1.001):

                best_epoch = epoch + 1
                if save_best_logits:
                    pd.DataFrame(logits).to_csv(save_best_logits, index=False)

                print(f'\t<Epoch {epoch+1} is currently the best>')

                if current_validation_loss < best_validation_loss:
                    best_validation_loss = current_validation_loss

                if valid_logs['f1'] > best_validation_f1:
                    best_validation_f1 = current_validation_f1

                # Testing the model
                test_logs = self.test(silent=True)
                test_logs.update({'At epoch': epoch + 1})

            # Save epoch logs as JSON
            with open(os.path.join(self._logs_dir, f'log_{next_index}.json'), 'w') as f:
                epoch_logs = {key: str(value) for key, value in ({'N epoch': self._current_epoch} | valid_logs | self.settings).items()}
                if fold_tag is not None:
                    epoch_logs.update({'fold_tag': str(fold_tag)})
                json.dump(epoch_logs, f)
            next_index += 1

            if (epoch + 1) - best_epoch >= self._patience:
                print(f'Early stop is triggered at epoch {epoch+1}')
                break

        # Testing is saved
        self.logs.update({'testing': self.settings | test_logs})
            

        if self._verbose >= 2:
            print(f'Best validation epoch = {best_epoch}')
            print('Testing at best val'.ljust(20) + f': Mean Loss: {test_logs["mean_loss"]:.5g}'.ljust(30) +
                    f' | Jaccard: {test_logs["jaccard"]:.5g}'.ljust(20) + f' | F1: {test_logs["f1"]}'.ljust(20)  + f'| {time.perf_counter() - start:.4g} s')

        if self._verbose >= 1:
            print('-----')
            print('TEST' + f' | {test_logs["mean_loss"]:.5g}'.ljust(10) + f'| {time.perf_counter() - start:.4g} s'.ljust(10) + f' | Jaccard: {test_logs["jaccard"]:.5g}')


        self._current_epoch = 0
        self.save_logs()
        self.cleanup_cuda(self.train_loader, self.valid_loader, self.test_loader, self.model, self.optimizer, self.scheduler)

        return self.logs
    
    def cleanup_cuda(self, *objs):
        for obj in objs:
            try:
                del obj
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    
    def save_logs(self):

        os.makedirs(self._logs_dir, exist_ok=True)
        already_existing = os.listdir(self._logs_dir)
        already_existing = [file.split('.')[0].replace('log_', '') for file in already_existing]
        already_existing = [int(file) for file in already_existing if file.isnumeric()]
        next_index = (max(already_existing) + 1) if already_existing else 0

        clean_logs = []
        for epoch, epoch_logs in self.logs.items():

            flattened_logs = {'N epoch': epoch if epoch != 'testing' else -1}

            for sub_key, sub_value in epoch_logs.items():
                if not isinstance(sub_value, dict):
                    flattened_logs.update({sub_key: sub_value})
                else:

                    if sub_key == 'train':
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            flattened_logs.update({f'{sub_key}_{sub_sub_key}': sub_sub_value})
                    else:
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            flattened_logs.update({sub_sub_key: sub_sub_value})

            clean_logs.append(flattened_logs)

        clean_logs = pd.DataFrame(clean_logs)
        clean_logs.to_csv(os.path.join(self._logs_dir, f'log_{next_index}.csv'))

    def inference(self):
        """
        Tool for inference. Inference is performed from the test set (loader)
        """

        progress = self._show_progress or (self._verbose >= 3)
        loader = self.test_loader
        model = self.model
        device = self._device
        logits = np.zeros(1)
        entries = [0]
        codes = [0]

        model.eval()
        model.to(device)
        start = time.perf_counter()

        if self._verbose >= 2:
            print('Inference'.ljust(70), end=': ' if not progress else '\n')

        with torch.no_grad():
            with tqdm(total=len(loader), disable=not progress) as pbar:
                for batch in loader:

                    x, _ = self._get_x_y(batch=batch)
                    y_hat = model(**x)
                
                    counter += 1
                    pbar.update(1)
                    logits = np.concatenate([logits, y_hat.squeeze().detach().numpy().astype(np.float32)])
                    entries.extend(batch['entries'])
                    codes.extend(batch['codes'])

        if self._verbose >= 2:
            print(f'| {time.perf_counter() - start:.4g} s')

        return {'logits': logits[1:],
                'entries': entries[1:],
                'codes': codes[1:]}
    
    def inference_per_protein(self,
                              path_to_protein_folder: str,
                              thereshold: float = 0.5,
                              from_saved: bool = True):
        """
        Tool for inference of a single protein
        """

        protein_name = path_to_protein_folder.split('/')[-1]
        protein_index = protein_name.split('_')[-1]

        if self._verbose >= 2:
            print(f'Inference for the protein: {protein_name}.')

        # Take the data
        residues_df = pd.read_csv(os.path.join(path_to_protein_folder, 'residues.csv'))
        residues_df = residues_df.dropna(subset=['X ALPHA', 'Y ALPHA', 'Z ALPHA', 'Substructure Name', 'Substructure Code', 'Binding']).copy()
        if self._verbose >= 3:
            print(f'Residues: {len(residues_df)}')

        # Make a batch out of protein residues
        self.constructor.register_protein(index=protein_index, group=protein_name.split('_')[0]) # samples
        raw_batch = self.constructor.get_samples()

        batch = {}
        for key in raw_batch[0]:
            if len(key) <= 3 and ('R' in key or 'P' in key):
                batch.update({key: torch.tensor([sample[key] for sample in raw_batch])})
            else:
                batch.update({key: [sample[key] for sample in raw_batch]})
        
        codes = residues_df['Substructure Code'].to_list()

        # Get predictions
        model = self.model
        model.eval()
        model.to(self._device)
        with torch.no_grad():

            x, y = self._get_x_y(batch=batch)
            logits, _ = model(**x)
                
        predictions = torch.sigmoid(logits).detach().numpy()
        binary_predictions = (predictions >= thereshold)
        positives = [codes[i] for i in np.nonzero(binary_predictions)]

        if self._verbose >= 3:
            print(f'Predicted as positives: {np.sum(binary_predictions)}; As negatives: {np.sum(1 - binary_predictions)}')

        return {'logits': logits,
                'binary': binary_predictions,
                'codes': codes,
                'id': protein_index,
                'positives': positives}



