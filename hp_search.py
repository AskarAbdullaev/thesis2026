import os
import json

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from training import TrainingWrapper
from gnn import GNNModel
from itertools import product

import json
def collect_logs(dir: str = 'GNNLogs'):
    t = []
    for file in os.listdir(dir):

        if file.split('.')[-1] != 'json':
            continue

        try:
            with open(os.path.join(dir, file), 'r') as f:
                t.append(json.load(f))
        except:
            continue
    df = pd.DataFrame(t)
    return df

def subsample_logs(logs_df: pd.DataFrame, specification: dict):

    new_df = logs_df.copy().convert_dtypes()

    for key, values in specification.items():

        assert str(key) in new_df.columns, f'unknown columns: {key}'
        assert isinstance(values, list), f'values of key must be a list, not {type(values)}'

        new_df = new_df.loc[new_df[key].notna()]

        _type = type(values[0])

        new_df = new_df.loc[new_df[key].astype(_type).isin(values)]

    return new_df


def hp_search(trainer: TrainingWrapper,
                 encoder_hidden_dim: list[int] = [16, 256],
                 encoder_depth: list[int] = [1, 3],
                 gnn_hidden_dim: list[int] = [16, 256],
                 gnn_depth: list[int] = [1, 3],
                 dense_dim: list[int] = [16, 256],
                 dense_depth: list[int] = [1, 3],
                 gnn_activation: list[str] = ['relu'],
                 dense_activation: list[str] = ['relu'],
                 dropout: list[float] = [0.2],
                 batch_norm: list[bool] = [True],
                 lr: list[float] = [0.001],
                 step_size: list[int] = [1],
                 gamma: list[float] = [0.8],
                 pos_weight: list[int] = [10],
                 seed: int = 42):


    assert isinstance(trainer, TrainingWrapper), f'trainer must be of class TrainingWrapper, not {type(trainer)}'
    assert isinstance(seed, int) and seed > 0, f'seed must be positive integer, not {type(seed)} ({seed})'
    for param, param_name in zip([encoder_hidden_dim, encoder_depth, gnn_hidden_dim, gnn_depth, dense_dim, dense_depth, gnn_activation,
                                  dense_activation, dropout, batch_norm, lr, step_size, gamma, pos_weight],
                                 ['encoder_hidden_dim', 'encoder_depth', 'gnn_hidden_dim', 'gnn_depth', 'dense_dim', 'dense_depth', 'gnn_activation',
                                  'dense_activation', 'dropout', 'batch_norm', 'lr', 'step_size', 'gamma', 'pos_weight']):
        assert isinstance(param, list), f'parameter {param_name} must be a list, not {type(param)} ({param})'
        assert len(param) >= 1, f'parameter {param_name} cannot be an empty list'
        for value in param:

            if param_name in ['encoder_dim', 'encoder_depth', 'gnn_dim', 'gnn_depth', 'dense_dim', 'dense_depth', 'step_size', 'pos_weight']:
                assert isinstance(value, int) and value >= 1, f'{param_name} values must be positive integers, not {type(value)} ({value})'
            elif param_name in ['gnn_activation', 'dense_activation']:
                assert isinstance(value, str), f'{param_name} values must str, not {type(value)} ({value})'
            elif param_name in ['dropout', 'lr', 'gamma']:
                assert isinstance(value, float), f'{param_name} values must float, not {type(value)} ({value})'
            else:
                assert isinstance(value, bool), f'{param_name} values must bool, not {type(value)} ({value})'

    counter = 0

    combs = product(encoder_hidden_dim, encoder_depth, gnn_hidden_dim, gnn_depth, dense_dim, dense_depth,
                    gnn_activation, dense_activation, dropout, batch_norm, lr, step_size, gamma, pos_weight)
    for (_encoder_dim, _encoder_depth, _gnn_dim, _gnn_depth, _dense_dim, _dense_depth, _gnn_activation, _dense_activation,
         _dropout, _batch_norm, _lr, _step_size, _gamma, _pos_weight) in combs:

        counter += 1
        trainer.clear_logs()
        
        print(f'-------------------{counter}--------------------')
        params = GNNModel(encoder_hidden_dim=_encoder_dim,
                            encoder_depth=_encoder_depth,
                            gnn_hidden_dim=_gnn_dim,
                            gnn_depth=_gnn_depth,
                            dense_dim=_dense_dim,
                            dense_depth=_dense_depth).total_params
        if params < 10_000:
            batch_size = 2048
        elif params < 50_000:
            batch_size = 1024
        elif params < 100_000:
            batch_size = 512
        elif params < 500_000:
            batch_size = 256
        elif params < 1_000_000:
            batch_size = 128
        elif params < 2_500_000:
            batch_size = 128
        else:
            batch_size = 128
        
        trainer.set_settings(verbose=1, device='cuda', epochs=31, show_progress=False)
        trainer.initialize_loaders(train_loader_kwargs={'batch_size': batch_size, 'shuffle': True},
                                valid_loader_kwargs={'batch_size': batch_size, 'shuffle': True},
                                test_loader_kwargs={'batch_size': batch_size, 'shuffle': True})
        trainer.initialize_model(encoder_hidden_dim=_encoder_dim,
                                encoder_depth=_encoder_depth,
                                gnn_hidden_dim=_gnn_dim,
                                gnn_depth=_gnn_depth,
                                gnn_activation=_gnn_activation,
                                dense_dim=_dense_dim,
                                dense_depth=_dense_depth,
                                dense_activation=_dense_activation,
                                
                                dropout=_dropout,
                                batch_norm=_batch_norm,
                                seed=seed
                                )
        trainer.initialize_optimizer(lr=_lr)
        trainer.initialize_scheduler(step_size=_step_size, gamma=_gamma)
        trainer.initialize_loss_function(pos_weight=_pos_weight)

        try:
            trainer.train()
        except BaseException as e:
            print('ERROR: ', e)
            continue


def plot_broad_hp(df: pd.DataFrame,
                  target_column: str = 'jaccard',
                  figsize: tuple = (10, 10),
                  save_scatter: str | None = None,
                  save_heatmap: str | None = None):

    """
    Tool to plot the results of broad hp search
    """

    # Extract features to plot
    df['Depth'] = df['encoder_depth'] + '/' + df['gnn_depth'] + '/' + df['dense_depth']
    df['Width'] = df['encoder_hidden_dim'] + '/' + df['gnn_hidden_dim'] + '/' + df['dense_dim']
    df_complexity = df[['total_params', target_column]]
    df_complexity[target_column] = df_complexity[target_column].astype(float)
    df_complexity['total_params'] = df_complexity['total_params'].astype(int)

    # Use linear regression
    lm = LinearRegression()
    lm.fit(np.log(df_complexity['total_params'].values.reshape(-1, 1)), y=df_complexity[target_column])
    intercept = lm.intercept_
    slope = lm.coef_[0]

    # Plot the scatterplot of complexity
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(df_complexity, x='total_params', y=target_column, ax=ax)
    ax.plot([df_complexity['total_params'].min(), df_complexity['total_params'].max()],
            [slope * np.log(df_complexity['total_params'].min()) + intercept, np.log(df_complexity['total_params'].max()) * slope + intercept],
            c='orange', label=f'Fitted Log-Linear: {target_column} = {slope:.3g} * ln(P) + {intercept:.3g}')
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Performance')
    ax.set_title(f'Broad Hyperparameters Search, attribute: {target_column} by complexity')
    ax.set_xscale('log')
    ax.legend()
    plt.tight_layout()
    if save_scatter is not None:
        fig.savefig(save_scatter)
    plt.plot()

    # Plot the heatmap of scores
    df_small = df[['Depth', 'Width', target_column]]
    df_small[target_column] = df_small[target_column].astype(float)
    df_small = pd.pivot_table(df_small, values=target_column, index='Width', columns='Depth')
    display(df_small)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_small, cbar=True, ax=ax, cbar_kws={'shrink': 0.63})
    ax.set_xlabel('Depths')
    ax.set_ylabel('Widths')
    ax.set_title(f'Broad Hyperparameters Search, attribute: {target_column}')
    ax.set_aspect(1)
    plt.tight_layout()
    if save_heatmap is not None:
        fig.savefig(save_heatmap)
    plt.plot()


def analyse_hyperparameters(df: pd.DataFrame,
                            param_columns: list[str],
                            target_column: str = 'f1',
                            save: str = None) -> pd.DataFrame:
    """
    Allows to briefly analyse hyperparameters

    Args:
        df (pd.DataFrame): dataframe with hyperparameters
        param_columns (list[str]): list of parameters to groupby
        target_column (str, optional): target metric to visualize. Defaults to 'f1'.
        save (str, optional): path to save the plot. Defaults to None.

    Returns:
        pd.DataFrame (df of hyperparameters)
    """

    if save is not None:
        assert isinstance(save, str), f'save must be str, not {type(save)}'

    stats = []
    seen_params = set()
    best_run = -1
    best_target = -np.inf
    max_epochs = 0

    for i, row in df.iterrows():

        params =  tuple(row[param_columns].to_list())
        if params in seen_params:
            continue
        seen_params.add(params)
        
        run_dict = row[param_columns].to_dict()
        run_dict.update({'title': '/'.join(
            [f'{value}' for _, value in run_dict.items()])}
        )

        same_params_df = df.loc[np.prod([(df[p] == row[p]).values for p in param_columns], axis=0) & (df['N epoch'].astype(int) > 0)].sort_values('N epoch')

        targets  = same_params_df[target_column].astype(float).to_list()
        for i, target in enumerate(targets):
            if target == 0:
                targets[i] = ((targets[i-1] if i-1>=0 else targets[i]) + (targets[i+1] if i+1 < len(targets) else targets[i])) / 2
        run_dict.update({'test_losses': same_params_df.sort_values('N epoch')['mean_loss'].astype(float).to_list(),
                         'test_targets': targets,
                        'total_epochs': len(same_params_df),
                         'best_test_loss': min(same_params_df['mean_loss'].astype(float)),
                         'best_test_target': max(same_params_df[target_column].astype(float)),
                         'best_epoch': np.argmin(same_params_df['mean_loss'].astype(float)).flatten(),
                         'best_epoch_target': np.argmax(same_params_df[target_column].astype(float)).flatten()
                         })
        
        if float(run_dict['best_test_target']) > best_target:
            best_target = float(run_dict['best_test_target'])
            best_run = len(stats) - 1
        if len(run_dict['test_targets']) > max_epochs:
            max_epochs = len(run_dict['test_targets'])

        stats.append(run_dict)

    # Plot the runs
    rows = (int(len(stats) % 3 > 0) + (len(stats) // 3))
    fig, axs = plt.subplots(ncols=3, nrows=rows,
                            sharex=True, sharey=True,
                            figsize=(12, rows * 2.2))

    for i, run_dict in enumerate(stats):

        ax = axs[i//3][i%3]

        # Plot train losses and test losses
        ax.plot(run_dict['test_targets'], c='grey', label=f'Test {target_column}')

        # Plot dashed lines of best epoch / best test loss
        color = 'black' if i != best_run else 'red'
        ax.plot([run_dict['best_epoch_target'], run_dict['best_epoch_target']], [0, 1], c=color, linestyle='--', linewidth=0.5)
        ax.plot([0, max_epochs], [run_dict['best_test_target'], run_dict['best_test_target']], c=color, linestyle='--', linewidth=0.5)

        # Set limits
        ax.set_xlim([0, max_epochs])
        ax.set_ylim([0, best_target * 1.1])

        if i // 3 == rows - 1:
            ax.set_xlabel('Epochs')
        if i % 3 == 0:
            ax.set_ylabel(target_column)

        # Set title, legend and appropriate epoch ticks
        ax.legend()
        ax.set_title(run_dict['title'])
        ax.set_xticks(np.arange(0, max_epochs, 5))
        ax.set_xticklabels([str(tick+1) for tick in ax.get_xticks()])

    # Add suptitle and show the result
    fig.suptitle(f'Validation {target_column} per HP combination')
    plt.tight_layout()

    if save:
        fig.savefig(save, bbox_inches="tight", pad_inches=0)
    plt.show()

    stats = pd.DataFrame(stats)
    return stats