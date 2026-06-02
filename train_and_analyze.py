import os
from collections import Counter
from itertools import combinations

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns

from scipy.stats import kruskal
import scikit_posthocs as sp
from sklearn.linear_model import LinearRegression


from sklearn.metrics import (f1_score,
                             roc_auc_score,
                             average_precision_score,
                             accuracy_score,
                             balanced_accuracy_score,
                             precision_score,
                             recall_score,
                             jaccard_score)

from dataset import Mapper


def plot_cv_logs(cv_logs_path: str = 'cv_logs.csv',
                 save: str | None = None):
    """
    A tool to plot the trainng curve based on cross-validatiob logs
    """

    # Load, process and show the frame
    cv_logs = pd.read_csv(cv_logs_path,
                        usecols=[
                                'N epoch', 'valid_folds', 'train_mean_loss',
                                'mean_loss', 'average_precision', 'roc_auc',
                                'f1', 'accuracy', 'balanced_accuracy', 'recall',
                                'precision', 'jaccard'
                                ]
                        ).reset_index(drop=True)
    cv_logs = cv_logs.loc[cv_logs['N epoch'] != -1]
    cv_logs['fold'] = cv_logs['valid_folds'].apply(lambda x: 1 + (int(x[1]) // 2))
    cv_logs = cv_logs.drop(columns=['valid_folds'])

    # Collect data per epoch
    max_epoch = cv_logs['N epoch'].max()
    train_losses = [[] for _ in range(max_epoch)]
    valid_losses = [[] for _ in range(max_epoch)]
    jaccards = [[] for _ in range(max_epoch)]
    for fold in range(1, 6):
        fold_epochs = cv_logs.loc[cv_logs['fold'] == fold].sort_values(by='N epoch')
        train_losses_ = fold_epochs['train_mean_loss'].to_list()
        valid_losses_ = fold_epochs['mean_loss'].to_list()
        jaccards_ = fold_epochs['f1'].to_list()
        for i in range(max_epoch):
            train_losses[i].append(
                round(train_losses_[i], 3) if i < len(train_losses_) else None
            )
            valid_losses[i].append(
                round(valid_losses_[i], 3) if i < len(valid_losses_) else None
            )
            jaccards[i].append(
                round(jaccards_[i], 3) if i < len(jaccards_) else None
            )

    train_losses = np.array(train_losses).astype(float)
    valid_losses = np.array(valid_losses).astype(float)
    jaccards = np.array(jaccards).astype(float)

    # Plot data per epoch
    fig, ax = plt.subplots()

    ax.plot(np.mean(train_losses, axis=1, where=~np.isnan(train_losses)), c='black', alpha=0.7, label='Train Loss (avg)')
    ax.plot(np.mean(valid_losses, axis=1, where=~np.isnan(valid_losses)), c='orange', alpha=0.7, label='Validation Loss (avg)')
    ax.plot(np.mean(jaccards, axis=1, where=~np.isnan(jaccards)), c='blue', alpha=0.7, label='Validation F1 (avg)')

    ax.set_xticks(np.arange(0, max_epoch, 1))
    ax.set_xticklabels(np.arange(1, max_epoch+1, 1).astype(str))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric Per Sample')
    ax.set_title('Cross Validation Learning Curves')
    ax.grid(linestyle='-', alpha=0.5)
    ax.legend()

    if save is not None:
        fig.savefig(save)

    plt.show()

    return cv_logs


def all_scores(targets, predictions):

    binary = (predictions >= 0.5)
    targets = targets.astype(int)

    return {'f1': f1_score(targets, binary),
            'auc': roc_auc_score(targets, predictions),
            'ap': average_precision_score(targets, predictions),
            'acc': accuracy_score(targets, binary),
            'bacc': balanced_accuracy_score(targets, binary),
            'prec': precision_score(targets, binary),
            'rec': recall_score(targets, binary),
            'jac': jaccard_score(targets, binary),
            'n': len(targets)}


def analyse_logits(df: pd.DataFrame):
    """
    All-in-one tool to analyze bulk logits
    """

    os.makedirs('Metrics', exist_ok=True)

    if os.path.isfile('Metrics/cv_logits.csv'):
        logits_df = pd.read_csv('Metrics/cv_logits.csv')
    else:
        logits_df = df.copy()
        logits_df['predictions'] = 1 / (1 + np.exp(-logits_df['logits'].values))
        
        mapper = Mapper(pd.read_csv('Files/database_v0.csv', sep='@',
                                      usecols=['basic$scPDB ID', 'SCOPe', 'UniProt', 'EC', 'uniprot$Organism', 'html$Reign', 'mol2$Derived Ligand']))
        logits_df['id'] = logits_df['entries'].astype(int).apply(lambda x: u.CustomDataSet.entries_idx[x])
        logits_df['uniprot'] = logits_df['id'].apply(lambda x: mapper.to_uniprot(x))
        logits_df['ligand'] = logits_df['id'].apply(lambda x: mapper.to_ligand(x))
        logits_df['scope'] = logits_df['id'].apply(lambda x: mapper.to_scope(x))
        logits_df['scope_class'] = logits_df['scope'].apply(lambda x: x.split('.')[0] if isinstance(x, str) and '.' in x else '-')
        logits_df['scope_subclass'] = logits_df['scope'].apply(lambda x: '.'.join(x.split('.')[0:2]) if isinstance(x, str) and '.' in x else '-')
        logits_df['ec'] = logits_df['id'].apply(lambda x: mapper.to_ec(x))
        logits_df['ec_class'] = logits_df['ec'].apply(lambda x: x.split('.')[0] if isinstance(x, str) and '.' in x else '-')
        logits_df['ec_subclass'] = logits_df['ec'].apply(lambda x: '.'.join(x.split('.')[0:2]) if isinstance(x, str) and '.' in x else '-')
        logits_df['organism'] = logits_df['id'].apply(lambda x: mapper.to_organism(x)).fillna('-')
        logits_df['reign'] = logits_df['id'].apply(lambda x: mapper.to_reign(x)).fillna('-')
        logits_df.to_csv('Metrics/cv_logits.csv', index=False)

    
    # Overall metrics

    overall_metrics = [{'group': 'all', **all_scores(logits_df['labels'].values, logits_df['predictions'].values)}]

    # Bulk metrics per SCOPe

    if os.path.isfile('Metrics/scope_bulk.csv'):
        scope_bulk_metrics = pd.read_csv('Metrics/scope_bulk.csv')
    else:
        scope_bulk_metrics = {}
        for scope_class in tqdm(list(logits_df['scope_class'].unique()), desc='Metrics per SCOPe class..'):
            sub_df = logits_df[logits_df['scope_class'] == scope_class]
            scope_bulk_metrics.update({scope_class: all_scores(sub_df['labels'].values, sub_df['predictions'].values)})
        for scope_subclass in tqdm(list(logits_df['scope_subclass'].unique()), desc='Metrics per SCOPe subclass'):
            sub_df = logits_df[logits_df['scope_subclass'] == scope_subclass]
            scope_bulk_metrics.update({scope_subclass: all_scores(sub_df['labels'].values, sub_df['predictions'].values)})

        scope_bulk_metrics = pd.DataFrame(scope_bulk_metrics).transpose()
        scope_bulk_metrics.to_csv('Metrics/scope_bulk.csv')

    scope_bulk_metrics['group'] = scope_bulk_metrics.iloc[:, 0].apply(lambda x: f'SCOPe: {x}')
    overall_metrics.extend(scope_bulk_metrics.iloc[:, 1:].to_dict(orient='records'))

    # Bulk metrics per EC
    if os.path.isfile('Metrics/ec_bulk.csv'):
        ec_bulk_metrics = pd.read_csv('Metrics/ec_bulk.csv')
    else:
        ec_bulk_metrics = {}
        for ec_class in tqdm(list(logits_df['ec_class'].unique()), desc='Metrics per EC class..'):
            sub_df = logits_df[logits_df['ec_class'] == ec_class]
            ec_bulk_metrics.update({ec_class: all_scores(sub_df['labels'].values, sub_df['predictions'].values)})
        for ec_subclass in tqdm(list(logits_df['ec_subclass'].unique()), desc='Metrics per EC subclass...'):
            sub_df = logits_df[logits_df['ec_subclass'] == ec_subclass]
            ec_bulk_metrics.update({ec_subclass: all_scores(sub_df['labels'].values, sub_df['predictions'].values)})

        ec_bulk_metrics = pd.DataFrame(ec_bulk_metrics).transpose()
        ec_bulk_metrics.to_csv('Metrics/ec_bulk.csv')

    ec_bulk_metrics['group'] = ec_bulk_metrics.iloc[:, 0].apply(lambda x: f'EC: {x}')
    overall_metrics.extend(ec_bulk_metrics.iloc[:, 1:].to_dict(orient='records'))

    # Bulk metrics per Reign and Organism
    if os.path.isfile('Metrics/reign_bulk.csv'):
        reign_bulk_metrics = pd.read_csv('Metrics/reign_bulk.csv')
    else:
        reign_bulk_metrics = {}
        for reign in tqdm(list(logits_df['reign'].unique()), desc='Metrics per Reign..'):
            sub_df = logits_df[logits_df['reign'] == reign]
            reign_bulk_metrics.update({reign: all_scores(sub_df['labels'].values, sub_df['predictions'].values)})

        reign_bulk_metrics = pd.DataFrame(reign_bulk_metrics).transpose()
        reign_bulk_metrics.to_csv('Metrics/reign_bulk.csv')

    reign_bulk_metrics['group'] = reign_bulk_metrics.iloc[:, 0].apply(lambda x: f'Reign: {x}')
    overall_metrics.extend(reign_bulk_metrics.iloc[:, 1:].to_dict(orient='records'))

    if os.path.isfile('Metrics/organism_bulk.csv'):
        organism_bulk_metrics = pd.read_csv('Metrics/organism_bulk.csv')
    else:
        organism_bulk_metrics = {}
        for organism in tqdm(list(logits_df['organism'].unique()), desc='Metrics per Organism...s'):
            sub_df = logits_df[logits_df['organism'] == organism]
            organism_bulk_metrics.update({organism: all_scores(sub_df['labels'].values, sub_df['predictions'].values)})
        

        organism_bulk_metrics = pd.DataFrame(organism_bulk_metrics).transpose()
        organism_bulk_metrics.to_csv('Metrics/organism_bulk.csv')

    overall_metrics = pd.DataFrame(overall_metrics)
    overall_metrics['n'] = overall_metrics['n'].astype(int)
    overall_metrics = overall_metrics.loc[~overall_metrics['group'].str.contains('.', regex=False)]
    overall_metrics = overall_metrics.set_index('group')

    # Per entry
    if os.path.isfile('Metrics/per_entry.csv'):
        per_entry_metrics = pd.read_csv('Metrics/per_entry.csv')
    else:
        per_entry_metrics = {}
        unique_entries = list(logits_df['id'].unique())
        for e in tqdm(unique_entries, desc='Metrics per entry...'):
            sub_df = logits_df[logits_df['id'] == e]
            per_entry_metrics.update({e: all_scores(sub_df['labels'].values, sub_df['predictions'].values)})
            per_entry_metrics[e].update(sub_df[['scope', 'scope_class', 'scope_subclass', 'ec', 'ec_class', 'ec_subclass', 'organism', 'reign']].iloc[0].to_dict())
        per_entry_metrics = pd.DataFrame(per_entry_metrics).transpose()
        per_entry_metrics.to_csv('Metrics/per_entry.csv')


    # Per entry SCOPe
    if os.path.isfile('Metrics/scope_per_entry.csv'):
        scope_per_entry_metrics = pd.read_csv('Metrics/scope_per_entry.csv')
    else:
        scope_per_entry_metrics = {}
        for scope_class in tqdm(list(per_entry_metrics['scope_class'].unique()), desc='Metrics per SCOPe class..'):
            sub_df = per_entry_metrics[per_entry_metrics['scope_class'] == scope_class]
            n_res = int(sub_df['n'].sum())
            n = len(sub_df)
            mean = sub_df[['f1', 'auc', 'ap', 'acc', 'bacc', 'prec', 'rec', 'jac']].mean().to_dict()
            std = sub_df[['f1', 'auc', 'ap', 'acc', 'bacc', 'prec', 'rec', 'jac']].std().to_dict()
            scope_per_entry_metrics.update(
                {scope_class: {'n_res': n_res, 'n': n,
                               **{(k + '_mean'): v for k, v in mean.items()},
                               **{(k + '_std'): v for k, v in std.items()}}}
                )
        for scope_subclass in tqdm(list(per_entry_metrics['scope_subclass'].unique()), desc='Metrics per SCOPe subclass'):
            sub_df = per_entry_metrics[per_entry_metrics['scope_subclass'] == scope_subclass]
            n_res = int(sub_df['n'].sum())
            n = len(sub_df)
            mean = sub_df[['f1', 'auc', 'ap', 'acc', 'bacc', 'prec', 'rec', 'jac']].mean().to_dict()
            std = sub_df[['f1', 'auc', 'ap', 'acc', 'bacc', 'prec', 'rec', 'jac']].std().to_dict()
            scope_per_entry_metrics.update(
                {scope_subclass: {'n_res': n_res, 'n': n,
                               **{(k + '_mean'): v for k, v in mean.items()},
                               **{(k + '_std'): v for k, v in std.items()}}}
                )

        scope_per_entry_metrics = pd.DataFrame(scope_per_entry_metrics).transpose()
        scope_per_entry_metrics.to_csv('Metrics/scope_per_entry.csv')


    # Per entry EC
    if os.path.isfile('Metrics/ec_per_entry.csv'):
        ec_per_entry_metrics = pd.read_csv('Metrics/ec_per_entry.csv')
    else:
        ec_per_entry_metrics = {}
        for ec_class in tqdm(list(per_entry_metrics['ec_class'].unique()), desc='Metrics per EC class..'):
            sub_df = per_entry_metrics[per_entry_metrics['ec_class'] == ec_class]
            n_res = int(sub_df['n'].sum())
            n = len(sub_df)
            mean = sub_df[['f1', 'auc', 'ap', 'acc', 'bacc', 'prec', 'rec', 'jac']].mean().to_dict()
            std = sub_df[['f1', 'auc', 'ap', 'acc', 'bacc', 'prec', 'rec', 'jac']].std().to_dict()
            ec_per_entry_metrics.update(
                {ec_class: {'n_res': n_res, 'n': n,
                               **{(k + '_mean'): v for k, v in mean.items()},
                               **{(k + '_std'): v for k, v in std.items()}}}
                )
        for ec_subclass in tqdm(list(per_entry_metrics['ec_subclass'].unique()), desc='Metrics per EC subclass'):
            sub_df = per_entry_metrics[per_entry_metrics['ec_subclass'] == ec_subclass]
            n_res = int(sub_df['n'].sum())
            n = len(sub_df)
            mean = sub_df[['f1', 'auc', 'ap', 'acc', 'bacc', 'prec', 'rec', 'jac']].mean().to_dict()
            std = sub_df[['f1', 'auc', 'ap', 'acc', 'bacc', 'prec', 'rec', 'jac']].std().to_dict()
            ec_per_entry_metrics.update(
                {ec_subclass: {'n_res': n_res, 'n': n,
                               **{(k + '_mean'): v for k, v in mean.items()},
                               **{(k + '_std'): v for k, v in std.items()}}}
                )

        ec_per_entry_metrics = pd.DataFrame(ec_per_entry_metrics).transpose()
        ec_per_entry_metrics.to_csv('Metrics/ec_per_entry.csv')



    # Per entry Organism and Reign
    if os.path.isfile('Metrics/reign_per_entry.csv'):
        reign_per_entry_metrics = pd.read_csv('Metrics/reign_per_entry.csv')
    else:
        reign_per_entry_metrics = {}
        for reign in tqdm(list(per_entry_metrics['reign'].unique()), desc='Metrics per Reign..'):
            sub_df = per_entry_metrics[per_entry_metrics['reign'] == reign]
            n_res = int(sub_df['n'].sum())
            n = len(sub_df)
            mean = sub_df[['f1', 'auc', 'ap', 'acc', 'bacc', 'prec', 'rec', 'jac']].mean().to_dict()
            std = sub_df[['f1', 'auc', 'ap', 'acc', 'bacc', 'prec', 'rec', 'jac']].std().to_dict()
            reign_per_entry_metrics.update(
                {reign: {'n_res': n_res, 'n': n,
                               **{(k + '_mean'): v for k, v in mean.items()},
                               **{(k + '_std'): v for k, v in std.items()}}}
                )

        reign_per_entry_metrics = pd.DataFrame(reign_per_entry_metrics).transpose()
        reign_per_entry_metrics.to_csv('Metrics/reign_per_entry.csv')


    if os.path.isfile('Metrics/organism_per_entry.csv'):
        organism_per_entry_metrics = pd.read_csv('Metrics/organism_per_entry.csv')
    else:
        organism_per_entry_metrics = {}
        for organism in tqdm(list(per_entry_metrics['organism'].unique()), desc='Metrics per Organism'):
            sub_df = per_entry_metrics[per_entry_metrics['organism'] == organism]
            n_res = int(sub_df['n'].sum())
            n = len(sub_df)
            mean = sub_df[['f1', 'auc', 'ap', 'acc', 'bacc', 'prec', 'rec', 'jac']].mean().to_dict()
            std = sub_df[['f1', 'auc', 'ap', 'acc', 'bacc', 'prec', 'rec', 'jac']].std().to_dict()
            organism_per_entry_metrics.update(
                {organism: {'n_res': n_res, 'n': n,
                               **{(k + '_mean'): v for k, v in mean.items()},
                               **{(k + '_std'): v for k, v in std.items()}}}
                )

        organism_per_entry_metrics = pd.DataFrame(organism_per_entry_metrics).transpose()
        organism_per_entry_metrics.to_csv('Metrics/organism_per_entry.csv')

    return overall_metrics


def read_per_entry_metrics(per_entry_path: str = 'Metrics/per_entry.csv'):
    """
    Tool to create a complete dataframe with classes and metrics per protein
    """

    per_entry_df = pd.read_csv(per_entry_path)
    per_entry_df['size_class'] = (per_entry_df['n'].values / 100).astype(int)
    mapper = Mapper(pd.read_csv('Files/database_v0.csv', sep='@',
                                    usecols=['basic$scPDB ID', 'SCOPe', 'UniProt', 'EC', 'uniprot$Organism', 'html$Reign', 'mol2$Derived Ligand']))
    per_entry_df['uniprot'] = per_entry_df['id'].apply(lambda x: mapper.to_uniprot(x))
    per_entry_df['ligand'] = per_entry_df['id'].apply(lambda x: mapper.to_ligand(x))

    per_entry_df['size_class'] = per_entry_df['size_class'].map(
        {0: '0-100', 1: '100-200', 2: '200-300', 3: '300-400', 4: '400-500',
        5: '500-700', 6: '500-700', 7: '700-1000', 8: '700-1000', 9: '700-1000',
        10: '>1000', 11: '>1000', 12: '>1000', 13: '>1000', 14: '>1000', 15: '>1000'}
    )
    per_entry_df['scope_class_desc'] = per_entry_df['scope_class'].map(
        {'-': 'Unknown',
        'a': 'Alpha-Helices',
        'b': 'Beta-Sheets',
        'c': 'Helices + Sheets\n(interleaved)',
        'd': 'Helices + Sheets\n(segregated)',
        'e': 'Multi-Domain',
        'f': 'Membrane Proteins',
        'g': 'Small \nUnconventional',
        'h': 'Coiled Coils',
        'l': 'Artifacts'}
    )
    per_entry_df['ec_class_desc'] = per_entry_df['ec_class'].map(
        {'-': 'Unknown',
        '1': 'Oxidoreductases',
        '2': 'Transferases',
        '3': 'Hydrolases',
        '4': 'Lyases',
        '5': 'Isomerases',
        '6': 'Ligases',
        '7': 'Translocases'}
    )

    per_entry_df['reign_desc'] = per_entry_df['reign'].apply(
        lambda x: 'Unknown' if x == '-' else x
    )

    return per_entry_df


def plot_stats(df: pd.DataFrame,
               metric: str = 'f1',
               group: str ='scope_class',
               limit: int = 10,
               rot: int = 90,
               title: str | None = None,
               save: str | None = None):
    """
    A tool that plots a combination of box and violin plots per class (group)
    showing the values (metric)

    limit: the minimum number of proteins in a class to be shown
    rot: the rotation of labels
    """

    enough_mask = (df[['n', group]].groupby(group).count() >= limit)
    enough_mask = list(enough_mask.loc[enough_mask['n']].index)
    enough_mask = list(filter(lambda x: x not in ['-', 'Unknown', np.nan], enough_mask))
    
    group_to_title = {'scope_class': 'SCOPe Class', 'ec_class': 'EC Class',
                    'scope_class_desc': 'SCOPe Class', 'ec_class_desc': 'EC Class',
                    'size_class': 'Protein Size', 'organism': 'Source Organism',
                    'reign': 'Source Organism Reign', 'reign_desc': 'Source Organism Reign',
                    'ligand': 'Ligand'}
    ax = df.loc[df[group].isin(enough_mask), [metric, group]].plot.box(
        by=group, ylabel=metric.upper(), xlabel=group_to_title[group], zorder=1, notch=False,
        showcaps=False, showbox=False, showfliers=False, whis=0.0
        )
    ax = ax[metric]
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rot)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'{metric.upper()} score by {group_to_title[group]}')
    ax.set_ylim((-0.05, df[metric].max() * 1.2))

    from collections import Counter
    counts = Counter(df[group].to_list())
    counts = [counts[x] for x in enough_mask]
    violin_groups = []
    for c, count in zip(enough_mask, counts):
        text = str(int(count)) 
        x = enough_mask.index(c) + 1 - 0.05 * len(str(count))
        y = df[metric].max() * 1.1
        ax.annotate(text, (x, y), fontsize=8)

        ys = df.loc[df[group] == c, metric].values
        violin_groups.append(ys)

    parts = ax.violinplot(
        violin_groups,
        list(range(1, len(violin_groups) + 1)),
        showmeans=False,
        showmedians=False,
        showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)

    plt.tight_layout()

    if save is not None:
        ax.get_figure().savefig(save)
    
    return ax


def stats_analysis(df_: pd.DataFrame,
                   group: str = 'scope_class_desc',
                   metric: str = 'f1'):
    """
    A function that takes a dataframe of metrics and 
    performs group statistics according to the provided grouping
    along the provided metric
    """
    
    result = {}
    df = df_.copy()
    df = df[[group, metric]].dropna()

    # Extract group descriptors (sizes and metrics)
    groups = []
    group_sizes = {}
    group_metrics = {}

    for c in df[group].unique():
        sub_df = (df.loc[df[group] == c, metric])
        groups.append(sub_df.values)
        group_sizes.update({c: len(sub_df)})
        group_metrics.update({c: sub_df.mean()})

    result.update({'group_size': group_sizes, 'group_metric': group_metrics})

    # Get group names
    group_labels = df[group].unique()
    result.update({'n': len(group_labels)})

    # 1. Kruskal–Wallis test
    H, p = kruskal(*groups)
    result.update({'kruskal': {'p': float(p), 'H': float(H)}})

    # 2. Dunn's post-hoc test
    if p < 0.05:

        dunn = sp.posthoc_dunn(
            df,
            val_col=metric,
            group_col=group,
            p_adjust='holm'
        )
        result.update({'dunn': dunn})

    # 3. Effect size: Cliff's Delta
    def cliffs_delta(x, y):
        """
        Computes Cliff's delta effect size.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        n = len(x) * len(y)
        gt = sum(i > j for i in x for j in y)
        lt = sum(i < j for i in x for j in y)

        return (gt - lt) / n

    if p < 0.05:

        result.update({'cliff': {}})

        for g1, g2 in combinations(group_labels, 2):
            x = df.loc[df[group] == g1, 'f1'].values
            y = df.loc[df[group] == g2, 'f1'].values

            delta = cliffs_delta(x, y)
            # print(f"{g1} vs {g2}: delta = {delta:.3f}")
            result['cliff'].update({(g1, g2): delta})

    return result


def dunns_heatmap(stats: dict,
                  title: str = '',
                  save: str | None = None):
    """
    A function that takes a stats dict from stata_analysis function and plots a dunn's matrix as
    a seaborn heatmap
    """

    dunn_matrix = stats['dunn'].copy()
    mask = np.eye(len(dunn_matrix)).astype(bool)
    for i in range(len(mask)):
        for j in range(len(mask)):
            if j > i:
                mask[i, j] = 1
    
    annots = dunn_matrix.values
    annots = np.where(annots > 0.01, np.round(annots, 3), np.round(np.log10(annots)))
    mask_large = (annots > 0.05)
    mask_small = (annots < 0.05)

    dunn_matrix.iloc[:, :] = annots
    ax = sns.heatmap(-dunn_matrix.iloc[1:,:-1], annot=annots[1:,:-1], fmt='.3g', cmap='Blues', mask=mask[1:,:-1] | mask_large[1:,:-1], vmax=np.max(-annots), vmin=0.05, cbar=False)
    sns.heatmap(dunn_matrix.iloc[1:,:-1], annot=annots[1:,:-1], fmt='.3g', cmap='Reds', mask=mask[1:,:-1] | mask_small[1:,:-1], vmax=np.max(annots), vmin=0.05, cbar=False, ax=ax, linewidths=0.1)
    # sns.heatmap(dunn_matrix, center=0.05, annot=annots, fmt='.3g', cmap='coolwarm', mask=mask | mask_small, vmax=np.max(annots), vmin=np.min(annots))
    ax.set_aspect(1)
    ax.get_figure().set_size_inches(len(annots) * 0.98 ** len(annots), len(annots) * 0.98 ** len(annots))
    ax.set_xticks(np.arange(len(annots)-1) + 0.5)
    ax.set_xticklabels(dunn_matrix.columns[:-1], rotation=90)
    ax.set_yticks(np.arange(len(annots)-1) + 0.5)
    ax.set_yticklabels(dunn_matrix.columns[1:], rotation=0)
    ax.set_title(title)
    plt.tight_layout()

    if save is not None:
        ax.get_figure().savefig(save)
    plt.show()


def effect_size_label(delta_abs: float) -> str:
    """
    Textual description of effect size magnitude
    """
    
    if delta_abs >= 0.60:
        return "very large"
    elif delta_abs >= 0.47:
        return "large"
    elif delta_abs >= 0.33:
        return "medium"
    elif delta_abs >= 0.15:
        return "small"
    else:
        return "negligible"


def dunn_to_long(dunn_matrix: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:

    """
    Unravaling Dunn matrix to long (pairs) format
    """
    labels = list(dunn_matrix.index)
    rows = []

    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if j <= i:
                continue

            p = dunn_matrix.loc[a, b]
            if pd.isna(p):
                continue

            rows.append({
                "A": a,
                "B": b,
                "p_value": float(p),
                "significant": float(p) < alpha
            })

    return pd.DataFrame(rows)


def cliff_to_long(deltas: dict) -> pd.DataFrame:
    """
    Making a more convebient view of cliff's deltas
    """
    rows = []

    for (a, b), d in deltas.items():
        rows.append({"A": a, "B": b, "Delta": float(d)})

    return pd.DataFrame(rows)


def merge_dunn_cliff(stats: dict, alpha: float = 0.05) -> pd.DataFrame:
    """
    A tool to combine Dunn and Cliff (take significance into account)
    """

    dunn_long = dunn_to_long(stats["dunn"], alpha=alpha)
    cliff_long = cliff_to_long(stats["cliff"])

    # Normalize pairs so that (a,b) is like (b,a)
    def canonical_pair(a, b):
        return tuple(sorted([a, b]))

    dunn_long["pair"] = dunn_long.apply(lambda r: canonical_pair(r["A"], r["B"]), axis=1)
    cliff_long["pair"] = cliff_long.apply(lambda r: canonical_pair(r["A"], r["B"]), axis=1)

    # Reorient delta (because it os order-sensitive)
    delta_map = {}
    for (a, b), d in stats["cliff"].items():
        delta_map[(a, b)] = float(d)
        delta_map[(b, a)] = -float(d)

    # Making a single dataframe
    rows = []
    for _, r in dunn_long.iterrows():
        a, b = r["A"], r["B"]
        d = delta_map.get((a, b), np.nan)

        rows.append({
            "A": a,
            "B": b,
            "p_value": r["p_value"],
            "significant": r["significant"],
            "Delta": d,
            "abs_delta": abs(d),
            "amplitude": effect_size_label(abs(d)),
        })

    return pd.DataFrame(rows)


def latent_rating(pair_df: pd.DataFrame,
                  labels: list[str],
                  only_significant: bool = True,
                  min_abs_delta: float = 0.15):
    """
    Using LinearRegression (from sklearn) to find the latent factors based on
    a series of pairwise comparisons.
    """
    df = pair_df.copy()

    if only_significant:
        df = df[df["significant"]]

    # Mininum significant delta to consider:
    df = df[df["abs_delta"] >= min_abs_delta]

    if len(df) == 0:
        return {}, np.nan

    label_to_idx = {g: i for i, g in enumerate(labels)}

    # Populate X and y for regression task
    X = []
    y = []

    for _, r in df.iterrows():
        a, b, d = r["A"], r["B"], r["Delta"]

        row = np.zeros(len(labels))
        row[label_to_idx[a]] = 1
        row[label_to_idx[b]] = -1

        X.append(row)
        y.append(d)

    X = np.vstack(X)
    y = np.asarray(y)

    # Regression
    lm = LinearRegression(fit_intercept=False)
    lm.fit(X, y)

    # Get factors for rating
    pred = lm.predict(X)
    corr = np.corrcoef(pred, y)[0, 1] if len(y) > 1 else np.nan

    scores = lm.coef_
    scores = scores - scores.mean()

    rating = {
        g: round(float(s), 3)
        for g, s in zip(labels, scores)
    }
    rating = dict(sorted(rating.items(), key=lambda x: x[1], reverse=True))

    return rating, corr


def count_pairwise_wins(pair_df: pd.DataFrame,
                        only_significant: bool = True,
                        min_abs_delta: float = 0.15):
    """
    A tool to return Win/Lose counts: just the number of
    times an object A is larger than some other objects minus
    the reversed situation
    """
    df = pair_df.copy()

    if only_significant:
        df = df[df["significant"]]

    df = df[df["abs_delta"] >= min_abs_delta]

    counts = Counter()

    for _, r in df.iterrows():
        a, b, d = r["A"], r["B"], r["Delta"]

        if d > 0:
            counts[a] += 1
            counts[b] -= 1
        elif d < 0:
            counts[b] += 1
            counts[a] -= 1

    return counts


def find_cycles(pair_df: pd.DataFrame,
                labels: list[str],
                only_significant: bool = True,
                min_abs_delta: float = 0.15):
    
    """
    Finds cycles like A > B, B > C, C > A.
    It is needed to ensure the sanity of linear regression approach
    """
    df = pair_df.copy()

    if only_significant:
        df = df[df["significant"]]

    df = df[df["abs_delta"] >= min_abs_delta]

    preference = {}

    for _, r in df.iterrows():
        a, b, d = r["A"], r["B"], r["Delta"]

        if d > 0:
            preference[(a, b)] = True
        elif d < 0:
            preference[(b, a)] = True

    cycles = []

    for a, b, c in combinations(labels, 3):
        if preference.get((a, b), False) and preference.get((b, c), False) and preference.get((c, a), False):
            cycles.append((a, b, c))

        if preference.get((a, c), False) and preference.get((c, b), False) and preference.get((b, a), False):
            cycles.append((a, c, b))

    return cycles


def analyze_pairs(stats: dict,
                  alpha: float = 0.05,
                  min_abs_delta: float = 0.15):
    
    """
    A wrapper function to get Dunn, Cliff, Win/Lose and rating from pairs
    """
    labels = list(stats["dunn"].index)

    pair_df = merge_dunn_cliff(stats, alpha=alpha)

    counts = count_pairwise_wins(
        pair_df,
        only_significant=True,
        min_abs_delta=min_abs_delta
    )

    rating, corr = latent_rating(
        pair_df,
        labels=labels,
        only_significant=True,
        min_abs_delta=min_abs_delta
    )

    cycles = find_cycles(
        pair_df,
        labels=labels,
        only_significant=True,
        min_abs_delta=min_abs_delta
    )

    return {
        "pairs": pair_df,
        "counts": counts,
        "rating": rating,
        "rating_corr": corr,
        "cycles": cycles,
    }
