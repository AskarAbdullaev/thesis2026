import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool


class EGNNLayer(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_dim=128,
                 dropout=0.0,
                 activation: nn.Module = nn.SiLU,
                 residual=True):
        super().__init__()

        
        self.residual = residual and (in_dim == out_dim)


        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + 1, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, 1),
        )

        self.dropout = dropout

    def forward(self, x, pos, edge_index):

        src, dst = edge_index

        rel = pos[src] - pos[dst]                         # (E, 3)
        dist2 = (rel ** 2).sum(dim=-1, keepdim=True)     # (E, 1)

        m_in = torch.cat([x[src], x[dst], dist2], dim=-1)
        m_ij = self.edge_mlp(m_in)

        # coordinate update
        coord_weight = self.coord_mlp(m_ij)              # (E, 1)
        coord_msg = rel * coord_weight                   # (E, 3)
        delta_pos = scatter(coord_msg, dst, dim=0, dim_size=pos.size(0), reduce="mean")
        pos = pos + delta_pos

        # feature update
        agg_msg = scatter(m_ij, dst, dim=0, dim_size=x.size(0), reduce="sum")
        x_new = self.node_mlp(torch.cat([x, agg_msg], dim=-1))

        if self.residual:
            x_new = x_new + x

        if self.dropout > 0:
            x_new = nn.functional.dropout(x_new, p=self.dropout, training=self.training)

        return x_new, pos


class GNNModel(nn.Module):

    residue_fixed_features = {
            'ALA': {'Letter': 'A', 'MW': 89, 'Charge': 0, 'Aliphatic': 1, 'Polar': 0, 'Width': 3.3, 'Length': 5.5, 'Aromatic': 0, 'Sulphur': 0, 'pKa': 6.11, 'N Atoms': 13, 'N Rot Bonds': 0, 'Volume': 88.6, 'Hydropathy': 1.8},
            'ARG': {'Letter': 'R', 'MW': 174, 'Charge': 1, 'Aliphatic': 0, 'Polar': 1, 'Width': 4.7, 'Length': 11.0, 'Aromatic': 0, 'Sulphur': 0, 'pKa': 10.76, 'N Atoms': 26, 'N Rot Bonds': 4, 'Volume': 173.4, 'Hydropathy': -4.5},
            'ASN': {'Letter': 'N', 'MW': 132, 'Charge': 0, 'Aliphatic': 0, 'Polar': 1, 'Width': 3.5, 'Length': 7.5, 'Aromatic': 0, 'Sulphur': 0, 'pKa': 5.43, 'N Atoms': 17, 'N Rot Bonds': 2, 'Volume': 114.1, 'Hydropathy': -3.5},
            'ASP': {'Letter': 'D', 'MW': 133, 'Charge': -1, 'Aliphatic': 0, 'Polar': 1, 'Width': 3.5, 'Length': 6.5, 'Aromatic': 0, 'Sulphur': 0, 'pKa': 2.98, 'N Atoms': 16, 'N Rot Bonds': 2, 'Volume': 111.1, 'Hydropathy': -3.5},
            'CYS': {'Letter': 'C', 'MW': 121, 'Charge': 0, 'Aliphatic': 0, 'Polar': 1, 'Width': 3.7, 'Length': 6.4, 'Aromatic': 0, 'Sulphur': 1, 'pKa': 5.15, 'N Atoms': 14, 'N Rot Bonds': 1, 'Volume': 108.5, 'Hydropathy': 2.5},
            'GLN': {'Letter': 'Q', 'MW': 146, 'Charge': 0, 'Aliphatic': 0, 'Polar': 1, 'Width': 3.5, 'Length': 9.0, 'Aromatic': 0, 'Sulphur': 0, 'pKa': 5.65, 'N Atoms': 20, 'N Rot Bonds': 3, 'Volume': 143.8, 'Hydropathy': -3.5},
            'GLU': {'Letter': 'E', 'MW': 147, 'Charge': -1, 'Aliphatic': 0, 'Polar': 1, 'Width': 3.5, 'Length': 8.0, 'Aromatic': 0, 'Sulphur': 0, 'pKa': 3.08, 'N Atoms': 19, 'N Rot Bonds': 3, 'Volume': 138.4, 'Hydropathy': -3.5},
            'GLY': {'Letter': 'G', 'MW': 75, 'Charge': 0, 'Aliphatic': 1, 'Polar': 0, 'Width': 3.5, 'Length': 3.9, 'Aromatic': 0, 'Sulphur': 0, 'pKa': 6.06, 'N Atoms': 10, 'N Rot Bonds': 0, 'Volume': 60.1, 'Hydropathy': -0.4},
            'HIS': {'Letter': 'H', 'MW': 155, 'Charge': 0.5, 'Aliphatic': 0, 'Polar': 1, 'Width': 6.1, 'Length': 8.5, 'Aromatic': 1, 'Sulphur': 0, 'pKa': 7.64, 'N Atoms': 20, 'N Rot Bonds': 2, 'Volume': 153.2, 'Hydropathy': -3.2},
            'ILE': {'Letter': 'I', 'MW': 131, 'Charge': 0, 'Aliphatic': 1, 'Polar': 0, 'Width': 5.0, 'Length': 8.5, 'Aromatic': 0, 'Sulphur': 0, 'pKa': 6.04, 'N Atoms': 22, 'N Rot Bonds': 2, 'Volume': 166.7, 'Hydropathy': 4.5},
            'LEU': {'Letter': 'L', 'MW': 131, 'Charge': 0, 'Aliphatic': 1, 'Polar': 0, 'Width': 5.0, 'Length': 8.5, 'Aromatic': 0, 'Sulphur': 0, 'pKa': 6.04, 'N Atoms': 22, 'N Rot Bonds': 0, 'Volume': 166.7, 'Hydropathy': 3.8},
            'LYS': {'Letter': 'K', 'MW': 146, 'Charge': 1, 'Aliphatic': 0, 'Polar': 1, 'Width': 3.5, 'Length': 11.3, 'Aromatic': 0, 'Sulphur': 0, 'pKa': 9.47, 'N Atoms': 24, 'N Rot Bonds': 4, 'Volume': 168.6, 'Hydropathy': -3.9},
            'MET': {'Letter': 'M', 'MW': 149, 'Charge': 0, 'Aliphatic': 1, 'Polar': 0, 'Width': 3.5, 'Length': 10.3, 'Aromatic': 0, 'Sulphur': 1, 'pKa': 5.71, 'N Atoms': 20, 'N Rot Bonds': 3, 'Volume': 162.9, 'Hydropathy': 1.9},
            'PHE': {'Letter': 'F', 'MW': 165, 'Charge': 0, 'Aliphatic': 0, 'Polar': 0, 'Width': 4.7, 'Length': 9.7, 'Aromatic': 1, 'Sulphur': 0, 'pKa': 5.76, 'N Atoms': 23, 'N Rot Bonds': 2, 'Volume': 189.9, 'Hydropathy': 2.8},
            'PRO': {'Letter': 'P', 'MW': 115, 'Charge': 0, 'Aliphatic': 1, 'Polar': 0, 'Width': 5.1, 'Length': 6.2, 'Aromatic': 0, 'Sulphur': 0, 'pKa': 6.3, 'N Atoms': 17, 'N Rot Bonds': 0, 'Volume': 112.7, 'Hydropathy': -1.6},
            'SER': {'Letter': 'S', 'MW': 105, 'Charge': 0, 'Aliphatic': 0, 'Polar': 1, 'Width': 3.5, 'Length': 6.1, 'Aromatic': 0, 'Sulphur': 0, 'pKa': 5.7, 'N Atoms': 14, 'N Rot Bonds': 1, 'Volume': 89.0, 'Hydropathy': -0.8},
            'THR': {'Letter': 'T', 'MW': 119, 'Charge': 0, 'Aliphatic': 0, 'Polar': 1, 'Width': 5.0, 'Length': 6.1, 'Aromatic': 0, 'Sulphur': 0, 'pKa': 5.6, 'N Atoms': 17, 'N Rot Bonds': 1, 'Volume': 116.1, 'Hydropathy': -0.7},
            'TRP': {'Letter': 'W', 'MW': 204, 'Charge': 0, 'Aliphatic': 0, 'Polar': 0, 'Width': 5.6, 'Length': 10.9, 'Aromatic': 1, 'Sulphur': 0, 'pKa': 5.88, 'N Atoms': 27, 'N Rot Bonds': 2, 'Volume': 227.8, 'Hydropathy': -0.9},
            'TYR': {'Letter': 'Y', 'MW': 181, 'Charge': 0, 'Aliphatic': 0, 'Polar': 1, 'Width': 4.7, 'Length': 10.4, 'Aromatic': 1, 'Sulphur': 0, 'pKa': 5.63, 'N Atoms': 24, 'N Rot Bonds': 2, 'Volume': 193.6, 'Hydropathy': -1.3},
            'VAL': {'Letter': 'V', 'MW': 117, 'Charge': 0, 'Aliphatic': 1, 'Polar': 0, 'Width': 5.0, 'Length': 7.0, 'Aromatic': 0, 'Sulphur': 0, 'pKa': 6.02, 'N Atoms': 19, 'N Rot Bonds': 1, 'Volume': 140.0, 'Hydropathy': 4.2}    
        }
    

    def __init__(self,
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
                 seed: int = 1234):
        
        super().__init__()
        self.model_name = 'GNN'


        embedding = []
        for i, res in enumerate(self.residue_fixed_features):

            template = np.zeros(20)
            template[i] = 1
            row = np.concatenate([np.array(list(self.residue_fixed_features[res].values())[1:], dtype=np.float32), template])
            embedding.append(torch.tensor(row, dtype=torch.float32))

        embedding =  torch.vstack(embedding)
        min_vals = embedding.min(dim=0).values
        max_vals = embedding.max(dim=0).values
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        embedding = (embedding - min_vals) / range_vals
        self.register_buffer("embedding", embedding)

        assert isinstance(encoder_hidden_dim, int) and encoder_hidden_dim > 0, f'encoder_hidden_dim must be positive int, not {type(encoder_hidden_dim)} ({encoder_hidden_dim})'
        assert isinstance(encoder_depth, int) and encoder_depth > 0, f'encoder_depths must be positive int, not {type(encoder_depth)} ({encoder_depth})'
    
        assert isinstance(dropout, float) and dropout >= 0, f'dropout must be non-negative float, not {type(dropout)} ({dropout})'

        assert isinstance(gnn_hidden_dim, int), f'gnn_hidden_dim must be int, not {type(gnn_hidden_dim)}'
        assert gnn_hidden_dim > 0, f'gnn_hidden_dim must positive, not {gnn_hidden_dim}'

        assert isinstance(gnn_depth, int), f'gnn_depth must be int, not {type(gnn_depth)}'
        assert gnn_depth > 0, f'gnn_depth must positive, not {gnn_depth}'

        assert isinstance(dense_dim, int), f'dense_dim must be int, not {type(dense_dim)}'
        assert dense_dim > 0, f'dense_dim must positive, not {dense_dim}'

        assert isinstance(dense_depth, int), f'dense_depth must be int, not {type(dense_depth)}'
        assert dense_depth > 0, f'dense_depth must positive, not {dense_depth}'

        afs = {'elu': nn.ELU, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'selu': nn.SELU, 'gelu': nn.GELU,
               'silu': nn.SiLU, 'mish': nn.Mish, 'leakyrelu': nn.LeakyReLU}
        assert isinstance(gnn_activation, str), f'gnn_activation must be str, not {type(gnn_activation)}'
        assert isinstance(dense_activation, str), f'dense_activation must be str, not {type(dense_activation)}'
        gnn_activation = gnn_activation.lower().strip()
        dense_activation = dense_activation.lower().strip()
        assert gnn_activation in afs, f'unknown gnn_activation: {gnn_activation}'
        assert dense_activation in afs, f'unknown dense_activation: {dense_activation}'

        self.geometric_dim = geometric_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_depth = encoder_depth

        self.gnn_hidden_dim = gnn_hidden_dim
        self.dense_dim = dense_dim
        self.gnn_depth = gnn_depth
        self.protein_dim = protein_dim
        self.dense_depth = dense_depth
        self.dense_activation = dense_activation
        self.gnn_activation = gnn_activation
        self.batch_norm = batch_norm
        self.seed = seed
        self.dropout = dropout

        # Encoder Part: from 33 + 5 residue embedding features to hidden encoder dim
        self.res_encoder = []
        for i in range(self.encoder_depth + 1):
            self.res_encoder.append(nn.Linear((self.embedding.shape[1] + self.geometric_dim) if i == 0 else self.encoder_hidden_dim, self.encoder_hidden_dim))
            if self.batch_norm:
                self.res_encoder.append(nn.BatchNorm1d(self.encoder_hidden_dim))
            self.res_encoder.append(afs[self.dense_activation]())
            self.res_encoder.append(nn.Dropout(self.dropout))
        self.encoder = nn.Sequential(*self.res_encoder)

        # GNN blocks
        self.gnn = nn.ModuleList()
        
        for i in range(self.gnn_depth):
            self.gnn.append(
                EGNNLayer(
                    in_dim=self.encoder_hidden_dim if i == 0 else self.gnn_hidden_dim,
                    out_dim=self.gnn_hidden_dim,
                    hidden_dim=self.gnn_hidden_dim, 
                    dropout=self.dropout,
                    activation=afs[self.gnn_activation])
            )

        # Dense Head
        self.dense_head = []
        for i in range(self.dense_depth + 1):
            self.dense_head.append(
                nn.Linear(
                    in_features=(self.gnn_hidden_dim + self.protein_dim) if i == 0 else self.dense_dim,
                    out_features=1 if i == self.dense_depth else self.dense_dim
                )
            )
            if i < self.dense_depth:
                if self.batch_norm:
                    self.dense_head.append(nn.BatchNorm1d(self.dense_dim))
                self.dense_head.append(afs[self.dense_activation]())
                if self.dropout > 0:
                    self.dense_head.append(nn.Dropout(self.dropout))

        self.dense_head = nn.Sequential(*self.dense_head)
        self.total_params = sum([np.prod(p.shape) for p in self.parameters()])

    def forward(self, x: Data):
        
        # Make embeddings
        geometric_features = x.geo_features
        embeddings_fixed = self.embedding[x.residue_idx]
        embeddings = torch.cat([embeddings_fixed, geometric_features], dim=-1)
        nodes = self.encoder(embeddings)

        # Apply encoder
        nodes = self.encoder(embeddings)

        # GNN
        coords = x.pos
        edges = x.edge_index
        central = x.center_mask

        for layer in self.gnn:
            nodes, coords = layer(nodes, coords, edges)

        central_nodes = nodes[central]
        central_batch = x.batch[central]
        pooled = global_mean_pool(central_nodes, central_batch)

        # Concatenate with protein features
        with_protein_data = torch.cat([pooled, x.protein_features], dim=-1)

        # Dense Head
        out = self.dense_head(with_protein_data)
        return out
