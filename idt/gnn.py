from itertools import islice
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import ReLU, Linear, Sequential, Tanh, Sigmoid, Identity
from torch_geometric.nn import (
    GINConv,
    GCNConv,
    global_add_pool, 
    global_mean_pool, 
    GraphNorm,
    GATConv,
    SAGEConv,
)
from lightning import LightningModule

def gin_conv(in_channels, out_channels):
    nn = Sequential(Linear(in_channels, 2 * in_channels), GraphNorm(2 * in_channels), ReLU(), Linear(2 * in_channels, out_channels))
    return GINConv(nn)

def get_conv(conv_name, dim, alpha=0.5, heads=1):
    match conv_name:
        case "GCN":
            return GCNConv(dim, dim)
        case "GIN":
            return gin_conv(dim, dim)
        case "SAGE":
            return SAGEConv(dim, dim)
        case "GAT":
            return GATConv(dim, dim, heads=heads)
        case "DIR-GCN":
            return DirGCNConv(dim, dim, alpha)
        # case "DIR-GIN":
        #     return DirGINConv(dim, dim, alpha)
        case "DIR-SAGE":
            return DirSageConv(dim, dim, alpha)
        case "DIR-GAT":
            return DirGATConv(dim, dim, heads=heads, alpha=alpha)
        case _: 
            raise ValueError(f"Convolution type {conv_name} not supported")
        
def get_norm(conv_name, dim, norm=True):
    if norm:
        return GraphNorm(dim)
    return Identity()
    # match conv_name:
    #     case "GCN":
    #         return GCNConv(dim, dim), GraphNorm(dim)
    #     case "GIN":
    #         return gin_conv(dim, dim), GraphNorm(dim)
    #     case "SAGE":
    #         return SAGEConv(dim, dim), Identity()
    #     case "GAT":
    #         return GATConv(dim, dim, heads=heads), Identity()
    #     case "DIR-GCN":
    #         return DirGCNConv(dim, dim, alpha), GraphNorm(dim)
    #     # case "DIR-GIN":
    #     #     return DirGINConv(dim, dim, alpha), Identity()
    #     case "DIR-SAGE":
    #         return DirSageConv(dim, dim, alpha), Identity()
    #     case "DIR-GAT":
    #         return DirGATConv(dim, dim, heads=heads, alpha=alpha), Identity()
    #     case _: 
    #         raise ValueError(f"Convolution type {conv_name} not supported")

# TODO this is a bit hacked together not 100% sure on it all. seems to work
class DirGCNConv(nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv, self).__init__()
        # Separate linear transforms for each direction
        self.lin_src_to_dst = nn.Linear(input_dim, output_dim)
        self.lin_dst_to_src = nn.Linear(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # Unpack edge indices: row = source nodes, col = destination nodes.
        row, col = edge_index

        # --- Forward Direction (source -> destination) ---
        # Compute degrees using torch.bincount for target nodes
        deg = torch.bincount(col, minlength=num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        
        # Compute normalisation for each edge
        norm_forward = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Create a sparse adjacency matrix for the forward direction
        indices_forward = edge_index.to(x.device)
        values_forward = norm_forward.to(x.device)
        A_forward = torch.sparse_coo_tensor(indices_forward, values_forward, (num_nodes, num_nodes), device=x.device)
        
        # Aggregate messages
        out_forward = torch.sparse.mm(A_forward, x)

        # --- Backward Direction (destination -> source) ---
        # Reverse the edge direction
        row_rev, col_rev = col, row
        deg_rev = torch.bincount(col_rev, minlength=num_nodes).float()
        deg_inv_sqrt_rev = deg_rev.pow(-0.5)
        deg_inv_sqrt_rev[deg_inv_sqrt_rev == float('inf')] = 0.0
        
        norm_backward = deg_inv_sqrt_rev[row_rev] * deg_inv_sqrt_rev[col_rev]
        
        indices_backward = torch.stack([row_rev, col_rev], dim=0).to(x.device)
        values_backward = norm_backward.to(x.device)
        A_backward = torch.sparse_coo_tensor(indices_backward, values_backward, (num_nodes, num_nodes), device=x.device)
        
        out_backward = torch.sparse.mm(A_backward, x)
        
        # --- Combine the Two Directions ---
        out = self.alpha * self.lin_src_to_dst(out_forward) + (1 - self.alpha) * self.lin_dst_to_src(out_backward)
        return out


class DirSageConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirSageConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = SAGEConv(input_dim, output_dim, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = SAGEConv(input_dim, output_dim, flow="target_to_source", root_weight=False)
        self.lin_self = Linear(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index):
        return (
            self.lin_self(x)
            + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index)
            + self.alpha * self.conv_dst_to_src(x, edge_index)
        )


class DirGATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads, alpha):
        super(DirGATConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = GATConv(input_dim, output_dim, heads=heads)
        self.conv_dst_to_src = GATConv(input_dim, output_dim, heads=heads)
        self.alpha = alpha

    def forward(self, x, edge_index):
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)

        return (1 - self.alpha) * self.conv_src_to_dst(x, edge_index) + self.alpha * self.conv_dst_to_src(
            x, edge_index_t
        )


class GNN(LightningModule):
    def __init__(self, num_features, num_classes, layers, dim, conv="GCN", activation="ReLU", pool="mean", lr=1e-4, weight=None, norm=1):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.layers = layers
        self.dim = dim
        self.pool = pool
        self.lr = lr
        self.conv_name = conv
        self.norm = norm == 1

        self.embedding = Linear(num_features, dim)
        match activation:
            case "ReLU": self.act = ReLU()
            case "Tanh": self.act = Tanh()
            case "Sigmoid": self.act = Sigmoid()
            case _: raise ValueError(f"Unknown activation {activation}")
        
        self.norms = torch.nn.ModuleList()
        self.conv_layers = torch.nn.ModuleList()

        for _ in range(layers):
            conv_inst = get_conv(self.conv_name, dim)
            norm_inst = get_norm(self.conv_name, dim, norm=self.norm)
            self.conv_layers.append(conv_inst)
            self.norms.append(norm_inst)

        self.out = Sequential(
            Linear(dim, dim),
            self.act,
            Linear(dim, num_classes)
        )
        self.loss = torch.nn.NLLLoss(weight=weight)
        
        self.save_hyperparameters('num_features', 'num_classes', 'layers', 'dim', 'activation', 'pool', 'lr', 'weight', 'norm')

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        for conv, norm in zip(self.conv_layers, self.norms):
            x = x + norm(conv(x, edge_index))
            x = self.act(x)
        match self.pool:
            case "mean": x = global_mean_pool(x, batch)
            case "add": x = global_add_pool(x, batch)
            case _: raise ValueError(f"Unknown aggregation {self.pool}")
        x = self.out(x)
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(out, batch.y)
        acc = (out.argmax(dim=1) == batch.y).float().mean().item()
        f1_macro = f1_score(batch.y.cpu(), out.argmax(dim=1).cpu(), average='macro')
        self.log(f'{self.conv_name}_train_loss', loss, on_step=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f'{self.conv_name}_train_acc', acc, on_step=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f'{self.conv_name}_train_f1_macro', f1_macro, on_step=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(out, batch.y)
        acc = (out.argmax(dim=1) == batch.y).float().mean().item()
        f1_macro = f1_score(batch.y.cpu(), out.argmax(dim=1).cpu(), average='macro')
        self.log(f'{self.conv_name}_val_loss', loss, on_epoch=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f'{self.conv_name}_val_acc', acc, on_epoch=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f'{self.conv_name}_val_f1_macro', f1_macro, on_epoch=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        return loss
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
