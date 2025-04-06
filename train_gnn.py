from argparse import ArgumentParser
from joblib import Parallel, delayed
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch

from idt.data import data
from idt.gnn import GNN

def train_fold(args, fold, conv_type, device):
    num_features, num_classes, train_loader, val_loader, train_val_batch, _ = data(args.dataset, args.kfold, fold, seed=42)

    bincount = torch.bincount(train_val_batch.y, minlength=2)
    weight = len(train_val_batch) / (2 * bincount.float())

    # Initialise the model with the specified convolution type and hyperparameters.
    model = GNN(
        num_features=num_features,
        num_classes=num_classes,
        layers=args.layers,
        dim=args.dim,
        activation=args.activation,
        conv=conv_type,
        pool=args.pooling,
        lr=args.lr,
        weight=weight,
        norm=args.norm
    )
    
    # Set up callbacks: EarlyStopping and ModelCheckpoint (which saves the best validation loss)
    monitor_metric = f"{conv_type}_val_loss"
    early_stop_callback = EarlyStopping(monitor=monitor_metric, patience=1000, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor=monitor_metric, mode="min", save_top_k=1,
                                          filename=f"fold{fold}" + "-{epoch}-{"+monitor_metric+":.4f}")
    
    trainer = Trainer(
        max_steps=args.max_steps,
        logger=None,  # Logging for each fold is handled by the main run
        devices=[device],
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=1
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Retrieve the best validation loss from this fold
    best_val_loss = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score is not None else float('inf')
    return best_val_loss

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='EMLC0', help='Name of the dataset')
    parser.add_argument('--activation', type=str, default='ReLU', choices=['ReLU', 'Tanh', 'Sigmoid'], help='Activation function')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'add'], help='Pooling function')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--kfold', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of node embeddings')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum training steps per fold')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices to use')
    parser.add_argument('--conv', type=str, default='GCN', help='', choices=['GCN', 'GIN', 'SAGE', 'GAT', 'DIR-GCN', 'DIR-GIN', 'DIR-SAGE', 'DIR-GAT'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--norm', type=int, default=1, help='', choices=[0,1])
    args = parser.parse_args()

    
    # Initialise a single wandb run for the current hyperparameter configuration
    wandb_logger = WandbLogger(project="gnn_kfold_sweep", config=vars(args), group=f'args.dataset')
    
    conv_type = args.conv
    fold_val_losses = []

    torch.set_float32_matmul_precision('high')
    
    #Run k‐fold cross validation
    for fold in range(args.kfold):
        print(f"Training fold {fold + 1}/{args.kfold} for {conv_type}")
        best_loss = train_fold(args, fold, conv_type, device=0)
        print(f"Fold {fold + 1} best validation loss: {best_loss:.4f}")
        fold_val_losses.append(best_loss)
    
    # DOES NOT WORK 
    # fold_val_losses = Parallel(n_jobs=args.kfold)(
    #         delayed(train_fold)(args, fold, conv_type, device=0) 
    #         for fold in range(args.kfold)
    #     )
    
    # Compute the average best epoch validation loss across folds
    avg_val_loss = sum(fold_val_losses) / len(fold_val_losses)
    print(f"Average best validation loss over {args.kfold} folds: {avg_val_loss:.4f}")
    
    # Log the average metric to wandb; this is the quantity to be minimised during sweeps.
    wandb_logger.experiment.summary["avg_val_loss"] = avg_val_loss
    wandb_logger.experiment.log({"avg_val_loss": avg_val_loss})
    
    # Complete the wandb run
    wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()
