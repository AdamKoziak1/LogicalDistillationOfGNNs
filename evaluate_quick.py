from argparse import ArgumentParser
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import os
from pytorch_lightning.loggers import WandbLogger
import signal
import sys
import time
import torch
import wandb

import json
import numpy as np

from idt.data import data
from idt.idt import IDT, get_activations
from idt.gnn import GNN


def create_and_train_GNN(num_features, num_classes, weight, args, train_loader, val_loader, logger, device, conv="GCN", cv_split=0):
    print(f"creating {conv} model")
    model = GNN(num_features, num_classes, layers=args.layers, dim=args.dim, activation=args.activation, conv=conv, pool=args.pooling, lr=args.lr, weight=weight, norm=args.norm)

    early_stop_callback = EarlyStopping(monitor=f"{conv}_val_loss", patience=150, mode="min")
    name = f"{args.dataset}_{conv}_split_{cv_split}_best_"
    
    checkpoint_callback = ModelCheckpoint(
        monitor=f"{conv}_val_loss",
        save_top_k=1,  # Save only the best model
        mode="min",  # Minimize validation loss
        dirpath="./checkpoints/",
        filename=name+"{epoch}_{"+f"{conv}"+"_val_acc:.4f}",
    )

    trainer = Trainer(
        max_steps=args.max_steps,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        devices=[device],
        enable_checkpointing=True,
        enable_progress_bar=True,
        log_every_n_steps=1
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model.eval()
    print(f"Trained {conv} model {cv_split}")
    return model
    
def run_split(args, cv_split, run_id, device=0):
    torch.set_float32_matmul_precision('high')

    num_features, num_classes, train_loader, val_loader, train_val_batch, test_batch = data(args.dataset, args.kfold, cv_split, seed=42)

    bincount = torch.bincount(train_val_batch.y, minlength=2)
    weight = len(train_val_batch) / (2 * bincount.float())
    
    os.environ["WANDB_SILENT"] = "true"
    logger = WandbLogger(project="gnnexplain_quick", group=f'{args.dataset}_{args.conv}', name=f'{args.dataset}_{args.conv}_cv_split_{cv_split}_{run_id}')

    # train GNN
    gnn = create_and_train_GNN(
        num_features, num_classes, weight, args, train_loader,
        val_loader, logger, device, conv=args.conv, cv_split=cv_split
    )

    # compute GNN test set predictions 

    name = args.conv
    
    with torch.no_grad():
        prediction = gnn(test_batch).argmax(-1).detach().numpy()

        gnn_acc = (prediction == test_batch.y.detach().numpy()).mean()

        print(f"{name} params: {gnn.count_params()} acc:{gnn_acc}")

    def run_idt(values, y, sample_weight, name, dir_modal):
        idt = IDT(width=args.width, sample_size=args.sample_size, layer_depth=args.layer_depth, max_depth=args.max_depth, ccp_alpha=args.ccp_alpha, directed=dir_modal).fit(
            train_val_batch, values, y=y, sample_weight=sample_weight)
        
        idt_prediction = idt.predict(test_batch)

        test_acc = (idt_prediction == test_batch.y.detach().numpy()).mean()

        idt.prune()
        return test_acc
           
    sample_weight = weight[train_val_batch.y]

    ground_truth_labels = train_val_batch.y.detach().numpy()

    # fit idts 
    # get model activations
    print(f"{name} getting activations")
    activations = get_activations(train_val_batch, gnn)
    gnn_labels = gnn(train_val_batch).argmax(-1)

    # fit IDT on model’s predicted labels
    print(f"{name} fitting idts")
    idt_gnn_acc = run_idt(activations, gnn_labels, sample_weight, "idt_gnn", dir_modal=False)
        
    # fit IDT with ground truth labels
    idt_gnn_true_acc = run_idt(activations, ground_truth_labels, sample_weight, "idt_gnn_true", dir_modal=False)

    print("true fitting idt")
    # fit pure IDT
    idt_true_acc = run_idt(get_activations(train_val_batch, args.layers), ground_truth_labels, sample_weight, "idt_true", dir_modal=False)


    # fit IDT on model’s predicted labels
    print(f"{name} fitting idts with a^T modal")
    idt_gnn_acc_modal = run_idt(activations, gnn_labels, sample_weight, "idt_gnn_a^T", dir_modal=True)
        
    # fit IDT with ground truth labels
    idt_gnn_true_modal_acc = run_idt(activations, ground_truth_labels, sample_weight, "idt_gnn_true_a^T", dir_modal=True)

    print("true fitting idt")
    # fit pure IDT
    idt_true_modal_acc = run_idt(get_activations(train_val_batch, args.layers), ground_truth_labels, sample_weight, "idt_true_a^T", dir_modal=True)

    logger.experiment.finish()

    return {
        'gnn': gnn_acc,  
        'idt_gnn': idt_gnn_acc,  
        'idt_gnn_true': idt_gnn_true_acc,  
        'idt_true': idt_true_acc,  
        'idt_gnn_a^T': idt_gnn_acc_modal,  
        'idt_gnn_true_a^T': idt_gnn_true_modal_acc,  
        'idt_true_a^T': idt_true_modal_acc
    }

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='EMLC0', help='Name of the dataset')
    parser.add_argument('--activation', type=str, default='ReLU', help='Activation Function', choices=['ReLU', 'Tanh', 'Sigmoid'])
    parser.add_argument('--pooling', type=str, default='mean', help='Pooling Function', choices=['mean', 'add'])
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--kfold', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of node embeddings')
    parser.add_argument('--max_steps', type=int, default=2500, help='Upper bound for the number of training steps')
    parser.add_argument('--width', type=int, default=10, help='Number of decision trees per layer')
    parser.add_argument('--sample_size', type=int, default=None, help='Size of subsamples to train decision trees on')
    parser.add_argument('--layer_depth', type=int, default=2, help='Depth of iterated decision trees')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of final tree')
    parser.add_argument('--ccp_alpha', type=float, default=1e-3, help='ccp_alpha of final tree')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices')
    parser.add_argument('--dir_modal', type=int, default=1, help='Whether to include the A^T modal parameters in the idt distillation', choices=[0,1])
    parser.add_argument('--conv', type=str, default='GCN', help='', choices=['GCN', 'GIN', 'SAGE', 'GAT', 'DIR-GCN', 'DIR-GIN', 'DIR-SAGE', 'DIR-GAT'])
    parser.add_argument('--norm', type=int, default=1, help='', choices=[0,1])

    def signal_handler(sig, frame):
        signal.signal(sig, signal.SIG_IGN)
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parser.parse_args()

    curr_time = time.time()
    run_id=curr_time

    results = {
        'gnn': [],
        'idt_gnn': [],
        'idt_gnn_true': [],
        'idt_true': [],
        'idt_gnn_a^T': [],
        'idt_gnn_true_a^T': [],
        'idt_true_a^T': []
    }
    
    for cv_split in range(args.kfold):
        fold_results = run_split(args, cv_split, run_id=run_id, device=cv_split%args.devices) 
        for key in results:
            results[key].append(fold_results[key])

    avg_results = {key: np.mean(val) for key, val in results.items()}

    # Save the results as a JSON dump
    results_file = f'./results/cv_results_{args.dataset}_{args.conv}.json'
    with open(results_file, 'w') as f:
        json.dump(avg_results, f, indent=4)

    print(f"Results saved to {results_file}")

    print("all good")