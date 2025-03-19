from argparse import ArgumentParser
from joblib import Parallel, delayed
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import seaborn as sns
import signal
from sklearn.metrics import f1_score
import sys
import time
import torch
import wandb

from idt.data import data
from idt.idt import IDT, get_activations
from idt.gnn import GNN


def create_and_train_GNN(num_features, num_classes, weight, args, train_loader, val_loader, logger, device, conv="GCN"):
    print(f"creating {conv} model")
    model = GNN(num_features, num_classes, layers=args.layers, dim=args.dim, activation=args.activation, conv=conv, pool=args.pooling, lr=args.lr, weight=weight)
    early_stop_callback = EarlyStopping(monitor=f"{conv}_val_loss", patience=25, mode="min")
    trainer = Trainer(
        max_steps=args.max_steps,
        callbacks=[early_stop_callback],
        logger=logger,
        devices=[device],
        enable_checkpointing=False,
        enable_progress_bar=True,
        log_every_n_steps=1
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(f'./checkpoints/{args.dataset}_{conv}_{logger.version}.ckpt')
    model.eval()
    print(f"Trained {conv} model")
    return model

def gen_heatmap(
    fidelity_matrix,
    title,
    row_labels,
    col_labels,
    logger,
    mask_zeros=False,
    reverse_y=False
):
    """
    Creates and logs a heatmap of the given matrix.
    - mask_zeros=True will hide all zero entries in the heatmap.
    - reverse_y=True flips the y-axis top-to-bottom.
    """
    # Create mask if requested
    mask = None
    if mask_zeros:
        mask = (fidelity_matrix == 0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        fidelity_matrix,
        annot=True,
        fmt=".4f",
        cmap="viridis",
        xticklabels=col_labels,
        yticklabels=row_labels,
        mask=mask
    )
    if reverse_y:
        plt.gca().invert_yaxis()
    plt.title(title)

    # Log the heatmap to wandb
    logger.experiment.summary[f"{title}_image"] = wandb.Image(plt)

    # Also log as a wandb Table
    fidelity_df = pd.DataFrame(fidelity_matrix, index=row_labels, columns=col_labels)
    logger.experiment.summary[f"{title}_table"] = wandb.Table(dataframe=fidelity_df)

    plt.close()

def run_cv(args, run_id):
    Parallel(n_jobs=args.kfold)(
            delayed(run_split)(args, cv_split, run_id=run_id, device=cv_split%args.devices) 
            for cv_split in range(args.kfold)
        )


def run_split(args, cv_split, run_id, device=0):
    torch.set_float32_matmul_precision('high')

    import time 
    num_features, num_classes, train_loader, val_loader, train_val_batch, test_batch = data(args.dataset, args.kfold, cv_split, seed=42)

    bincount = torch.bincount(train_val_batch.y, minlength=2)
    weight = len(train_val_batch) / (2 * bincount.float())
    
    os.environ["WANDB_SILENT"] = "true"
    logger = WandbLogger(project="gnnexplain", group=f'{args.dataset}_{run_id}')

    conv_types = [
        "GCN", 
        #"GIN", 
        #"SAGE", 
        #"GAT", 
        "DIR-GCN", 
        #"DIR-GIN", 
        #"DIR-SAGE", 
        #"DIR-GAT"
    ]

    # train GNNs
    models_dict = {}
    time_dict = {}
    for conv_name in conv_types:
        model_start = time.time()

        models_dict[conv_name] = create_and_train_GNN(
            num_features, num_classes, weight, args, train_loader,
            val_loader, logger, device, conv=conv_name
        )
        model_time = time.time() - model_start
        time_dict[conv_name] = model_time

    # compute GNN test set predictions and pairwise fidelities    
    n_models = len(conv_types)
    fidelity_matrix_gnn = np.zeros((n_models, n_models))

    predictions = {}

    
    eval_start = time.time()
    with torch.no_grad():
        for i, name in enumerate(conv_types):
            model = models_dict[name]
            prediction = model(test_batch).argmax(-1).detach().numpy()
            predictions[name] = prediction

            test_acc = (prediction == test_batch.y.detach().numpy()).mean()
            f1_macro = f1_score(test_batch.y.detach().numpy(), prediction, average='macro')

            logger.experiment.log({f'{name}_test_acc_{args.dataset}': test_acc})
            logger.experiment.log({f'{name}_f1_macro_{args.dataset}':f1_macro})
            print(f"{name} params: {model.count_params()} acc:{test_acc} f1:{f1_macro}")
            
            for j, other_name in enumerate(predictions.keys()):
                fidelity = (prediction == predictions[other_name]).mean()
                fidelity_matrix_gnn[i, j] = fidelity
                logger.experiment.log({f'{name}_{other_name}_fidelity_{args.dataset}': fidelity})
                print(f"   {name}_{other_name}_fidelity_{args.dataset}: {fidelity}")

    eval_end = time.time()-eval_start

    gen_heatmap(
        fidelity_matrix_gnn,
        title="Pairwise_GNN_Fidelity",
        row_labels=conv_types,
        col_labels=conv_types,
        logger=logger,
        mask_zeros=True,    # mask out zero entries
        reverse_y=True
    )
    print("gnns done")

    def run_idt(values, y, sample_weight, name):
        idt = IDT(width=args.width, sample_size=args.sample_size, layer_depth=args.layer_depth, max_depth=args.max_depth, ccp_alpha=args.ccp_alpha, directed=(args.dir_modal == 1)).fit(
            train_val_batch, values, y=y, sample_weight=sample_weight)
        
        idt_prediction = idt.predict(test_batch)

        test_acc = (idt_prediction == test_batch.y.detach().numpy()).mean()
        f1_macro = f1_score(test_batch.y.detach().numpy(), idt_prediction, average='macro')

        logger.experiment.log({f'{name}_test_acc_{args.dataset}': test_acc})
        logger.experiment.log({f'{name}_f1_macro_{args.dataset}': f1_macro})
        idt.prune()
        return idt_prediction
           
    sample_weight = weight[train_val_batch.y]

    ground_truth_labels = train_val_batch.y.detach().numpy()

    fidelity_matrix_idt = np.zeros((n_models, n_models))
    fidelity_matrix_idt_true = np.zeros((n_models, n_models))

    idt_fit_times = {}

    # fit idts and compute pairwise fidelities with original GNN outputs
    for i, name in enumerate(conv_types):
        model = models_dict[name]

        # get model activations
        print(f"{name} getting activations")
        activations = get_activations(train_val_batch, model)

        idt1_start = time.time()

        # fit IDT on model’s predicted labels
        print(f"{name} fitting idts")
        idt_pred_name = f"idt_{name}"
        idt_prediction = run_idt(activations, model(train_val_batch).argmax(-1), sample_weight, idt_pred_name)
        
        idt1_end = time.time()
        idt_fit_times[idt_pred_name] = idt1_end-idt1_start
        
        # fit IDT with ground truth labels
        idt_true_name = f"idt_{name}_true"
        idt_true_prediction = run_idt(activations, ground_truth_labels, sample_weight, idt_true_name)

        idt2_end = time.time()
        idt_fit_times[idt_true_name] = idt2_end-idt1_end

        for j, conv_type in enumerate(conv_types):
            fidelity_idt = (idt_prediction == predictions[conv_type]).mean()
            fidelity_matrix_idt[i, j] = fidelity_idt
            logger.experiment.log({f'{idt_pred_name}_{conv_type}_fidelity_{args.dataset}': fidelity_idt})
            print(f"   {idt_pred_name}_{conv_type}_fidelity_{args.dataset}: {fidelity_idt}")

            fidelity_idt_true = (idt_true_prediction == predictions[conv_type]).mean()
            fidelity_matrix_idt_true[i, j] = fidelity_idt_true
            logger.experiment.log[f'{idt_true_name}_{conv_type}_fidelity_{args.dataset}'] = fidelity_idt_true
            print(f"   {idt_true_name}_{conv_type}_fidelity_{args.dataset}: {fidelity_idt_true}")

    row_labels_idt = [f"idt_{ct}_" for ct in conv_types]
    row_labels_idt_true = [f"idt_{ct}_true" for ct in conv_types]

    gen_heatmap(
        fidelity_matrix_idt,
        title=f"Pairwise_IDT_Fidelity_{args.dataset}",
        row_labels=row_labels_idt,
        col_labels=conv_types,
        logger=logger,
        mask_zeros=False,
        reverse_y=True
    )

    gen_heatmap(
        fidelity_matrix_idt_true,
        title=f"Pairwise_True_IDT_Fidelity_{args.dataset}",
        row_labels=row_labels_idt_true,
        col_labels=conv_types,
        logger=logger,
        mask_zeros=False,
        reverse_y=True
    )

    print("true fitting idt")
    idt_true_prediction = run_idt(get_activations(train_val_batch, args.layers), ground_truth_labels, sample_weight, "idt_true")

    for j, conv_type in enumerate(conv_types):
        fidelity = (idt_true_prediction == predictions[conv_type]).mean()
        logger.experiment.log[f'idt_true_{conv_type}_fidelity_{args.dataset}'] = fidelity
        print(f"   idt_true_{conv_type}_fidelity: {fidelity}_{args.dataset}")

    logger.experiment.finish()

    for name, time in time_dict.items():
        print(f"{name}: {time}")

    print(f"eval: {eval_end}")

    for name, time in idt_fit_times.items():
        print(f"{name}: {time}")
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='EMLC0', help='Name of the dataset')
    parser.add_argument('--activation', type=str, default='ReLU', help='Activation Function', choices=['ReLU', 'Tanh', 'Sigmoid'])
    parser.add_argument('--pooling', type=str, default='mean', help='Pooling Function', choices=['mean', 'add'])
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--kfold', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of node embeddings')
    parser.add_argument('--max_steps', type=int, default=5000, help='Upper bound for the number of training steps')
    parser.add_argument('--width', type=int, default=8, help='Number of decision trees per layer')
    parser.add_argument('--sample_size', type=int, default=None, help='Size of subsamples to train decision trees on')
    parser.add_argument('--layer_depth', type=int, default=2, help='Depth of iterated decision trees')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of final tree')
    parser.add_argument('--ccp_alpha', type=float, default=1e-3, help='ccp_alpha of final tree')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices')
    parser.add_argument('--dir_modal', type=int, default=1, help='Whether to include the A^T modal parameters in the idt distillation', choices=[0,1])
    parser.add_argument('--conv', type=str, default='GCN', help='', choices=['GCN', 'GIN', 'SAGE', 'GAT', 'DIR-GCN', 'DIR-GIN', 'DIR-SAGE', 'DIR-GAT'])

    def signal_handler(sig, frame):
        signal.signal(sig, signal.SIG_IGN)
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parser.parse_args()

    curr_time = time.time()
    run_split(args, 0, run_id=curr_time, device=0)

    #run_cv(args, time.time())
    print("all good")