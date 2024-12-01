import torch

from idt.data import data

for i in range(10,11):
    dataset = 'EMLC' + str(i)

    num_features, num_classes, train_loader, val_loader, train_val_batch, test_batch = data(dataset, 10, 0, seed=42)
    
    bincount = torch.bincount(train_val_batch.y, minlength=2)
    weight = len(train_val_batch) / (2 * bincount.float())
    
    from lightning import Trainer
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping

    from idt.gnn import GNN

    torch.set_float32_matmul_precision('high')

    GCN = GNN(num_features, num_classes, layers=8, dim=128, activation="ReLU", conv="GCN", pool="mean", lr=1e-4, weight=weight)
    early_stop_callback = EarlyStopping(monitor="GCN_val_loss", patience=10, mode="min")

    trainer = Trainer(
        max_steps=1000,
        enable_checkpointing=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
        callbacks=[early_stop_callback],
    )
    trainer.fit(GCN, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    from idt.idt import IDT, get_activations
    import logging

    idt = IDT(width=8, sample_size=1000, layer_depth=2, max_depth=None, ccp_alpha=1e-3)
    values = get_activations(train_val_batch, GCN)
    idt.fit(train_val_batch, values, train_val_batch.y)

    logging.basicConfig(filename='output.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    def log_run_results():
        logging.info(f"Dataset: {dataset}")
        logging.info(f"GCN test accuracy: {(GCN(test_batch).argmax(-1) == test_batch.y).float().mean().item()}")
        logging.info(f"IDT test accuracy: {idt.accuracy(test_batch)}")
        logging.info(f"IDT F1 score: {idt.f1_score(test_batch)}")
        logging.info(f"Fidelity: {idt.fidelity(test_batch, GCN)}")
        logging.info(f"Binary Count: {bincount}")
        logging.info(f"Weight: {weight}")

    log_run_results()
    
# Change to call evaluate
# Change number of samples
# return for all types of trees and differentiate them + log
# latex code output