import torch
from torch.utils.data import DataLoader
from pat_utils import Data, Model
from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.tuner.tuning import Tuner
torch.set_float32_matmul_precision('medium')

L = 144      # window length
T = 24      # future time-steps
hidden_layers = []
building = 5
dataset_type = 'carbon'


if __name__ == '__main__':
    h_str = str(hidden_layers).replace('[', '').replace(']', '').replace(', ', '_')
    expt_name = f'b{building}{dataset_type}_L{L}_T{T}_h{h_str}'

    train_dataset = Data(building_index=building, L=L, T=T, version='train', dataset_type=dataset_type)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = Data(building_index=building, L=L, T=T, version='validate', dataset_type=dataset_type)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=15, verbose=False, mode="min")

    model = Model(L, T, hidden_layers)
    logger = TensorBoardLogger('logs/', name=expt_name)
    trainer = Trainer(max_epochs=100, logger=logger, accelerator="cuda", devices=find_usable_cuda_devices(1),
                      log_every_n_steps=10,
                      callbacks=[early_stop_callback])

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_dataloader, val_dataloader, min_lr=1e-6, max_lr=1e-1)

    lr = lr_finder.suggestion()
    model.learning_rate = lr

    trainer.fit(model, train_dataloader, val_dataloader)
