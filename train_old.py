import torch
from torch.utils.data import DataLoader
from pat_utils import Data, model_finder, get_expt_name
from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.tuner.tuning import Tuner
# from lightning.pytorch.callbacks.progress import TQDMProgressBar

# class CustomProgressBar(TQDMProgressBar):
#     def get_metrics(self, *args, **kwargs):
#         # don't show the version number
#         items = super().get_metrics(args[0], args[1])
#         items.pop("v_num", None)
#         items['hellow'] = 'world'
#         return items

torch.set_float32_matmul_precision('medium')

config = {
          'b': 5,   # building index
          'dataset_type': 'price',
          'model': 'vanilla'}
mparam = {
    'L': 144,       # input window          # generally improves performance with increasing L
    'T': 48,       # future time-steps
    'layers': [],  # layers for MLP
    # 'mean': False,   # normalise model
    # 'std': False     # normalise model
    }
log_dir = 'archive/logtesting'

if __name__ == '__main__':
    expt_name = get_expt_name(config, mparam)
    train_dataset = Data(building_index=config['b'], L=mparam['L'], T=mparam['T'], version='train',
                         dataset_type=config['dataset_type'])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = Data(building_index=config['b'], L=mparam['L'], T=mparam['T'], version='validate',
                       dataset_type=config['dataset_type'])
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=25, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    model = model_finder(config, mparam)
    logger = TensorBoardLogger(f'{log_dir}/', name=expt_name)

    trainer = Trainer(max_epochs=200, logger=logger, accelerator="cuda", devices=find_usable_cuda_devices(1),
                      log_every_n_steps=10,
                      callbacks=[
                          # checkpoint_callback,   # use either this or early stop callback
                          early_stop_callback, # use either this or checkpoint callback
                          # CustomProgressBar(),
                      ]
                      )

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_dataloader, val_dataloader, min_lr=1e-6, max_lr=1e-1)

    lr = lr_finder.suggestion()
    model.learning_rate = lr

    trainer.fit(model, train_dataloader, val_dataloader)
