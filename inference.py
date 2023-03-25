import os
import numpy as np
from torch.utils.data import DataLoader
from pat_utils import Data, model_finder, get_expt_name
from train import config, mparam, log_dir
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


mparam['L'] = 144
mparam['T'] = 24
# mparam['layers'] = []

# mparam['mean'] = True
# mparam['std'] = True

config['b'] = 5
config['dataset_type'] = 'carbon'
config['model'] = 'vanilla'

version = 0
log_dir = 'archive/logs/'

save_result = False
filename = 'results.csv'

expt_name = get_expt_name(config, mparam)
expt_dir = os.path.join(log_dir, expt_name, f'version_{version}', 'checkpoints')
checkpoint_name = os.listdir(expt_dir)[0]
load_path = os.path.join(expt_dir, checkpoint_name)

test_dataset = Data(building_index=config['b'], L=mparam['L'], T=mparam['T'], version='test',
                    dataset_type=config['dataset_type'])
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = model_finder(config, mparam)
model = model.load_from_checkpoint(load_path, **mparam)
model.eval()

pred_list = []
gt_list = []
loss_list = []
gt_t_list = []
pred_t_list = []
for t, (x, y) in enumerate(test_dataloader):
    y_hat = model(x)
    pred_list.append(y_hat.detach().numpy())
    gt = np.concatenate([x, y], axis=1)
    gt_list.append(gt)
    gt_t_list.append(np.arange(t, t + gt.shape[1]))
    pred_t_list.append(np.arange(t + x.shape[1], t + x.shape[1] + y.shape[1]))
pred_list = np.concatenate(pred_list)
gt_list = np.concatenate(gt_list)
gt_t_list = gt_t_list
pred_t_list = pred_t_list


fig, ax = plt.subplots(figsize=[15, 5])
plt.subplots_adjust(left=0.1, bottom=0.25)

i = 0
l_gt, = ax.plot(gt_t_list[i], gt_list[i], color='red')
line_gt = [l_gt]
l_pred, = ax.plot(pred_t_list[i], pred_list[i], color='blue')
line_pred = [l_pred]

l_v = ax.vlines(pred_t_list[i][0] - 1, ax.get_ylim()[0], ax.get_ylim()[1], colors='grey', linestyles='--', linewidth=1)
line_v = [l_v]

slider_ax = plt.axes([0.1, 0.1, 0.8, 0.05])
slider = Slider(slider_ax, 'Index', 0, len(gt_t_list)-1, valinit=0, valstep=1)


def update(val):
    i = int(val)

    line_gt[0].set_xdata(gt_t_list[i])
    line_gt[0].set_ydata(gt_list[i])

    line_pred[0].set_xdata(pred_t_list[i])
    line_pred[0].set_ydata(pred_list[i])

    min_y, max_y = np.min(gt_list[i]), np.max(gt_list[i])
    min_test, max_test = np.min(pred_list[i]), np.max(pred_list[i])
    if min_test < min_y:
        min_y = min_test
    if max_test > max_y:
        max_y = max_test

    xlim_offset = len(gt_t_list[i]) * 0.05
    ax.set_xlim([gt_t_list[i][0] - xlim_offset, gt_t_list[i][-1] + xlim_offset])

    ylim_offset = (max_y - min_y) * 0.05
    ax.set_ylim([min_y-ylim_offset, max_y + ylim_offset])

    # plt.vlines(gt_t_list[i][-1], ax.get_ylim()[0], ax.get_ylim()[1], colors='grey', linestyles='--', linewidth=1)
    line_v[0].remove()
    line_v[0] = ax.vlines(pred_t_list[i][0] - 1, ax.get_ylim()[0], ax.get_ylim()[1],
                          colors='grey', linestyles='--', linewidth=1)
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
