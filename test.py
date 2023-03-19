import os
import numpy as np
from torch.utils.data import DataLoader
from pat_utils import Model, Data
from train import L, T
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

expt_dir = os.path.join('logs', 'building_5', 'version_11', 'checkpoints')
checkpoint_name = os.listdir(expt_dir)[0]
load_path = os.path.join(expt_dir, checkpoint_name)

test_dataset = Data(building_index=5, L=L, T=T, version='test')
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = Model(L, T)
model = model.load_from_checkpoint(load_path, input_dim=L, output_dim=T)
model.eval()

pred_list = []
x_list = []
loss_list = []
for x, y in test_dataloader:
    x_list.append(x[:, -1])
    y_hat = model(x)
    pred_list.append(y_hat.detach().numpy())
    error = (y[:, 0] - y_hat[:, 0])**2
    loss_list.append(error.detach().numpy())

x = np.concatenate(x_list)
pred = np.concatenate(pred_list)
loss = np.concatenate(loss_list)
time_idx = np.arange(0, len(x))

window_size = 500
start_index = 0
end_index = start_index + window_size

fig, ax = plt.subplots(figsize=[15, 5])
plt.subplots_adjust(left=0.1, bottom=0.25)
alpha_list = np.linspace(0, 1, pred.shape[1])
lines = []
for i in range(pred.shape[1]):
    l, = ax.plot(time_idx[0:window_size] + 1 + i, pred[0:window_size, i],
                 color=(0.298, 0.447, 0.690, alpha_list[i]))
    lines.append(l)
gt, = ax.plot(time_idx[0:window_size], x[0:window_size], color='red')
line_gt = [gt]

slider_ax = plt.axes([0.1, 0.1, 0.8, 0.05])
slider = Slider(slider_ax, 'Index', 0, len(x)-window_size, valinit=start_index, valstep=1)

def update(val):
    start_index = int(val)
    end_index = start_index + window_size

    min_y, max_y = np.inf, -np.inf
    for i, line in enumerate(lines):
        line.set_xdata(time_idx[start_index:end_index] + 1 + i)
        line.set_ydata(pred[start_index:end_index, i])
        min_test, max_test = np.min(pred[start_index:end_index, i]), np.max(pred[start_index:end_index, i])
        if min_test < min_y:
            min_y = min_test
        if max_test > max_y:
            max_y = max_test

    line_gt[0].set_xdata(time_idx[start_index:end_index])
    line_gt[0].set_ydata(x[start_index:end_index])
    min_test, max_test = np.min(x[start_index:end_index]), np.max(x[start_index:end_index])
    if min_test < min_y:
        min_y = min_test
    if max_test > max_y:
        max_y = max_test

    xlim_offset = window_size * 0.05
    ax.set_xlim([time_idx[start_index] - xlim_offset, time_idx[end_index - 1] + xlim_offset])

    ylim_offset = (max_y - min_y) * 0.05
    ax.set_ylim([min_y-ylim_offset, max_y + ylim_offset])
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()

mse = np.mean(loss)
print(f'mse = {mse}')
