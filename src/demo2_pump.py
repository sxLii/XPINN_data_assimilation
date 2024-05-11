import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import add_noise
from XPINN import XPINNs

# data
# %%
DFh = pd.read_csv(r'../data/demo2_pump_h.csv')
DFq = pd.read_csv(r'../data/demo2_pump_q.csv')

# numbers of pipe's length
length_all = 20 * 50
length = [10 * 50, 10 * 50]
delta_length = [25, 12.5, 11.25]

# numbers of inner nodes
reaches_all = 22
reaches = [11, 11]

# pipe % node
pipe_number = 2
node_number = 1

# %%
t_space = DFh.iloc[:, 0].to_numpy()  # time serise, array
x_space = np.linspace(0, length_all, reaches_all)  # pipe length seris, shape(reaches)
x_m_space = []
x_mt_space = []

N_collo = 100  # number of collocation points
# Training Data
tlen = t_space.shape[0] - 1  # time points for training

for i in range(pipe_number):
    x_m_space.append(np.linspace(0, length[i], reaches[i]))
    x_mt_space.append(np.linspace(0, length[i], N_collo))
# Get pressure
p_all = DFh.iloc[:, 1:-1].values.T
q_all = DFq.iloc[:, 1:-1].values.T
# %%
p_m_all, q_m_all = [], []
for i in range(pipe_number):
    p_m_all.append(p_all[i * 11:(i + 1) * 11, :].copy())
    q_m_all.append(q_all[i * 11:(i + 1) * 11, :].copy())

# %% pipe data
x_all, t_all, x_all_cl = [], [], []
x_obs_train, t_obs_train = [], []
h_obs_train, q_obs_train = [], []

obs_idx = []

t_collo_train, x_collo_train = [], []

for i in range(pipe_number):
    # calculation domain
    x_all.append(np.tile(x_m_space[i], (len(DFh), 1)).transpose())
    x_all_cl.append(np.tile(x_mt_space[i], (len(DFh), 1)).transpose())
    t_all.append(np.tile(t_space, (reaches[i], 1)))

    # observation data
    idx = np.array([1, 5, -2], dtype='int')  # 3p
    x_obs_train.append(np.zeros(shape=(len(idx), tlen)))
    t_obs_train.append(np.zeros(shape=(len(idx), tlen)))
    h_obs_train.append(np.zeros(shape=(len(idx), tlen)))
    q_obs_train.append(np.zeros(shape=(len(idx), tlen)))
    for j in range(0, len(idx)):
        x_obs_train[i][j] = x_all[i][idx[j]][0:tlen]
        t_obs_train[i][j] = t_all[i][idx[j]][0:tlen]
        h_obs_train[i][j] = add_noise(p_m_all[i][idx[j]][0:tlen])
        # q_obs_train[i][j]=add_noise(q_m_all[i][idx[j]][0:tlen])
        if np.isnan(add_noise(q_m_all[i][idx[j]][0:tlen])).any:
            q_obs_train[i][j] = q_m_all[i][idx[j]][0:tlen]
        else:
            q_obs_train[i][j] = add_noise(q_m_all[i][idx[j]][0:tlen])

    obs_idx.append(idx)

    # Collocation Dataset
    idx_collo = np.linspace(0, reaches[i] - 1, N_collo, dtype='int')
    idx_collo_x = np.linspace(0, N_collo - 1, N_collo, dtype='int')

    x_collo_train.append(np.zeros(shape=(len(idx_collo), tlen)))
    t_collo_train.append(np.zeros(shape=(len(idx_collo), tlen)))

    for j in range(0, len(idx_collo)):
        x_collo_train[i][j] = x_all_cl[i][idx_collo_x[j]][0:tlen]
        t_collo_train[i][j] = t_all[i][idx_collo[j]][0:tlen]

# node data (interface)
node_idx = [[1, 2]]
inner_idx = [[-1, 0]]
inflow = [0.35]

x_inter_train = [[] for _ in node_idx]
t_inter_train = [[] for _ in node_idx]
q_inter_train = [[] for _ in node_idx]
h_inter_train = [[] for _ in node_idx]
qi_inflow = []

for i in range(node_number):
    for j, jid in enumerate(node_idx[i]):
        x_inter_train[i].append(x_all[jid - 1][inner_idx[i][j]][0:tlen])
        t_inter_train[i].append(t_all[jid - 1][inner_idx[i][j]][0:tlen])
        h_inter_train[i].append(p_m_all[jid - 1][inner_idx[i][j]][0:tlen])
        if np.isnan(add_noise(q_m_all[jid - 1][inner_idx[i][j]][0:tlen])).any:
            q_inter_train[i].append(q_m_all[jid - 1][inner_idx[i][j]][0:tlen])
        else:
            q_inter_train[i].append(add_noise(q_m_all[jid - 1][inner_idx[i][j]][0:tlen]))

    qi_inflow.append(np.zeros_like(q_inter_train[i][0]) + inflow[i])

a = [1000, 1164]
D = [1.0, 0.5]
fric = [0.013, 0.014]
layersh = [2] + 6 * [32] + [1]
layersu = [2] + 6 * [32] + [1]
layers = []
for i in range(pipe_number):
    layers.append(layersh)
    layers.append(layersu)

lb = []
ub = []
for i in range(pipe_number):
    lb.append(np.array([0., 0.]))
    ub.append(np.array([length[i], t_space[tlen]]))

model = XPINNs(pipe_number=pipe_number, x=x_collo_train, t=t_collo_train,
               x0=x_obs_train, t0=t_obs_train,
               h0=h_obs_train, q0=q_obs_train,
               lb=lb, ub=ub,
               node_number=node_number, node_index=node_idx, inner_index=inner_idx,
               xi=x_inter_train, ti=t_inter_train, qiflow=qi_inflow,
               a=a, D=D, fric=fric,
               layers_number=2 * pipe_number, layers=layers,
               w1=0.1, w2=0.01, w3=[500, 500])
model.train(100000)

# post process
h_pred_flat, q_pred_flat = [], []
x_test_flat, t_test_flat = [], []
h_test_flat, q_test_flat = [], []
x_test, t_test = [[] for _ in range(pipe_number)], [[] for _ in range(pipe_number)]
h_test, q_test = [[] for _ in range(pipe_number)], [[] for _ in range(pipe_number)]
Ntest = 11
tlen0 = tlen + 1

# storage datas
for i in range(pipe_number):
    idx_test = np.arange(0, reaches[i], 1)
    for j in range(0, len(idx_test)):
        x_test[i].append(x_all[i][idx_test[j]][0:tlen0])
        t_test[i].append(t_all[i][idx_test[j]][0:tlen0])
        h_test[i].append(p_m_all[i][idx_test[j]][0:tlen0])
        q_test[i].append(q_m_all[i][idx_test[j]][0:tlen0])

    x_test_flat.append(np.concatenate(x_test[i], axis=0)[:, None])
    t_test_flat.append(np.concatenate(t_test[i], axis=0)[:, None])
    h_test_flat.append(np.concatenate(h_test[i], axis=0)[:, None])
    q_test_flat.append(np.concatenate(q_test[i], axis=0)[:, None])

    hpred, qpred = model.predict(i, x_test_flat[i], t_test_flat[i])
    h_pred_flat.append(hpred)
    q_pred_flat.append(qpred)

plt.figure(100)
for i in range(pipe_number):
    h_test_reshape = h_test_flat[i].reshape(Ntest, tlen0)
    h_pred_reshape = h_pred_flat[i].reshape(Ntest, tlen0)
    for j, jnx in enumerate(obs_idx[i]):
        plt.subplot(pipe_number, len(obs_idx[0]), len(obs_idx[0]) * i + j + 1)
        plt.plot(h_test_reshape[jnx, :], label='true')
        plt.plot(h_pred_reshape[jnx, :], label='predicted')
plt.savefig(f'../res/demo2_pump/png/height.png', dpi=100, bbox_inches='tight')  # 会裁掉多余的白边

plt.figure(101)
for i in range(pipe_number):
    q_test_reshape = q_test_flat[i].reshape(Ntest, tlen0)
    q_pred_reshape = q_pred_flat[i].reshape(Ntest, tlen0)
    for j, jnx in enumerate(obs_idx[i]):
        plt.subplot(pipe_number, len(obs_idx[0]), len(obs_idx[0]) * i + j + 1)
        plt.plot(q_test_reshape[jnx, :], label='true')
        plt.plot(q_pred_reshape[jnx, :], label='predicted')

plt.savefig(f'../res/demo2_pump/png/flow.png', dpi=100, bbox_inches='tight')  # 会裁掉多余的白边

plt.figure(200)
errors_h = []
errors_q = []
for i in range(pipe_number):

    h_test_reshape = h_test_flat[i].reshape(reaches[i], tlen0)
    h_pred_reshape = h_pred_flat[i].reshape(reaches[i], tlen0)
    q_test_reshape = q_test_flat[i].reshape(reaches[i], tlen0)
    q_pred_reshape = q_pred_flat[i].reshape(reaches[i], tlen0)

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    variables = [('Pressure (m) ', h_pred_reshape, h_test_reshape), ('Flowrate (cms)', q_pred_reshape, q_test_reshape)]

    labels = ['(a)', '(c)', '(e)', '(b)', '(d)', '(f)']

    for j, (variable, pred, test) in enumerate(variables):
        titles = [f'PREDICT:{variable}', f'MOC:{variable}', f'Error:{variable}']
        images = [pred, test, np.abs(test - pred)]

        for k, ax in enumerate(axs[:, j]):
            im = ax.imshow(images[k], aspect='auto', cmap='plasma')
            ax.set_title(titles[k], fontsize=30)
            cbar = plt.colorbar(im, ax=ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=18)  # 设置colorbar的刻度大小

            if k == 2:  # 在最后一次迭代时设置 x 轴和 y 轴的标签
                ax.set_xlabel('Distance (m) ', fontsize=18)
            if j == 0:  # 在最右边的子图中设置 y 轴的标签
                ax.set_ylabel('Time (s) ', fontsize=18)

            # ax.set_xticks([0, 40, 80, 120, 160, 200], [0, 2, 4, 6, 8, 10], fontsize=18)
            # ax.set_yticks([0, 139, 278, 416], [0, 5, 10, 15], fontsize=18)

            ax.text(0.05, 0.05, labels[k + j * 3], transform=ax.transAxes, fontsize=22, weight='bold', color='white')
    plt.tight_layout()
    plt.savefig(f'../res/demo2_pump/png/cour{i + 1}.png', dpi=100, bbox_inches='tight')  # 会裁掉多余的白边

 # error
error_h = []
error_q = []
error_avg = []
rmse_h=[]
rmse_q=[]
for i in range(pipe_number):
    error_h0 = np.sqrt(np.sum(np.square(h_test_flat[i] - h_pred_flat[i])) / np.sum(np.square(h_test_flat[i])))
    error_h.append(error_h0)

    error_q0 = np.sqrt(np.sum(np.square(q_test_flat[i] - q_pred_flat[i])) / np.sum(np.square(q_test_flat[i])))
    error_q.append(error_q0)

    w3t = np.sum(np.square(h_test_flat[i] - h_pred_flat[i]))/np.sum(np.square(q_test_flat[i] - q_pred_flat[i]))
    error_avg0 = np.sqrt(np.sum(np.square(h_test_flat[i] - h_pred_flat[i])+w3t*np.square(q_test_flat[i] - q_pred_flat[i]))
                         / np.sum(np.square(h_test_flat[i])+w3t*np.square(q_test_flat[i])))
    error_avg.append(error_avg0)

    rmse_h0 = np.sqrt(np.sum(np.square(h_test_flat[i] - h_pred_flat[i])) / h_test_flat[i].shape[0])
    rmse_h.append(rmse_h0)

    rmse_q0 = np.sqrt(np.sum(np.square(q_test_flat[i] - q_pred_flat[i])) / q_test_flat[i].shape[0])
    rmse_q.append(rmse_q0)

## save errors
err_file = f"../res/demo2_pump/err.csv"
err = np.array([error_h, error_q, rmse_h, rmse_q, error_avg]).reshape((5, 2)).T
np.savetxt(err_file, err, delimiter=",")

# interface
h_inter_pred, q_inter_pred = [[] for _ in node_idx], [[] for _ in node_idx]
for i in range(node_number):
    plt.figure(i + 10)
    for j, jid in enumerate(node_idx[i]):
        x_inter_train_flat = x_inter_train[i][j].flatten()[:, None]
        t_inter_train_flat = t_inter_train[i][j].flatten()[:, None]
        hipred, qipred = model.predict(jid - 1, x_inter_train_flat, t_inter_train_flat)
        h_inter_pred[i].append(hipred)
        q_inter_pred[i].append(qipred)

        # vis
        plt.subplot(len(node_idx[i]), 2, 2 * j + 1)
        plt.plot(h_inter_train[i][j], label='true')
        plt.plot(h_inter_pred[i][j], label='predicted')
        plt.legend()

        plt.subplot(len(node_idx[i]), 2, 2 * j + 2)
        plt.plot(q_inter_train[i][j], label='true')
        plt.plot(q_inter_pred[i][j], label='predicted')
        plt.legend()

    plt.savefig(f'../res/demo2_pump/png/inter{i + 1}.png', dpi=100, bbox_inches='tight')  # 会裁掉多余的白边

# data output
name1 = r'../res/demo2_pump/all/'
name2 = r'../res/demo2_pump/obs/'
name3 = r'../res/demo2_pump/inter/'
name4 = r'../res/demo2_pump/loss/'

for i in range(pipe_number):
    h_test_reshape = h_test_flat[i].reshape(reaches[i], tlen0)
    h_pred_reshape = h_pred_flat[i].reshape(reaches[i], tlen0)
    q_test_reshape = q_test_flat[i].reshape(reaches[i], tlen0)
    q_pred_reshape = q_pred_flat[i].reshape(reaches[i], tlen0)
    np.save(name1 + f'h_test{i + 1}.npy', h_test_reshape)
    np.save(name1 + f'q_test{i + 1}.npy', q_test_reshape)
    np.save(name1 + f'h_pred{i + 1}.npy', h_pred_reshape)
    np.save(name1 + f'q_pred{i + 1}.npy', q_pred_reshape)

    for j, jnx in enumerate(obs_idx[i]):
        np.save(name2 + f'h_test_obs{jnx}', h_test_reshape[jnx, :])
        np.save(name2 + f'h_pred_obs{jnx}', h_pred_reshape[jnx, :])
        np.save(name2 + f'q_test_obs{jnx}', q_test_reshape[jnx, :])
        np.save(name2 + f'q_pred_obs{jnx}', q_pred_reshape[jnx, :])

for i in range(node_number):
    for j, jid in enumerate(node_idx[i]):
        np.save(name3 + f'h_inter_train{i + 1}{j}', h_inter_train[i][j])
        np.save(name3 + f'h_inter_pred{i + 1}{j}', h_inter_pred[i][j])
        np.save(name3 + f'q_inter_train{i + 1}{j}', q_inter_train[i][j])
        np.save(name3 + f'q_inter_pred{i + 1}{j}', q_inter_pred[i][j])

np.save(name4 + 'loss_pde.npy', model.loss_pdes)
np.save(name4 + 'loss_obs.npy', model.loss_observaitons)
np.save(name4 + 'loss_ints.npy', model.loss_interfaces)
np.save(name4 + 'loss.npy', model.losses)