import random

import torch
import torch.nn as nn
import torch.utils.data as Data

import argparse
import os
import time
import numpy as np
import pandas as pd

from grasp.lstm_ae.models.LSTMAE import LSTMAE
from grasp.lstm_ae.train_utils import train_model, eval_model

import matplotlib.pyplot as plt

from sklearn import preprocessing

parser = argparse.ArgumentParser(description='LSTM_AE TOY EXAMPLE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train')
parser.add_argument('--optim', default='Adam', type=str, help='Optimizer to use')
parser.add_argument('--hidden-size', type=int, default=128, metavar='N', help='LSTM hidden state size')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--input-size', type=int, default=12, metavar='N', help='input size')
parser.add_argument('--dropout', type=float, default=0.0, metavar='D', help='dropout ratio')
parser.add_argument('--wd', type=float, default=0, metavar='WD', help='weight decay')
parser.add_argument('--grad-clipping', type=float, default=None, metavar='GC', help='gradient clipping value')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batch iteration to log status')
parser.add_argument('--model-type', default='LSTMAE', help='currently only LSTMAE')
parser.add_argument('--model-dir', default='trained_models', help='directory of model for saving checkpoint')
parser.add_argument('--seq-len', default=150, help='sequence full size')
parser.add_argument('--run-grid-search', action='store_true', default=False, help='Running hyper-parameters grid search')

args = parser.parse_args(args=[])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())  # for npy data recording
norm_in = preprocessing.MinMaxScaler(feature_range=(-1, 1))
norm_out = preprocessing.MinMaxScaler(feature_range=(-1, 1))
def get_train_data():
    def get_tensor_from_pd(dataframe_series) -> torch.Tensor:
        return torch.tensor(data=dataframe_series.values)

    def create_inout_sequences(input_data, tw):
        # input_data: input raw data
        # tw: train window, how many sequence you want, such as the window is 10, means 10 steps for data like 13*10
        inout_seq = []
        L = input_data.shape[1]  # 13 * 150
        inout_seq = torch.zeros(L - tw, args.input_size, tw) # 13 need to be considered
        for i in range(L - tw):
            train_seq = input_data[:, i:i + tw]
            train_label = input_data[:, i + tw:i + tw + 1]
            # inout_seq.append((train_seq, train_label))
            inout_seq[i, :, :] = train_seq
        return inout_seq

    #
    import numpy as np
    import pandas as pd
    import random
    # # 生成训练数据x并做归一化后，构造成dataframe格式，再转换为tensor格式
    # df = pd.DataFrame(data=preprocessing.MinMaxScaler().fit_transform(np.random.randint(0, 10, size=(2000, 300))))
    # y = pd.Series(np.random.randint(0, 2, 2000))
    # return get_tensor_from_pd(df).float(), get_tensor_from_pd(y).float()

    # data_path_dir = './grasp/data/labeled_data/'
    data_path_dir = './grasp/data/clip_data/'
    file_name = '20240402223150_clipped_data.npz'
    tac_data = np.load(data_path_dir + file_name)['labeled_data'].transpose()
    in_data = tac_data[:args.input_size, :].copy()
    out_data = tac_data[:args.input_size, :].copy()
    if args.input_size > 12:
        rate = 0.2
        for i in range(in_data.shape[1]):
            if random.random() < rate:
                in_data[-1, i] = -1
    # input and output for reconstruct
    norm = False
    if norm is True:
        norm_in_max = norm_in.fit(in_data.reshape(-1, 1)).data_max_
        norm_in_min = norm_in.fit(in_data.reshape(-1, 1)).data_min_
        # print(norm_in_max, norm_in_min, in_data.reshape(-1, 1).shape)
        in_data = norm_in.fit_transform(in_data.reshape(args.input_size,-1))
        # print(in_data)
        # time.sleep(10)
        out_data = norm_out.fit_transform(out_data.reshape(args.input_size,-1))
        in_data = torch.FloatTensor(in_data).view(args.input_size, -1)
        out_data = torch.FloatTensor(out_data).view(args.input_size, -1)

        # _Mm = 20 # for normalization
        # _CMm = 3 # for class normalization
        # tac_data[:, :12] = tac_data[:, :12] / _Mm
        # tac_data[:, 12] = (tac_data[:, 12] + 1) / _CMm
    # else:
    #     in_ = pd.DataFrame(data=in_data)
    #     out_ = pd.DataFrame(data=out_data)
    train_data = in_data[:, :in_data.shape[1]-150*4]
    val_data = in_data[:, in_data.shape[1]-150*4:in_data.shape[1]-150*2]
    test_data = in_data[:, :150]
    print(test_data.transpose().shape)
    # time.sleep(10)
    plot_toy_data(test_data.transpose(), f'Original sequence #{1}', color='g')

    tac_train_data = create_inout_sequences(train_data, args.seq_len)
    tac_val_data = create_inout_sequences(val_data, args.seq_len)
    tac_test_data = create_inout_sequences(test_data, args.seq_len)
    real_data = create_inout_sequences(out_data, args.seq_len)

    # y = pd.Series(np.random.randint(0, 2, tac_data.shape[0]))

    # print(len(tac_data), tac_data[0][0].shape)
    # print(tac_data.shape)
    # # print(torch.Tensor(tac_data))
    # time.sleep(10)

    # return get_tensor_from_pd(tac_data[:][0]).float(), get_tensor_from_pd(tac_data[:][0]).float()
    return tac_train_data.transpose(1,2), tac_val_data.transpose(1,2), tac_test_data.transpose(1,2), norm_in_max, norm_in_min

class toy_dataset(torch.utils.data.Dataset):
    def __init__(self, toy_data):
        self.toy_data = toy_data

    def __len__(self):
        return self.toy_data.shape[0]

    def __getitem__(self, index):
        return self.toy_data[index]

def plot_toy_data(toy_example, description, color='b'):
    """
    Recieves a toy raw data sequence and plot it
    :param toy_example: toy data example sequence
    :param description: additional description to the plot
    :param color: graph color
    :return:
    """
    # time_lst = [t for t in range(toy_example.shape[0])]
    #
    # plt.figure()
    # plt.plot(time_lst, toy_example.tolist(), color=color)
    # plt.xlabel('Time')
    # plt.ylabel('Signal Value')
    # # plt.legend()
    # plt.title(f'Single value vs. time for toy example {description}')
    # plt.show()

    my_dpi = 90
    row = 2
    col = 6
    fig, axs = plt.subplots(row, col, figsize=(480 / my_dpi, 480 / my_dpi), dpi=my_dpi, sharex=False, sharey=False)
    _legend = False
    ylabel = [
        'sensor1_x', 'sensor1_y', 'sensor1_z',
        'sensor2_x', 'sensor2_y', 'sensor2_z',
        'sensor3_x', 'sensor3_y', 'sensor3_z',
        'sensor4_x', 'sensor4_y', 'sensor4_z',
    ]
    # print(np.linspace(0, len(tac_datalists[0])-1, len(tac_datalists[0])))
    # print(tac_datalists)
    for i in range(row):
        for j in range(col):
            if i == 1:
                yl = i * j + 6
            elif i == 0:
                yl = (i + 1) * j

            axs[i][j].plot(np.linspace(0, len(toy_example[:, 1]) - 1, len(toy_example[:, 1])),
                               toy_example[:, yl])
            if _legend is False:
                axs[i][j].legend()
                _legend = True
            axs[i][j].legend()
            # axs[i][j].plot(data_xlen, tac_data[:, i+j])
            axs[i][j].set_ylabel(ylabel[yl])
            axs[i][j].set_xlabel('time')
    plt.show()

if __name__ == '__main__':
    # 得到数据
    x, y, z, norm_max, norm_min = get_train_data() # x: input, y: output for reconstruction
    x = toy_dataset(x)
    y = toy_dataset(y)
    z = toy_dataset(z)
    train_loader = Data.DataLoader(
        dataset=x,  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=args.batch_size,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
    )
    val_loader = Data.DataLoader(
        dataset=y,  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=args.batch_size,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
    )
    test_loader = Data.DataLoader(
        dataset=z,  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=args.batch_size,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
    )
    train = False
    if train is True:
        # loss，optim，epochs
        model = LSTMAE(input_size=args.input_size, hidden_size=args.hidden_size, dropout_ratio=args.dropout,
                       seq_len=args.seq_len)
        model.to(device)
        # model = LstmAutoEncoder(input_layer=13, hidden_layer=5, batch_size=50)  # lstm
        # model = LstmFcAutoEncoder()  # lstm+fc模型
        loss_function = nn.MSELoss(reduction='sum')  # loss
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
        optimizer = getattr(torch.optim, args.optim)(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
        # # Grid search run if run-grid-search flag is active
        # if args.run_grid_search:
        #     hyper_params_grid_search(train_iter, val_iter, criterion)
        #     return
        for epoch in range(args.epochs):
            # for batch_id in train_loader:
            train_model(loss_function, epoch, model, args.model_type,
                        optimizer, train_loader, args.batch_size,
                        args.grad_clipping, args.log_interval)
            eval_model(loss_function, model, args.model_type, val_loader)
        eval_model(loss_function, model, args.model_type, test_loader, mode='Test')
        # Save model
        torch.save(model.state_dict(), os.path.join(args.model_dir, f'model_hs={args.hidden_size}_bs={args.batch_size}'
                                                                        f'_epochs={args.epochs}_clip={args.grad_clipping}_{current_time}.pt'))
    else:
        model = LSTMAE(input_size=args.input_size, hidden_size=args.hidden_size, dropout_ratio=args.dropout,
                       seq_len=args.seq_len)
        model.to(device)
        model.load_state_dict(torch.load('trained_models/model_hs=128_bs=128_epochs=1000_clip=None.pt'))
        # model = torch.load('trained_models/model_hs=128_bs=128_epochs=1000_clip=None.pt')
        model.eval()
        plot_test_iter = iter(torch.utils.data.DataLoader(test_loader.dataset, batch_size=1, shuffle=False))
        orig = next(plot_test_iter).to(device)
        with torch.no_grad():
            rec = model(orig)
        time_lst = [t for t in range(orig.shape[1])]
        # print(orig.cpu().numpy().reshape(args.seq_len, args.input_size))
        # time.sleep(10)
        # inverse norm
        orig = (orig.squeeze().cpu().numpy() - (-1)) / (1 - (-1)) * (norm_max - norm_min) + norm_min
        rec = (rec.squeeze().cpu().numpy() - (-1)) / (1 - (-1)) * (norm_max - norm_min) + norm_min
        # orig = norm_in.inverse_transform(orig.squeeze().cpu().numpy())
        # rec = norm_in.inverse_transform(rec.squeeze().cpu().numpy())
        plot_toy_data(orig.squeeze(), f'Original sequence #{1}', color='g')
        plot_toy_data(rec.squeeze(), f'Reconstructed sequence #{1}', color='r')
        plt.figure()
        plt.plot(time_lst, orig.squeeze().tolist(), color='g', label='Original signal')
        plt.plot(time_lst, rec.squeeze().tolist(), color='r', label='Reconstructed signal')
        plt.xlabel('Time')

        plt.ylabel('Signal Value')
        plt.legend()
        title = f'Original and Reconstruction of Single values vs. time for toy example #{1}'
        plt.title(title)
        plt.savefig(f'{title}.png')
        plt.show()





