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

parser = argparse.ArgumentParser(description='LSTM_AE TOY EXAMPLE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train')
parser.add_argument('--optim', default='Adam', type=str, help='Optimizer to use')
parser.add_argument('--hidden-size', type=int, default=128, metavar='N', help='LSTM hidden state size')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--input-size', type=int, default=13, metavar='N', help='input size')
parser.add_argument('--dropout', type=float, default=0.0, metavar='D', help='dropout ratio')
parser.add_argument('--wd', type=float, default=0, metavar='WD', help='weight decay')
parser.add_argument('--grad-clipping', type=float, default=None, metavar='GC', help='gradient clipping value')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batch iteration to log status')
parser.add_argument('--model-type', default='LSTMAE', help='currently only LSTMAE')
parser.add_argument('--model-dir', default='trained_models', help='directory of model for saving checkpoint')
parser.add_argument('--seq-len', default=50, help='sequence full size')
parser.add_argument('--run-grid-search', action='store_true', default=False, help='Running hyper-parameters grid search')

args = parser.parse_args(args=[])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_train_data():
    def get_tensor_from_pd(dataframe_series) -> torch.Tensor:
        return torch.tensor(data=dataframe_series.values)

    def create_inout_sequences(input_data, tw):
        # input_data: input raw data
        # tw: train window, how many sequence you want, such as the window is 10, means 10 steps for data like 13*10
        inout_seq = []
        L = input_data.shape[1]  # 13 * 150
        inout_seq = torch.zeros(L - tw, 13, tw) # 13 need to be considered
        for i in range(L - tw):
            train_seq = input_data[:, i:i + tw]
            train_label = input_data[:, i + tw:i + tw + 1]
            # inout_seq.append((train_seq, train_label))
            inout_seq[i, :, :] = train_seq
        return inout_seq

    #
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    import random
    # # 生成训练数据x并做归一化后，构造成dataframe格式，再转换为tensor格式
    # df = pd.DataFrame(data=preprocessing.MinMaxScaler().fit_transform(np.random.randint(0, 10, size=(2000, 300))))
    # y = pd.Series(np.random.randint(0, 2, 2000))
    # return get_tensor_from_pd(df).float(), get_tensor_from_pd(y).float()

    # data_path_dir = './grasp/data/labeled_data/'
    data_path_dir = './grasp/data/clip_data/'
    file_name = '20240402223150_clipped_data.npz'
    tac_data = np.load(data_path_dir + file_name)['labeled_data'].transpose()
    in_data = tac_data[:13, :].copy()
    out_data = tac_data[:13, :].copy()
    rate = 0.2
    for i in range(in_data.shape[1]):
        if random.random() < rate:
            in_data[-1, i] = -1
    # input and output for reconstruct
    norm = True
    if norm is True:
        in_data = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit_transform(in_data.reshape(13,-1))
        out_data = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit_transform(out_data.reshape(13,-1))
        in_data = torch.FloatTensor(in_data).view(13, -1)
        out_data = torch.FloatTensor(out_data).view(13, -1)
        # _Mm = 20 # for normalization
        # _CMm = 3 # for class normalization
        # tac_data[:, :12] = tac_data[:, :12] / _Mm
        # tac_data[:, 12] = (tac_data[:, 12] + 1) / _CMm
    # else:
    #     in_ = pd.DataFrame(data=in_data)
    #     out_ = pd.DataFrame(data=out_data)
    train_data = in_data[:, :in_data.shape[1]-600]
    val_data = in_data[:, in_data.shape[1]-600:]
    test_data = in_data[:, in_data.shape[1]-300:]

    tac_train_data = create_inout_sequences(train_data, 10)
    tac_val_data = create_inout_sequences(val_data, 10)
    tac_test_data = create_inout_sequences(test_data, 10)
    real_data = create_inout_sequences(out_data, 10)

    # y = pd.Series(np.random.randint(0, 2, tac_data.shape[0]))

    # print(len(tac_data), tac_data[0][0].shape)
    # print(tac_data.shape)
    # # print(torch.Tensor(tac_data))
    # time.sleep(10)

    # return get_tensor_from_pd(tac_data[:][0]).float(), get_tensor_from_pd(tac_data[:][0]).float()
    return tac_train_data.transpose(1,2), tac_val_data.transpose(1,2), tac_test_data.transpose(1,2)

class LstmAutoEncoder(nn.Module):
    def __init__(self, input_layer=300, hidden_layer=100, batch_size=20):
        super(LstmAutoEncoder, self).__init__()

        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.batch_size = batch_size
        self.encoder_lstm = nn.LSTM(self.input_layer, self.hidden_layer, batch_first=True)
        self.decoder_lstm = nn.LSTM(self.hidden_layer, self.input_layer, batch_first=True)

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        # encoder
        encoder_lstm, (n, c) = self.encoder_lstm(input_x,
                                                 (torch.zeros(1, self.batch_size, self.hidden_layer),
                                                  torch.zeros(1, self.batch_size, self.hidden_layer)))
        # decoder
        decoder_lstm, (n, c) = self.decoder_lstm(encoder_lstm,
                                                 (torch.zeros(1, self.batch_size, self.input_layer),
                                                  torch.zeros(1, self.batch_size, self.input_layer)))
        return decoder_lstm.squeeze()


class LstmFcAutoEncoder(nn.Module):
    def __init__(self, input_layer=300, hidden_layer=100, batch_size=20):
        super(LstmFcAutoEncoder, self).__init__()

        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.batch_size = batch_size

        self.encoder_lstm = nn.LSTM(self.input_layer, self.hidden_layer, batch_first=True)
        self.encoder_fc = nn.Linear(self.hidden_layer, self.hidden_layer)
        self.decoder_lstm = nn.LSTM(self.hidden_layer, self.input_layer, batch_first=True)
        self.decoder_fc = nn.Linear(self.hidden_layer, self.hidden_layer)
        self.relu = nn.ReLU()

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        # encoder
        encoder_lstm, (n, c) = self.encoder_lstm(input_x,
                                                 # shape: (n_layers, batch, hidden_size)
                                                 (torch.zeros(1, self.batch_size, self.hidden_layer),
                                                  torch.zeros(1, self.batch_size, self.hidden_layer)))
        encoder_fc = self.encoder_fc(encoder_lstm)
        encoder_out = self.relu(encoder_fc)
        # decoder
        decoder_fc = self.relu(self.decoder_fc(encoder_out))
        decoder_lstm, (n, c) = self.decoder_lstm(decoder_fc,
                                                 (torch.zeros(1, 20, self.input_layer),
                                                  torch.zeros(1, 20, self.input_layer)))
        return decoder_lstm.squeeze()

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
    time_lst = [t for t in range(toy_example.shape[0])]

    plt.figure()
    plt.plot(time_lst, toy_example.tolist(), color=color)
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    # plt.legend()
    plt.title(f'Single value vs. time for toy example {description}')
    plt.show()

if __name__ == '__main__':
    # 得到数据
    x, y, z = get_train_data() # x: input, y: output for reconstruction
    x = toy_dataset(x)
    y = toy_dataset(y)
    z = toy_dataset(z)
    train_loader = Data.DataLoader(
        dataset=x,  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=50,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
    )
    val_loader = Data.DataLoader(
        dataset=y,  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=50,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
    )
    test_loader = Data.DataLoader(
        dataset=z,  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=50,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
    )
    train = False
    if train is True:
        # 建模三件套：loss，优化，epochs
        model = LSTMAE(input_size=args.input_size, hidden_size=args.hidden_size, dropout_ratio=args.dropout,
                       seq_len=args.seq_len)
        model.to(device)
        # model = LstmAutoEncoder(input_layer=13, hidden_layer=5, batch_size=50)  # lstm
        # model = LstmFcAutoEncoder()  # lstm+fc模型
        loss_function = nn.MSELoss(reduction='sum')  # loss
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
        optimizer = getattr(torch.optim, args.optim)(params=model.parameters(), lr=args.lr, weight_decay=args.wd)

        epochs = 1000000
        # 开始训练
        # model.train()
        for i in range(args.epochs):
            # for batch_id in train_loader:
            model.train()
            loss_sum = 0
            pred_loss_sum = 0
            correct_sum = 0
            num_samples_iter = 0
            for batch_idx, seq in enumerate(train_loader, 1):
                # print(len(seq))
                if len(seq) == 2:
                    seq, labels = seq[0].to(device), seq[1].to(device)
                else:
                    # print(seq.shape)
                    seq = seq.to(device)
                # print(seq.shape)
                # print(labels.shape)
                # time.sleep(10)
                num_samples_iter += len(seq)
                optimizer.zero_grad()
                y_pred = model(seq) # 压缩维度：得到输出，并将维度为1的去除
                single_loss = loss_function(y_pred, seq)
                # 若想要获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
                single_loss.backward()
                loss_sum += single_loss.item()
                # # Gradient clipping
                # if clip_val is not None:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                optimizer.step()
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} \tLoss: {:.6f}'.format(
                        i, loss_sum / num_samples_iter))
            train_loss = loss_sum / len(train_loader.dataset)
            train_pred_loss = pred_loss_sum / len(train_loader.dataset)
            train_acc = round(correct_sum / len(train_loader.dataset) * 100, 2)
            acc_out_str = ''
            print(f'Train Average Loss: {train_loss}{acc_out_str}')

            model.eval()
            loss_sum = 0
            correct_sum = 0
            with torch.no_grad():
                for data in val_loader:
                    if len(data) == 2:
                        data, labels = data[0].to(device), data[1].to(device)
                    else:
                        data = data.to(device)
                    model_out = model(data)
                    single_loss = loss_function(model_out, data)
                    loss_sum += single_loss.item()
            val_loss = loss_sum / len(val_loader.dataset)
            val_acc = round(correct_sum / len(val_loader.dataset) * 100, 2)
            acc_out_str = f'; Average Accuracy: {val_acc}' if args.model_type == 'LSTMAECLF' else ''
            print(f' Validation: Average Loss: {val_loss}{acc_out_str}')
            # 每20次，输出一次前20个的结果，对比一下效果
            # if i % 20 == 0:
            #     test_data = x[:20]
            #     y_pred = model(test_data)  # 压缩维度：得到输出，并将维度为1的去除
            #     # print("TEST: ", test_data)
            #     # print("PRED: ", y_pred)
            #     print('epoch:', i)
            #     print("LOSS: ", loss_function(y_pred, test_data))
        # Save model
        torch.save(model.state_dict(), os.path.join(args.model_dir, f'model_hs={args.hidden_size}_bs={args.batch_size}'
                                                                        f'_epochs={args.epochs}_clip={args.grad_clipping}.pt'))

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





