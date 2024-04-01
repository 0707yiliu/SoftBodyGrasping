import random

import torch
import torch.nn as nn
import torch.utils.data as Data

import argparse
import os
import time
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='LSTM_AE TOY EXAMPLE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train')
parser.add_argument('--optim', default='Adam', type=str, help='Optimizer to use')
parser.add_argument('--hidden-size', type=int, default=256, metavar='N', help='LSTM hidden state size')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--input-size', type=int, default=1, metavar='N', help='input size')
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
    """得到训练数据，这里使用随机数生成训练数据，由此导致最终结果并不好"""

    def get_tensor_from_pd(dataframe_series) -> torch.Tensor:
        return torch.tensor(data=dataframe_series.values)
    #
    # import numpy as np
    # import pandas as pd
    # from sklearn import preprocessing
    # # 生成训练数据x并做归一化后，构造成dataframe格式，再转换为tensor格式
    # df = pd.DataFrame(data=preprocessing.MinMaxScaler().fit_transform(np.random.randint(0, 10, size=(2000, 300))))
    # y = pd.Series(np.random.randint(0, 2, 2000))
    # return get_tensor_from_pd(df).float(), get_tensor_from_pd(y).float()

    data_path_dir = './grasp/data/labeled_data/'
    dirs = os.listdir(data_path_dir)
    file_item = 0
    # loading data
    for file in dirs:
        if file_item == 0:
            tac_data = np.load(data_path_dir+file)['labeled_data'].transpose()
        else:
            _tac_data = np.load(data_path_dir+file)['labeled_data'].transpose()
            tac_data = np.vstack((tac_data, _tac_data))
        file_item += 1
    # input and output for reconstruct
    _Mm = 20 # for normalization
    _CMm = 3 # for class normalization
    tac_data[:, :12] = tac_data[:, :12] / _Mm
    tac_data[:, 12] =  (tac_data[:, 12] + 1) / _CMm
    in_data = tac_data[:, :13].copy()
    out_data = tac_data[:, :13].copy()
    rate = 0.5
    for i in range(in_data.shape[0]):
        if random.random() < rate:
            in_data[i, -1] = -1
    in_df = pd.DataFrame(data=in_data)
    out_df = pd.DataFrame(data=out_data)
    y = pd.Series(np.random.randint(0, 2, tac_data.shape[0]))

    # print(df)
    # time.sleep(10)

    return get_tensor_from_pd(in_df).float(), get_tensor_from_pd(out_df).float()

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


if __name__ == '__main__':
    # 得到数据
    x, y = get_train_data() # x: input, y: output for reconstruction
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x, y),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=20,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=1,  # 多进程（multiprocess）来读数据
    )
    # 建模三件套：loss，优化，epochs
    model = LstmAutoEncoder(input_layer=13, hidden_layer=5, batch_size=20)  # lstm
    # model = LstmFcAutoEncoder()  # lstm+fc模型
    loss_function = nn.MSELoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    epochs = 10000
    # 开始训练
    model.train()
    for i in range(epochs):
        for seq, labels in train_loader:
            # print(seq.shape)
            # print(labels.shape)
            # time.sleep(10)
            optimizer.zero_grad()
            y_pred = model(seq).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            single_loss = loss_function(y_pred, labels)
            # 若想要获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
            single_loss.backward()
            optimizer.step()
            # print("Train Step:", i, " loss: ", single_loss)
        # 每20次，输出一次前20个的结果，对比一下效果
        if i % 20 == 0:
            test_data = x[:20]
            y_pred = model(test_data).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            # print("TEST: ", test_data)
            # print("PRED: ", y_pred)
            print('epoch:', i)
            print("LOSS: ", loss_function(y_pred, test_data))
    # Save model
    torch.save(model.state_dict(), os.path.join(args.model_dir, f'model_hs={args.hidden_size}_bs={args.batch_size}'
                                                                    f'_epochs={args.epochs}_clip={args.grad_clipping}.pt'))


