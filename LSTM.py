import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size,
                            num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]

class MultiStockPredictor:
    def __init__(self, stocks, learning_rate=0.001):
        self.stocks = stocks
        self.models = {stock: LSTMModel() for stock in stocks}
        self.optimizers = {stock: torch.optim.Adam(
            self.models[stock].parameters(), lr=learning_rate) for stock in stocks}
        self.loss_fn = nn.MSELoss()

    def train(self, train_data, epochs):
        for epoch in range(epochs):
            for stock, data in train_data.items():
                dataloader = DataLoader(data, batch_size=32, shuffle=True)
                for i, (inputs, targets) in enumerate(dataloader):
                    self.optimizers[stock].zero_grad()
                    outputs = self.models[stock](inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    self.optimizers[stock].step()

    def save_models(self, save_dir):
        for stock, model in self.models.items():
            torch.save(model.state_dict(), f"{save_dir}/{stock}.pt")

    def load_model(self, stock, load_dir):
        self.models[stock].load_state_dict(
            torch.load(f"{load_dir}/{stock}.pt"))

    def predict(self, stock, data):
        model = self.models[stock]
        model.eval()
        with torch.no_grad():
            inputs = torch.as_tensor(data)
            outputs = model(inputs)
        return outputs.numpy()


class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu


def prepare_data_x(x, window_size):
    # perform windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(
        x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]


def prepare_data_y(x, window_size):
    # # perform simple moving average
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size

    # use the next day as label
    output = x[window_size:]
    return output


def prepare_data(normalized_data_close_price, config, plot=False):
    data_x, data_x_unseen = prepare_data_x(
        normalized_data_close_price, window_size=config["data"]["window_size"])
    data_y = prepare_data_y(normalized_data_close_price,
                            window_size=config["data"]["window_size"])

    # split dataset

    split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    #     # prepare data for plotting

    #     to_plot_data_y_train = np.zeros(num_data_points)
    #     to_plot_data_y_val = np.zeros(num_data_points)

    #     to_plot_data_y_train[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(data_y_train)
    #     to_plot_data_y_val[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(data_y_val)

    #     to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
    #     to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)

    #     ## plots

    #     fig = figure(figsize=(25, 5), dpi=80)
    #     fig.patch.set_facecolor((1.0, 1.0, 1.0))
    #     plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", color=config["plots"]["color_train"])
    #     plt.plot(data_date, to_plot_data_y_val, label="Prices (validation)", color=config["plots"]["color_val"])
    #     xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    #     x = np.arange(0,len(xticks))
    #     plt.xticks(x, xticks, rotation='vertical')
    #     plt.title("Daily close prices for " + config["alpha_vantage"]["symbol"] + " - showing training and validation data")
    #     plt.grid(b=None, which='major', axis='y', linestyle='--')
    #     plt.legend()
    #     plt.show()

    return split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen


config = {
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "plots": {
        "show_plots": True,
        "xticks_interval": 90,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1,  # since we are only using 1 feature, close price
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 30,
    }
}


# End definitions
