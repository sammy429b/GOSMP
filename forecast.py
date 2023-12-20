from .LSTM import *


today_price_df = pd.read_csv("recv_dupli.csv")

columns_to_drop = ['companyName', 'change', 'pChange', 'updatedOn',
                   'previousClose', 'previousOpen', 'dayHigh', 'dayLow',
                   'totalTradedValue', 'totalTradedQuantity', 'buy', 'sell']

today_price_df.drop(columns=columns_to_drop, inplace=True)


def convert_to_int(value):
    try:
        # Split the value into numeric and text parts
        parts = value.strip().split()

        # Extract the numeric part and remove commas
        numeric_value = float(parts[0].replace(',', ''))

        # If the value contains 'Lakh', multiply by 100,000
        if 'Lakh' in parts:
            numeric_value *= 100000

        # If the value contains 'Cr.', multiply by 10,000,000
        elif 'Cr.' in parts:
            numeric_value *= 10000000

        # Round to the nearest integer
        result = int(round(numeric_value))

        return result

    except ValueError:
        # Return None if the conversion fails
        return None


# Apply the conversion function to each column
today_price_df['2WeekAvgQuantity'] = today_price_df['2WeekAvgQuantity'].apply(
    convert_to_int)
today_price_df['marketCapFull'] = today_price_df['marketCapFull'].apply(
    convert_to_int)
today_price_df['marketCapFreeFloat'] = today_price_df['marketCapFreeFloat'].apply(
    convert_to_int)


# Assuming timed_df is your DataFrame indexed by date
orig_df = pd.read_csv("close_dupli.csv")
timed_df = orig_df.fillna(0)

timed_df['Date'] = pd.to_datetime(timed_df['Date'])
timed_df.set_index('Date', inplace=True)
# Set the desired start and end dates
start_date = '2022-11-01'
end_date = '2023-01-01'

# Use loc to select rows within the specified date range
# timed_df = timed_df.loc[start_date:end_date]

# Set the desired number of columns to keep
num_columns_to_keep = 5  # Replace with your desired number

# Use iloc to select the first num_columns_to_keep columns
timed_df = timed_df.iloc[:, :num_columns_to_keep]
timed_df


# normalize
scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(np.array(timed_df))

split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen = prepare_data(
    normalized_data_close_price, config, plot=config["plots"]["show_plots"])


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        x = np.expand_dims(x, 2)
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)


model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"],
                  num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = model.to(config["training"]["device"])


def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr


# create `DataLoader`
train_dataloader = DataLoader(
    dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(
    dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

# define optimizer, scheduler and loss function
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(
), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

# begin training
for epoch in range(config["training"]["num_epoch"]):
    loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
    loss_val, lr_val = run_epoch(val_dataloader)
    scheduler.step()

    print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
          .format(epoch+1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))


model.eval()

x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(
    0).unsqueeze(2)  # this is the data type and shape required, [batch, sequence, feature]
prediction = model(x)
prediction = prediction.cpu().detach().numpy()
prediction = scaler.inverse_transform(prediction)[0]
