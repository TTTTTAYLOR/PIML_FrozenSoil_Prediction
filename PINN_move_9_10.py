import os
import numpy as np
import pandas as pd
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 数据加载与预处理
data = pd.read_csv(
    "D:\SOILDATA\W2017_2021_101.9_37.6_alldata2.csv",
    parse_dates=['Date'],
    index_col='Date'
)

# 数据编码与归一化
values = data.values
encoder = LabelEncoder()
values[:, 5] = encoder.fit_transform(values[:, 5])
values = values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]], axis=1, inplace=True)

# 数据划分
values = reframed.values
trainNum = int(len(values) * 0.8)
train = values[:trainNum, :]
test = values[trainNum:, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
print(train_y.shape)
train_y = train_y.reshape(-1, 1)
print(train_y.shape)
test_y = test_y.reshape(-1, 1)
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

train_X_tensor = torch.tensor(train_X, dtype=torch.float32, requires_grad=True)
train_y_tensor = torch.tensor(train_y, dtype=torch.float32, requires_grad=True)
test_X_tensor = torch.tensor(test_X, dtype=torch.float32, requires_grad=True)
test_y_tensor = torch.tensor(test_y, dtype=torch.float32, requires_grad=True)


# 定义 LSTM PINN 模型
class LSTM_PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM_PINN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


input_size = train_X_tensor.shape[2]
hidden_size = 64
output_size = 1
num_layers = 2

model = LSTM_PINN(input_size, hidden_size, output_size, num_layers)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def safe_grad(outputs, inputs, grad_outputs, allow_unused=True):
    inputs = inputs.clone().detach().requires_grad_(True)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs, create_graph=True, allow_unused=allow_unused)
    if grads[0] is not None:
        return grads[0].detach().clone().requires_grad_(True)
    else:
        return torch.zeros_like(inputs, requires_grad=True)


def pde_loss(outputs, x, y):
    T_01 = -1.83
    T_02 = -0.93
    T_03 = 1.4
    T_04 = 5
    T_f = -1
    e = 0.37
    arf_i = 0.00005
    arf_w = 0.000214
    arf_s = 0.00002
    a = 0.17
    b = -0.5



    y_predict_inv = torch.tensor(scaler.inverse_transform(
        np.concatenate((outputs.detach().cpu().numpy(), x[:, 0, 1:].detach().cpu().numpy()), axis=1)
    ).astype('float32'),
        requires_grad=True
    )[:, :1]

    x_inv = torch.tensor(
        scaler.inverse_transform(
            np.concatenate((outputs.detach().cpu().numpy(), x[:, 0, 1:].detach().cpu().numpy()), axis=1)
        ).astype('float32'),
        requires_grad=True
    )[:, 1:]

    serta_01 = 0.37128
    serta_02 = 0.36684
    serta_03 = 0.36345
    serta_04 = 0.39895

    serta_u1 = x_inv[:, 8]
    serta_u2 = x_inv[:, 9]
    serta_u3 = x_inv[:, 10]
    serta_u4 = x_inv[:, 11]

    T1 = x_inv[:, 0]
    T2 = x_inv[:, 1]
    T3 = x_inv[:, 2]
    T4 = x_inv[:, 3]
    serta_i1 = torch.where(T1 < 0, serta_01 - a * (abs(T1) ** b), 0)
    serta_i2 = torch.where(T2 < 0, serta_02 - a * (abs(T2) ** b), 0)
    serta_i3 = torch.where(T3 < 0, serta_03 - a * (abs(T3) ** b), 0)
    serta_i4 = torch.where(T4 < 0, serta_04 - a * (abs(T4) ** b), 0)

    derta_serta = (x_inv[:, 12] - x_inv[:, -1]) * 2890
    displace_f1 = (0.09 * (serta_01 - serta_u1) + serta_01 - e) * 70
    displace_f2 = (0.09 * (serta_02 - serta_u2) + serta_02 - e) * 210
    displace_f3 = (0.09 * (serta_03 - serta_u3) + serta_03 - e) * 720
    displace_f4 = (0.09 * (serta_04 - serta_u4) + serta_04 - e) * 1890
    displace_f = (displace_f1 + displace_f2 + displace_f3 + displace_f4) + (1.09 * derta_serta)

    displace_T1 = 3 * arf_s * 0.49 * (T1 - T_01) + 3 * arf_w * serta_u1 * (T1 - T_01) + 3 * arf_i * serta_i1 * (T1 - T_f)
    displace_T2 = 3 * arf_s * 0.49 * (T2 - T_02) + 3 * arf_w * serta_u2 * (T2 - T_02) + 3 * arf_i * serta_i2 * (T2 - T_f)
    displace_T3 = 3 * arf_s * 0.49 * (T3 - T_03) + 3 * arf_w * serta_u3 * (T3 - T_03) + 3 * arf_i * serta_i3 * (T3 - T_f)
    displace_T4 = 3 * arf_s * 0.49 * (T4 - T_04) + 3 * arf_w * serta_u4 * (T4 - T_04) + 3 * arf_i * serta_i4 * (T4 - T_f)
    displace_T = displace_T1 + displace_T2 + displace_T3 + displace_T4
    displace_T = displace_T.reshape(-1, 1)
    displace_f = displace_f.reshape(-1, 1)
    displacement = displace_T + displace_f

    R1 = y_predict_inv - displacement
    pde_loss = torch.mean(R1**2)

    return pde_loss


num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    outputs = model(train_X_tensor)
    loss = criterion(outputs, train_y_tensor) + pde_loss(outputs, train_X_tensor, train_y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = model(test_X_tensor)

# 将预测结果转换为NumPy数组
predictions = predictions.cpu().numpy()

# 逆归一化
inv_predictions = scaler.inverse_transform(
    np.concatenate((predictions, test_X[:, 0, 1:]), axis=1)
)[:, :1]

# 获取真实值，并逆归一化
test_y = test_y_tensor.detach().cpu().numpy()
inv_test_y = scaler.inverse_transform(
    np.concatenate((test_y, test_X[:, 0, 1:]), axis=1)
)[:, :1]

# 替换原始数据中的相应列
data.iloc[-len(inv_predictions):, 0] = inv_predictions[:, 0]

# 创建文件夹
# 创建文件夹
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 保存预测值和真实值到CSV文件
results_df = pd.DataFrame(data={'Predicted_diplace': inv_predictions[:, 0], 'Actual_displace': inv_test_y[:, 0]})
results_df.to_csv(os.path.join(results_dir, 'predictions_vs_actuals(PINN_move).csv'), index=False)

# 替换原始数据中的相应列
original_data = pd.read_csv("D:\SOILDATA\W2017_2021_101.9_37.6_alldata2.csv", parse_dates=['Date'], index_col='Date')

# 假设原始数据的相应列名为 'stl1(k)', 'stl2(k)', 'stl3(k)', 'stl4(k)'
original_data.loc[original_data.index[-len(inv_test_y[:, 0]):], 'Mean_Value'] = inv_predictions[:, 0]

# 保存替换后的数据到新的CSV文件
output_file = "D:\SOILDATA\W2017_2021_101.9_37.6_alldata2_PINN.csv"
original_data.to_csv(output_file)

print(f"预测结果已保存到 {output_file}")

# 可视化
plt.figure(figsize=(15, 10))

plt.plot(inv_predictions, label='Predicted', color='orange')
plt.plot(inv_test_y, label='Actual', color='blue')
plt.title('displacement_PINN')
plt.xlabel('Time')
plt.ylabel('displacement (mm)')
plt.legend()

plt.savefig(os.path.join(results_dir, 'predictions_vs_actuals_displacement(PINN101.9_37.6).png'))
plt.show()
