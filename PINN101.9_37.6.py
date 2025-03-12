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
    "D:\SOILDATA\W2017_2021_37.6_101.9_alldata.csv",
    parse_dates=['time'],
    index_col='time'
)

# 数据编码与归一化
values = data.values
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
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
reframed.drop(reframed.columns[[21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]], axis=1, inplace=True)

# 数据划分
values = reframed.values
trainNum = int(len(values) * 0.8)
train = values[:trainNum, :]
test = values[trainNum:, :]

train_X, train_y = train[:, :-4], train[:, [-4, -3, -2, -1]]
test_X, test_y = test[:, :-4], test[:, [-4, -3, -2, -1]]
print(train_y.shape)
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
output_size = train_y_tensor.shape[1]
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
    rho = 1740
    k_f = 1.539 * 24 * 60 * 60 / 1000  # (1.95*0.7+0.58*0.3)
    k_u = 2.061 * 24 * 60 * 60 / 1000  # (1.95*0.7+2.32*0.3)
    C_f = 0.76
    C_u = 0.84
    d = 1
    T_f = -1
    L = 334.560
    rho_i = 0.9

    def step_function(T, d):
        H = torch.where(T > d, torch.tensor(1.0),
                        torch.where(T < -d, torch.tensor(0.0),
                                    -(T ** 3) / (4 * (d ** 3)) + 3 * T / (4 * d) + 0.5))
        return H

    y_predict_inv = torch.tensor(scaler.inverse_transform(
        np.concatenate((outputs.detach().cpu().numpy(), x[:, 0, 4:].detach().cpu().numpy()), axis=1)
    ).astype('float32'),
        requires_grad=True
    )[:, :4]

    x_inv = torch.tensor(
        scaler.inverse_transform(
            np.concatenate((outputs.detach().cpu().numpy(), x[:, 0, 4:].detach().cpu().numpy()), axis=1)
        ).astype('float32'),
        requires_grad=True
    )[:, 4:]

    y_inv = torch.tensor(scaler.inverse_transform(
        np.concatenate((y.detach().cpu().numpy(), x[:, 0, 4:].detach().cpu().numpy()), axis=1)
    ).astype('float32'),
        requires_grad=True
    )[:, :4]

    dt = x_inv[:, 9]
    dz1 = x_inv[:, 1]
    dz2 = x_inv[:, 10]
    dz3 = x_inv[:, 11]
    dz4 = x_inv[:, 12]

    H1 = step_function(y_predict_inv[:, 0] - T_f + d / 2, d / 2)
    H2 = step_function(y_predict_inv[:, 1] - T_f + d / 2, d / 2)
    H3 = step_function(y_predict_inv[:, 2] - T_f + d / 2, d / 2)
    H4 = step_function(y_predict_inv[:, 3] - T_f + d / 2, d / 2)

    C1 = C_f + (C_u - C_f) * H1
    C2 = C_f + (C_u - C_f) * H2
    C3 = C_f + (C_u - C_f) * H3
    C4 = C_f + (C_u - C_f) * H4

    k1 = k_f + (k_u - k_f) * H1
    k2 = k_f + (k_u - k_f) * H2
    k3 = k_f + (k_u - k_f) * H3
    k4 = k_f + (k_u - k_f) * H4
    T_3_5 = y_predict_inv[:, 0]
    T_17_5 = y_predict_inv[:, 1]
    T_64 = y_predict_inv[:, 2]
    T_194_5 = y_predict_inv[:, 3]

    T_0 = T_3_5 + (0 - 3.5) * (T_17_5 - T_3_5) / (17.5 - 3.5)
    T_7 = T_3_5 + (7 - 3.5) * (T_17_5 - T_3_5) / (17.5 - 3.5)
    T_28 = T_3_5 + (28 - 3.5) * (T_17_5 - T_3_5) / (17.5 - 3.5)
    T_100 = T_28 + (100 - 28) * (T_64 - T_28) / (64 - 28)
    T_289 = T_100 + (289 - 100) * (T_194_5 - T_100) / (194.5 - 100)

    dT1 = T_7 - T_0
    dT2 = T_28 - T_7
    dT3 = T_100 - T_28
    dT4 = T_289 - T_100


    dT_dz_pred1 = dT1 / dz1
    dT_dz_pred2 = dT2 / dz2
    dT_dz_pred3 = dT3 / dz3
    dT_dz_pred4 = dT4 / dz4

    dT_dz_second_order1 = dT_dz_pred1[1:] - dT_dz_pred1[:-1] / dz1[1:]
    dT_dz_second_order2 = dT_dz_pred2[1:] - dT_dz_pred2[:-1] / dz2[1:]
    dT_dz_second_order3 = dT_dz_pred3[1:] - dT_dz_pred3[:-1] / dz3[1:]
    dT_dz_second_order4 = dT_dz_pred4[1:] - dT_dz_pred4[:-1] / dz4[1:]

    dT1_dt = (y_inv[1:, 0] - y_inv[:-1, 0]) / dt[1:]
    dT2_dt = (y_inv[1:, 1] - y_inv[:-1, 1]) / dt[1:]
    dT3_dt = (y_inv[1:, 2] - y_inv[:-1, 2]) / dt[1:]
    dT4_dt = (y_inv[1:, 3] - y_inv[:-1, 3]) / dt[1:]

    deter_seita1 = (x_inv[1:, 4] - x_inv[-1:, 4])
    deter_seita2 = (x_inv[1:, 5] - x_inv[-1:, 5])
    deter_seita3 = (x_inv[1:, 6] - x_inv[-1:, 6])
    deter_seita4 = (x_inv[1:, 7] - x_inv[-1:, 7])

    dT_dz_second_order1_L = k1[1:] * dT_dz_second_order1 - rho_i * L * deter_seita1
    dT_dz_second_order2_L = k2[1:] * dT_dz_second_order2 - rho_i * L * deter_seita2
    dT_dz_second_order3_L = k3[1:] * dT_dz_second_order3 - rho_i * L * deter_seita3
    dT_dz_second_order4_L = k4[1:] * dT_dz_second_order4 - rho_i * L * deter_seita4

    R1 = rho * 0.07 * C1[1:] * dT1_dt - dT_dz_second_order1_L
    R2 = rho * 0.21 * C2[1:] * dT2_dt - dT_dz_second_order2_L
    R3 = rho * 0.72 * C3[1:] * dT3_dt - dT_dz_second_order3_L
    R4 = rho * 1.89 * C4[1:] * dT4_dt - dT_dz_second_order4_L

    pde_loss = torch.mean(R1 ** 2) + torch.mean(R2 ** 2) + torch.mean(R3 ** 2) + torch.mean(R4 ** 2)
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
    np.concatenate((predictions, test_X[:, 0, 4:]), axis=1)
)[:, :4]

# 获取真实值，并逆归一化
test_y = test_y_tensor.detach().cpu().numpy()
inv_test_y = scaler.inverse_transform(
    np.concatenate((test_y, test_X[:, 0, 4:]), axis=1)
)[:, :4]

# 替换原始数据中的相应列
data.iloc[-len(inv_predictions):, 0] = inv_predictions[:, 0]
data.iloc[-len(inv_predictions):, 1] = inv_predictions[:, 1]
data.iloc[-len(inv_predictions):, 2] = inv_predictions[:, 2]
data.iloc[-len(inv_predictions):, 3] = inv_predictions[:, 3]

# 创建文件夹
# 创建文件夹
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 保存预测值和真实值到CSV文件
results_df = pd.DataFrame(data={'Predicted_T1': inv_predictions[:, 0], 'Actual_T1': inv_test_y[:, 0],
                                'Predicted_T2': inv_predictions[:, 1], 'Actual_T2': inv_test_y[:, 1],
                                'Predicted_T3': inv_predictions[:, 2], 'Actual_T3': inv_test_y[:, 2],
                                'Predicted_T4': inv_predictions[:, 3], 'Actual_T4': inv_test_y[:, 3]})
results_df.to_csv(os.path.join(results_dir, 'predictions_vs_actuals(PINN101.9_37.6).csv'), index=False)

# 替换原始数据中的相应列
original_data = pd.read_csv("D:\SOILDATA\W2017_2021_37.6_101.9_alldata.csv", parse_dates=['time'], index_col='time')

# 假设原始数据的相应列名为 'stl1(k)', 'stl2(k)', 'stl3(k)', 'stl4(k)'
original_data.loc[original_data.index[-len(inv_test_y[:, 0]):], 'stl1Mean'] = inv_predictions[:, 0]
original_data.loc[original_data.index[-len(inv_test_y[:, 1]):], 'stl2Mean'] = inv_predictions[:, 1]
original_data.loc[original_data.index[-len(inv_test_y[:, 2]):], 'stl3Mean'] = inv_predictions[:, 2]
original_data.loc[original_data.index[-len(inv_test_y[:, 3]):], 'stl4Mean'] = inv_predictions[:, 3]

# 保存替换后的数据到新的CSV文件
output_file = 'D:/SOILDATA/W101.9_37.6all_data_with_predictions_PINN.csv'
original_data.to_csv(output_file)

print(f"预测结果已保存到 {output_file}")

# 可视化
plt.figure(figsize=(15, 10))

for i in range(inv_predictions.shape[1]):
    plt.subplot(2, 2, i + 1)
    plt.plot(inv_predictions[:, i], label='Predicted')
    plt.plot(inv_test_y[:, i], label='Actual')
    plt.title(f'Temperature T{i + 1}')
    plt.xlabel('Time')
    plt.ylabel('Temperature (℃)')
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'predictions_vs_actuals(PINN101.9_37.6).png'))
plt.show()