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
reframed.drop(reframed.columns[[17, 18, 19, 20, 21, 22, 23, 24, 29, 30, 31, 32, 33]], axis=1, inplace=True)

# 数据划分
values = reframed.values
trainNum = int(len(values) * 0.8)
train = values[:trainNum, :]
test = values[trainNum:, :]

train_X, train_y = train[:, :-4], train[:, [-4, -3, -2, -1]]
test_X, test_y = test[:, :-4], test[:, [-4, -3, -2, -1]]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

train_X_tensor = torch.tensor(train_X, dtype=torch.float32, requires_grad=True)
train_y_tensor = torch.tensor(train_y, dtype=torch.float32, requires_grad=True)
test_X_tensor = torch.tensor(test_X, dtype=torch.float32, requires_grad=True)
test_y_tensor = torch.tensor(test_y, dtype=torch.float32, requires_grad=True)

# 检查 requires_grad
#print("train_X_tensor.requires_grad:", train_X_tensor.requires_grad)
#print("train_y_tensor.requires_grad:", train_y_tensor.requires_grad)
#print("test_X_tensor.requires_grad:", test_X_tensor.requires_grad)
#print("test_y_tensor.requires_grad:", test_y_tensor.requires_grad)


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

    rho_i = 900
    rho_u = 1000

    # Convert to NumPy, perform inverse transform, then convert back to PyTorch tensor with requires_grad=True
    y_predict_inv = torch.tensor(scaler.inverse_transform(
        np.concatenate((x[:, 0, :8].detach().cpu().numpy(), outputs.detach().cpu().numpy(), x[:, 0, -5:].detach().cpu().numpy()), axis=1)
    ).astype('float32'),
        requires_grad=True
    )[:, 8:12]

    x_inv = torch.tensor(scaler.inverse_transform(
        np.concatenate((x[:, 0, :8].detach().cpu().numpy(), outputs.detach().cpu().numpy(), x[:, 0, -5:].detach().cpu().numpy()), axis=1)
    ).astype('float32'),
        requires_grad=True
    )
    # 分离张量，不再跟踪梯度，然后转换为 NumPy 数组
    x_inv = x_inv.detach().cpu().numpy()

    # 拼接操作
    x_inv = torch.tensor(np.concatenate((x_inv[:, :8], x_inv[:, -5:]), axis=1).astype('float32'),
        requires_grad=True
    )

    y_inv = torch.tensor(scaler.inverse_transform(
        np.concatenate((y.detach().cpu().numpy(), x[:, 0, 4:].detach().cpu().numpy()), axis=1)
    ).astype('float32'),
        requires_grad=True
    )[:, :4]

    dt = x_inv[:, 9]
    dz1 = x_inv[:, 5]
    dz2 = x_inv[:, 10]
    dz3 = x_inv[:, 11]
    dz4 = x_inv[:, 12]
    T1 = x_inv[:, 0]
    T2 = x_inv[:, 1]
    T3 = x_inv[:, 2]
    T4 = x_inv[:, 3]

    seita_u1 = y_inv[:, 0]
    seita_u2 = y_inv[:, 1]
    seita_u3 = y_inv[:, 2]
    seita_u4 = y_inv[:, 3]
    a = 0.17
    b = -0.5
    I1 = 10 ** (10 * y_predict_inv[:, 0])
    k1 = 1.6438 * (y_predict_inv[:, 0] ** 0.7818)
    k1 = torch.where(T1 > 0, k1, k1 / I1)

    I2 = 10**(10*y_predict_inv[:, 1])
    k2 = 1.6438*(y_predict_inv[:, 1]**0.7818)/I2
    k2 = torch.where(T2 > 0, k2, k2 / I2)

    I3 = 10**(10*y_predict_inv[:, 2])
    k3 = 1.6438*(y_predict_inv[:, 2]**0.7818)/I3
    k3 = torch.where(T3 > 0, k3, k3 / I3)

    I4 = 10**(10*y_predict_inv[:, 3])
    k4 = 1.6438*(y_predict_inv[:, 3]**0.7818)/I4
    k4 = torch.where(T4 > 0, k4, k4 / I4)

    a_0 = 0.15
    m = 0.7
    seita_m = 0.42
    seita_n = 0.04
    s_r1 = (y_predict_inv[:, 0] - seita_n) / (seita_m - seita_n)
    s_r2 = (y_predict_inv[:, 1] - seita_n) / (seita_m - seita_n)
    s_r3 = (y_predict_inv[:, 2] - seita_n) / (seita_m - seita_n)
    s_r4 = (y_predict_inv[:, 3] - seita_n) / (seita_m - seita_n)
    fai_m1 = ((s_r1 ** (-1 / m) - 1) ** (1 - m)) / a_0
    fai_m2 = ((s_r2 ** (-1 / m) - 1) ** (1 - m)) / a_0
    fai_m3 = ((s_r3 ** (-1 / m) - 1) ** (1 - m)) / a_0
    fai_m4 = ((s_r4 ** (-1 / m) - 1) ** (1 - m)) / a_0
    fai_g1 = 1000 * 0.07 * 9.8 * 2.82
    fai_g2 = 1000 * 0.21 * 9.8 * 2.61
    fai_g3 = 1000 * 0.72 * 9.8 * 1.89
    fai_g4 = 1000 * 1.89 * 9.8 * 1
    fai1 = fai_m1 + fai_g1
    fai2 = fai_m2 + fai_g2
    fai3 = fai_m3 + fai_g3
    fai4 = fai_m4 + fai_g4

    deter_seitau1 = (seita_u1[1:] - seita_u1[:-1])
    deter_seitau2 = (seita_u1[1:] - seita_u1[:-1])
    deter_seitau3 = (seita_u1[1:] - seita_u1[:-1])
    deter_seitau4 = (seita_u1[1:] - seita_u1[:-1])
    Ddeter_seitau1_Dt = (deter_seitau1) / dt[1:]
    Ddeter_seitau2_Dt = (deter_seitau2) / dt[1:]
    Ddeter_seitau3_Dt = (deter_seitau3) / dt[1:]
    Ddeter_seitau4_Dt = (deter_seitau4) / dt[1:]

    T_3_5 = x_inv[:, 0]
    T_17_5 = x_inv[:, 1]
    T_64 = x_inv[:, 2]
    T_194_5 = x_inv[:, 3]
    seita_i1 = seita_u1 - a * (abs(T_3_5)) ** b
    seita_i2 = seita_u2 - a * (abs(T_17_5)) ** b
    seita_i3 = seita_u3 - a * (abs(T_64)) ** b
    seita_i4 = seita_u4 - a * (abs(T_194_5)) ** b
    Dseita_i1_Dt = (rho_i * (seita_i1[1:] - seita_i1[:-1])) / (rho_u * dt[1:])
    Dseita_i2_Dt = (rho_i * (seita_i2[1:] - seita_i2[:-1])) / (rho_u * dt[1:])
    Dseita_i3_Dt = (rho_i * (seita_i3[1:] - seita_i3[:-1])) / (rho_u * dt[1:])
    Dseita_i4_Dt = (rho_i * (seita_i4[1:] - seita_i4[:-1])) / (rho_u * dt[1:])
    q_z1 = (-k1[1:]) * ((fai1[1:] - fai1[:-1]) / dz1[1:])
    q_z2 = (-k2[1:]) * ((fai2[1:] - fai2[:-1]) / dz2[1:])
    q_z3 = (-k3[1:]) * ((fai3[1:] - fai3[:-1]) / dz3[1:])
    q_z4 = (-k4[1:]) * ((fai4[1:] - fai4[:-1]) / dz4[1:])
    Dq_z1_D_z = (q_z1[1:] - q_z1[:-1]) / dz1[2:]
    Dq_z2_D_z = (q_z2[1:] - q_z2[:-1]) / dz2[2:]
    Dq_z3_D_z = (q_z3[1:] - q_z3[:-1]) / dz3[2:]
    Dq_z4_D_z = (q_z4[1:] - q_z4[:-1]) / dz4[2:]


    # Calculate residuals for each PDE
    R1 = Ddeter_seitau1_Dt[1:] + Dseita_i1_Dt[1:] - Dq_z1_D_z
    R2 = Ddeter_seitau2_Dt[1:] + Dseita_i2_Dt[1:] - Dq_z2_D_z
    R3 = Ddeter_seitau3_Dt[1:] + Dseita_i3_Dt[1:] - Dq_z3_D_z
    R4 = Ddeter_seitau4_Dt[1:] + Dseita_i4_Dt[1:] - Dq_z4_D_z

    # Calculate the loss
    pde_loss = torch.mean(R1**2) + torch.mean(R2**2) + torch.mean(R3**2) + torch.mean(R4**2)
    return pde_loss
#这里的扩散系数里面没区分冻结区和未冻区，后面要改一下


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
    np.concatenate((test_X[:, 0, :8], predictions, test_X[:, 0, -5:]), axis=1)
)[:, 8:12]

# 获取真实值，并逆归一化
test_y = test_y_tensor.detach().cpu().numpy()
inv_test_y = scaler.inverse_transform(
    np.concatenate((test_X[:, 0, :8], test_y, test_X[:, 0, -5:]), axis=1)
)[:, 8:12]

# 创建文件夹
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 保存预测值和真实值到CSV文件
results_df = pd.DataFrame(data={'Predicted_W1': inv_predictions[:, 0], 'Actual_W1': inv_test_y[:, 0],
                                'Predicted_W2': inv_predictions[:, 1], 'Actual_W2': inv_test_y[:, 1],
                                'Predicted_W3': inv_predictions[:, 2], 'Actual_W3': inv_test_y[:, 2],
                                'Predicted_W4': inv_predictions[:, 3], 'Actual_W4': inv_test_y[:, 3]})
results_df.to_csv(os.path.join(results_dir, 'predictions_vs_actuals_water(PINN).csv'), index=False)

# 可视化
plt.figure(figsize=(15, 10))

for i in range(inv_predictions.shape[1]):
    plt.subplot(2, 2, i + 1)
    plt.plot(inv_predictions[:, i], label='Predicted')
    plt.plot(inv_test_y[:, i], label='Actual')
    plt.title(f' Moisture Content T{i + 1}')
    plt.xlabel('Time')
    plt.ylabel('Moisture Content (%)')
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'predictions_vs_actuals_water(PINN).png'))
plt.show()