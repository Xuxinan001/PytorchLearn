import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
def forward(x,b):
    return x * w+b
def loss(x, y,b):
    y_pred = forward(x,b)
    return (y_pred - y) * (y_pred - y)
w_list = []
b_list=[]
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(0.0,2.0,0.5):
        print('w=', w)
        print('b=',b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val,b)
            loss_val = loss(x_val, y_val,b)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print('MSE=', l_sum / 3)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum / 3)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 传入w_list、b_list和mse_list作为x、y、z坐标数据来绘制三维表面
ax.plot_trisurf(w_list, b_list, mse_list, cmap='viridis')

# 设置坐标轴标签
ax.set_xlabel('W')
ax.set_ylabel('B')
ax.set_zlabel('MSE')

# 显示图形
plt.show()