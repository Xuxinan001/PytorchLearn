import matplotlib.pyplot as plt
import torch

# PyTorch实现反向传播Backward
# 1.计算损失 2.Backward 3.梯度下降继续更新
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = torch.Tensor([1.0])     # 初始化权重
w.requires_grad = True      # 表明w需要计算梯度

# define the linear model
def forward(x):
    return x * w            # 这里会构建计算图 将x转化成Tensor

# 损失函数的求解，构建计算图，并不是乘法或者乘方运算
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# 打印学习之前的值，.item()表示输出张量的值
print("Predict (before training)", 4, forward(4).item())

learning_rate = 0.01
epoch_list = []
loss_list = []

# 训练过程
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)      # forward前馈
        l.backward()        # 成员函数backward()向后传播 自动求出所有需要的梯度
        print('\tgrad:', x, y, w.grad.item())   # 将梯度存到w之中,随后释放计算图 item()可以直接将梯度变成标量
        w.data = w.data - learning_rate * w.grad.data    # w的grad也是张量，计算应该取data 不去建立计算图
        w.grad.data.zero_()                     # 释放data
        epoch_list.append(epoch)
        loss_list.append(l.item())

    print("progress:", epoch, l.item())

print("predict (after training)", 4, forward(4).item())

# 绘制可视化
plt.plot(epoch_list, loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
