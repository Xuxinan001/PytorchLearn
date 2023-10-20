import matplotlib.pyplot as plt
# prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# initial guess of weight
w = 1.0

# define the model linear model y = w*x
def forward(x):
    return x * w

# define the cost function MSE
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# define the gradient function gd
'''
如何找到梯度下降的方向：用目标函数对权重w求导数可找到梯度的变化方向
'''
def gradient(x, y):
    return 2 * x * (x * w - y)

epoch_list = []
loss_list = []

print("Predict (before training)", 4, forward(4))

# 对每一个样本的梯度进行更新
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        loss_val = loss(x, y)
        grad_val = gradient(x, y)
        w = w - 0.01 * grad_val
        print("Epoch: ", epoch, "w: ", "%.2lf"%w, "loss: ", "%.2lf"%loss_val)
    epoch_list.append(epoch)
    loss_list.append(loss_val)

print("Predict (after training)", 4, forward(4))
plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
