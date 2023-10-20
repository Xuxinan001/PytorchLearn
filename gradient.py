import matplotlib.pyplot as plt

# prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# initial guess of weight
w = 1.0

# define the model linear model y = w*x
def forward(x):
    return x*w

# define the cost function MSE
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)

# define the gradient function gd
'''
如何找到梯度下降的方向：用目标函数对权重w求导数可找到梯度的变化方向
'''
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2*x*(x*w - y)
    return grad / len(xs)
epoch_list = []
cost_list = []

print("Predict (before training)", 4, forward(4))
for epoch in range(100):
    # 计算损失值
    cost_val = cost(x_data, y_data)
    # 计算梯度变化值
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print('Epoch:', epoch, "w=", "%.2f" %w, "loss=", "%.2f" %cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)


print("Predict (after training)", 4, forward(4))
plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
