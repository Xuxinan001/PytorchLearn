import torch

# mini-batch需要的数据是Tensor
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

# Design Model  重点目标在于构造计算图
"""
所有模型都要继承自Model
最少实现两个成员方法
    构造函数 初始化：__init__()
    前馈：forward()
Model自动实现backward
可以在Functions中构建自己的计算块
"""
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)      # 构造了一个包含 w和 b的对象

    def forward(self, x):
        y_pred = self.linear(x)                 # linear成为了可调用的对象 直接计算forward
        return y_pred

model = LinearModel()               # 创建类的实例

# 3.Construct Loss(MSE (y_pred - y)**2 ) and Optimizer
# 构造计算图就需要集成Model模块
criterion = torch.nn.MSELoss(size_average=False)    #需要的参数是y_pred和y
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# 4.Training Cycle
for epoch in range(100):
    y_pred = model(x_data)          # Forward
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()           # 梯度清零
    loss.backward()                 # 反馈
    optimizer.step()                # 更新

#Output weight and bias
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

#Test Model
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
