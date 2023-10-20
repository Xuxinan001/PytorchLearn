import torch

# 1、确定参数
input_size = 4
hidden_size = 4
batch_size = 1

# 2、准备数据
index2char = ['e', 'h', 'l', 'o']  #字典
x_data = [1, 0, 2, 2, 3]  #用字典中的索引（数字）表示来表示hello
y_data = [3, 1, 2, 3, 2]  #标签：ohlol

one_hot_lookup = [[1, 0, 0, 0],  # 用来将x_data转换为one-hot向量的参照表
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]  #将x_data转换为one-hot向量
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)  #(𝒔𝒆𝒒𝑳𝒆𝒏,𝒃𝒂𝒕𝒄𝒉𝑺𝒊𝒛𝒆,𝒊𝒏𝒑𝒖𝒕𝑺𝒊𝒛𝒆)
labels = torch.LongTensor(y_data).view(-1, 1)  # (seqLen*batchSize,𝟏).计算交叉熵损失时标签不需要我们进行one-hot编码，其内部会自动进行处理
# view是改变维度，-1是根据总的以及，后面的数字，自动计算


# 3、构建模型
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):  #初始化隐藏层，需要batch_size
        return torch.zeros(self.batch_size, self.hidden_size)


net = Model(input_size, hidden_size, batch_size)

# 4、损失和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)  # Adam优化器

# 5、训练
for epoch in range(15):
    loss = 0
    optimizer.zero_grad()  #梯度清零
    hidden = net.init_hidden()  # 初始化隐藏层
    print('Predicted string:', end='')
    for input, label in zip(inputs, labels):  #每次输入一个字符，即按序列次序进行循环
        hidden = net(input, hidden)
        loss += criterion(hidden, label)  # 计算损失，不用item()，因为后面还要反向传播
        _, idx = hidden.max(dim=1)  # 选取最大值的索引
        # max返回的是最大值和索引
        # 索引是tensor类型的
        print(index2char[idx.item()], end='')  # 打印预测的字符
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    print(', Epoch [%d/15] loss: %.4f' % (epoch + 1, loss.item()))
