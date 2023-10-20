import torch

# 1、确定参数
seq_len = 5
input_size = 4
hidden_size = 4
batch_size = 1
num_layers=1

# 2、准备数据
index2char = ['e', 'h', 'l', 'o']  #字典
x_data = [1, 0, 2, 2, 3]  #用字典中的索引（数字）表示来表示hello
y_data = [3, 1, 2, 3, 2]  #标签：ohlol

one_hot_lookup = [[1, 0, 0, 0],  # 用来将x_data转换为one-hot向量的参照表
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]  #将x_data转换为one-hot向量

# 我们构造了一个大小为(seq_len, batch_size, input_size)的输入数据inputs，
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size,
                                      input_size)  #(𝒔𝒆𝒒𝑳𝒆𝒏,𝒃𝒂𝒕𝒄𝒉𝑺𝒊𝒛𝒆,𝒊𝒏𝒑𝒖𝒕𝑺𝒊𝒛𝒆)

labels = torch.LongTensor(y_data)


# 3、构建模型
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=num_layers)

    def forward(self, input):
        # 并将隐藏状态hidden初始化为全零张量，大小为(num_layers, batch_size, hidden_size)。
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        # 这步骤self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=num_layers)
        # 是文档给的内容
        out, _ = self.rnn(input, hidden)  # out: tensor of shape (seq_len, batch, hidden_size)
        # 因为要适配交叉熵
        return out.view(-1, self.hidden_size)  # 将输出的三维张量转换为二维张量,(𝒔𝒆𝒒𝑳𝒆𝒏×𝒃𝒂𝒕𝒄𝒉𝑺𝒊𝒛𝒆,𝒉𝒊𝒅𝒅𝒆𝒏𝑺𝒊𝒛𝒆)



net = Model(input_size, hidden_size, batch_size, num_layers)

# 4、损失和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)  # Adam优化器

# 5、训练
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    # idx.data是取出idx中的数据
    idx = idx.data.numpy()
    print('Predicted string: ', ''.join([index2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss: %.4f' % (epoch + 1, loss.item()))
