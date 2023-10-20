import torch

# 1、确定参数
num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2
batch_size = 1
seq_len = 5

# 2、准备数据
index2char = ['e', 'h', 'l', 'o']  #字典
x_data = [[1, 0, 2, 2, 3]]  # (batch_size, seq_len) 用字典中的索引（数字）表示来表示hello
y_data = [3, 1, 2, 3, 2]  #  (batch_size * seq_len) 标签：ohlol

inputs = torch.LongTensor(x_data)  # (batch_size, seq_len)
labels = torch.LongTensor(y_data)  # (batch_size * seq_len)


# 3、构建模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(num_class, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                                batch_first=True)
        # 改变维度
        self.fc = torch.nn.Linear(hidden_size, num_class)


    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)  # (num_layers, batch_size, hidden_size)
        x = self.emb(x)  # 返回(batch_size, seq_len, embedding_size)
        x, _ = self.rnn(x, hidden)  # 返回(batch_size, seq_len, hidden_size)
        x = self.fc(x)  # 返回(batch_size, seq_len, num_class)
        return x.view(-1, num_class)  # (batch_size * seq_len, num_class)


net = Model()

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
    idx = idx.data.numpy()
    print('Predicted string: ', ''.join([index2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss: %.4f' % (epoch + 1, loss.item()))
