# 导入包和版本查询
import torch.nn as nn
import torchvision
import torch.backends.cudnn
import numpy as np
import os

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))

# 只需要一张显卡
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 指定多张显卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 若想每次都能得到相同的随机数，每次产生随机数之前，都需要调用一次seed()
np.random.seed(0)
# 为CPU中设置种子，生成随机数
torch.manual_seed(0)
# 为特定GPU设置种子，生成随机数
torch.cuda.manual_seed(0)
# 为所有GPU设置种子，生成随机数
torch.cuda.manual_seed_all(0)

"""
Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异。如果想要避免这种结果波动。
"""
torch.backends.cudnn.deterministic = True

"""
为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
适用场景是网络结构固定（不是动态变化的）。
网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的
"""
torch.backends.cudnn.benchmark = True
# 清除显存
torch.cuda.empty_cache()

tensor = torch.randn(3, 4, 5)
print(tensor.type())  # 数据类型
print(tensor.size())  # 张量的shape，是个元组
print(tensor.dim())  # 维度的数量

# 设置默认类型，pytorch中的FloatTensor远远快于DoubleTensor
torch.set_default_tensor_type(torch.FloatTensor)

# 类型转换
tensor = tensor.cuda()
tensor = tensor.cpu()
tensor = tensor.float()
tensor = tensor.long()

# torch.Tensor与np.ndarray转换
# 除了CharTensor，其他所有CPU上的张量都支持转换为numpy格式然后再转换回来。
ndarray = tensor.cpu().numpy()
tensor_1 = torch.from_numpy(ndarray).float()
tensor_2 = torch.from_numpy(ndarray.copy()).float()  # If ndarray has negative stride.

# 在将卷积层输入全连接层的情况下通常需要对张量做形变处理，
# 相比torch.view，torch.reshape可以自动处理输入张量不连续的情况。
tensor = torch.rand(2, 3, 4)
shape = (6, 4)
tensor = torch.reshape(tensor, shape)
print(tensor.shape)
'''
注意torch.cat和torch.stack的区别在于torch.cat沿着给定的维度拼接，
而torch.stack会新增一维。例如当参数是3个10x5的张量，torch.cat的结果是30x5的张量，
而torch.stack的结果是3x10x5的张量。
'''
T1 = torch.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

T2 = torch.tensor([[10, 20, 30],
                   [40, 50, 60],
                   [70, 80, 90]])
tensor = torch.cat((T1, T2), dim=0)
print(tensor)
tensor = torch.stack((T1, T2), dim=0)
print(tensor)

# 矩阵乘法
tensor1 = torch.rand(2, 3)
print(tensor1)
tensor2 = torch.rand(3, 2)
tensor3 = torch.rand(2, 2, 3)
tensor4 = torch.rand(2, 3, 4)

# Matrix multiplcation: (m*n) * (n*p) * -> (m*p).
result = torch.mm(tensor1, tensor2)
print(result.shape)

# Batch matrix multiplication: (b*m*n) * (b*n*p) -> (b*m*p)
result = torch.bmm(tensor3, tensor4)
print(result.shape)

# Element-wise multiplication.
result = T1 * T1

# 计算两组数据之间的两两欧式距离
# 利用broadcast机制
dist = torch.sqrt(torch.sum((T1[:, None, :] - T1) ** 2, dim=2))
print(dist)


# 模型定义和操作
# convolutional neural network (2 convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet(10).to(device)

# 计算模型整体参数量
num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
print(num_parameters)

# 可以通过model.state_dict()或者model.named_parameters()函数查看现在的全部可训练参数.
# print(model.state_dict())
# print(list(model.named_parameters()))


# 部分层使用预训练模型
# 注意如果保存的模型是 torch.nn.DataParallel，则当前的模型也需要是
model.load_state_dict(torch.load('model.pth'), strict=False)
# 将在 GPU 保存的模型加载到 CPU
model.load_state_dict(torch.load('model.pth', map_location='cpu'))

# 分类模型训练代码
# Loss and optimizer
train_loader = ''
learning_rate = 0.001
num_epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 分类模型测试代码
# Test the model
model.eval()  # eval mode(batch norm uses moving mean/varianceinstead of mini-batch mean/variance)
test_loader = ''
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test accuracy of the model on the 10000 test images: {} %'
          .format(100 * correct / total))


# 继承torch.nn.Module类写自己的loss。
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, x, y):
        loss = torch.mean((x - y) ** 2)
        return loss


# Mixup训练
alpha = 2
loss_function = MyLoss()
beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()

    # Mixup images and labels.
    lambda_ = beta_distribution.sample([]).item()
    index = torch.randperm(images.size(0)).cuda()
    mixed_images = lambda_ * images + (1 - lambda_) * images[index, :]
    label_a, label_b = labels, labels[index]

    # Mixup loss.
    scores = model(mixed_images)
    loss = (lambda_ * loss_function(scores, label_a)
            + (1 - lambda_) * loss_function(scores, label_b))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 梯度裁剪（gradient clipping）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)

# 模型训练可视化
"""
pip install tensorboard
tensorboard --logdir=runs
使用SummaryWriter类来收集和可视化相应的数据，放了方便查看，
可以使用不同的文件夹，比如'Loss/train'和'Loss/test'。
"""
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

"""
# pip install torchsnooper
使用TorchSnooper来调试PyTorch代码，程序在执行的时候，
就会自动 print 出来每一行的执行结果的 tensor 的形状、数据类型、设备、是否需要梯度的信息。
"""
import torchsnooper

# 对于函数，使用修饰器
@torchsnooper.snoop()
def myfunc(mask, x):
    y = torch.zeros(6, device='cuda')
    y.masked_scatter_(mask, x)
    return y

"""
如果不是函数，使用 with 语句来激活 TorchSnooper，把训练的那个循环装进 with 语句中去。
with torchsnooper.snoop():
    原本的代码
"""