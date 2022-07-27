# 导入相关库
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import cv2

# 使用随机化种子使得神经网络的每次初始化都相同
torch.manual_seed(1)  

# 各种可在初始阶段调整的超参数
EPOCH = 1  # 所有数据的迭代训练批次
BATCH_SIZE = 300 # 批训练数据数量
LR = 0.002  # 学习率
# 第一次使用该程序需要将该参数设置为TRUE来在线下载数据集
# 若已下载过并放在对应路径中则可将其设置为FALSE
DOWNLOAD_MNIST = True 

# 下载MINIST手写数据集
train_data = torchvision.datasets.MNIST(
    root = './data/',  # 训练数据集的所在路径
    train = True,  # True：用于训练的数据; False：用于测试的数据
    transform = torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray
    download = DOWNLOAD_MNIST,  # 是否需要在线下载训练数据集
)

# 测试集
test_data = torchvision.datasets.MNIST(
    root='./data/',
    train = False  # 若选用测试集则将其设置为True
)

# 批训练BATCH_SIZE个样本数据
# Torch中的DataLoader是用来包装数据的工具，它能帮我们有效迭代数据，这样就可以进行批训练
train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle = True  # True：对数据进行随机化打乱
)

# 进行测试
# 为节约时间，测试时只测试前2000个

# torch.unsqueeze(a) 是用来对数据维度进行扩充，这样shape就从(2000,28,28)->(2000,1,28,28)
# 对于图像像素pixel：除以255使得数据归一化
test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor)[:2000] / 255
test_y = test_data.test_labels[:2000]


# 用class类来建立CNN模型
# CNN流程：卷积(Conv2d) -> 激励函数(ReLU) - >池化(MaxPooling) -> 展平多维的卷积成的特征图 -> 接入全连接层(Linear) -> 输出

# 我们建立的CNN继承nn.Module这个模块
class CNN(nn.Module):  
    def __init__(self):
        super(CNN, self).__init__()
        # 建立第一个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv1 = nn.Sequential(
            # 第一个卷积con2d
            nn.Conv2d(  # 输入图像大小(1,28,28)
                in_channels = 1,  # 输入信号的通道数：因为MINIST数据集是灰度图像，所以只有一个通道
                out_channels = 16,  # 卷积后输出结果的通道数
                kernel_size = 5,  # 卷积核的形状为5*5
                stride = 1,  # 卷积每次移动的步长
                padding = 2,  # 处理边界时填充0的数量：padding = (kernel_size-1)/2
            ),  # 输出图像大小(16,28,28)
            # 激活函数
            nn.ReLU(),
            # 池化，下采样
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
            # 输出图像大小(16,14,14)
        )
        # 建立第二个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv2 = nn.Sequential(
            # 输入图像大小(16,14,14)
            nn.Conv2d(  # 也可以直接简化写成nn.Conv2d(16,32,5,1,2)
                in_channels = 16,
                out_channels = 32,
                kernel_size = 5,
                stride = 1,
                padding = 2
            ),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出图像大小(32,7,7)
        )
        # 建立全卷积连接层
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输出是10个类

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.conv1(x)  # x先通过conv1
        x = self.conv2(x)  # 再通过conv2
        # 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
        x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
        output = self.out(x)
        return output

# 调用CNN函数
cnn = CNN()
#print(cnn)

# 训练
# 把x和y都放入Variable中，然后放入CNN中计算结果和误差

# 优化器选择Adam
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# 损失函数
loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted

# 开始训练
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # 分配batch data
        output = cnn(b_x)  # 先将数据放到CNN中计算output
        loss = loss_func(output, b_y)  # 输出和真实标签的loss，二者位置不可颠倒
        optimizer.zero_grad()  # 清除之前学到的梯度的参数
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 应用梯度
        # 输出该批次数据的部分训练误差和准确度
        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

torch.save(cnn.state_dict(), 'cnn2.pkl') #保存模型


# 加载模型，若已经训练过，想要直接利用训练过的模型来对测试集进行检测，则需要注释107-130行代码，避免再次训练
cnn.load_state_dict(torch.load('cnn2.pkl'))
cnn.eval()
# 测试前32个数据，查看预测结果
inputs = test_x[:32]
test_output = cnn(inputs)
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'Prediction Number')  # 打印识别后的数字

img = torchvision.utils.make_grid(inputs)
img = img.numpy().transpose(1, 2, 0)

cv2.imshow('win', img)  # opencv显示需要识别的数据图片
key_pressed = cv2.waitKey(0)