`import torch.nn as nn`
`import torch.nn.functional as F`

`class SimpleCNN(nn.Module):`
    `def __init__(self, num_classes=4):`  //默认四分类
        `super().__init__()`
        
        # 第 1 层：卷积层。输入 1 个通道，输出 8 个特征图，
        # 3x3 卷积核，padding=1 让输出尺寸和输入一样大（32x32 不变）。
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)

        # 第 2 层：池化层。2x2 窗口，取窗口内最大值，尺寸减半（32x32 -> 16x16）。
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第 3 层：第二个卷积层。8 通道 -> 16 通道，3x3 卷积核，尺寸不变（16x16）。
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)

        # 第 4 层：全连接层。前面经过两次池化后尺寸是 8x8，共 16 个通道，
        # 所以展平后有 16*8*8 = 1024 个神经元，压缩到 64。
        self.fc1 = nn.Linear(16 * 8 * 8, 64)

        # 第 5 层：输出层。64 -> 4 个类别分数。
        self.fc2 = nn.Linear(64, num_classes)

       def forward(self, x):
        # 输入 x 形状：(batch, 1, 32, 32)

        x = F.relu(self.conv1(x))   # 卷积 + ReLU 激活  -> (batch, 8, 32, 32)
        x = self.pool(x)            # 最大池化          -> (batch, 8, 16, 16)

        x = F.relu(self.conv2(x))   # 卷积 + ReLU 激活  -> (batch, 16, 16, 16)
        x = self.pool(x)            # 最大池化          -> (batch, 16, 8, 8)

        x = x.view(x.size(0), -1)   # 展平              -> (batch, 1024)

        x = F.relu(self.fc1(x))     # 全连接 + ReLU     -> (batch, 64)
        x = self.fc2(x)             # 输出 4 类分数     -> (batch, 4)

        return x