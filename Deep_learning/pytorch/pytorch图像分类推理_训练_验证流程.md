**以图像分类任务为例：**
参考模型为[[SimpleCNN]]
# 模型的训练与验证
## 完整的模型训练过程

![[训练及验证流程.png]]
## 基本的数据处理流程：
**数据是以什么形式在项目中流转的？**

![[图像处理流程图.png]]

**① 在读取jpg或者png格式的图像后，通常不能直接使用，我们通常使用图像处理库读取：**

eg:使用 **Pillow** 将图片转化为参数`640×480, RGB：

`from PIL import image`
`Img = Image.open("tower_crane.jpg")`
`print(img.size)`
`print(img.mode)`

假设输出为 `640×480, RGB

意味着：
- 宽度640像素
- 高度480像素
- channel = 3。(红、绿、蓝，三个颜色通道)

**②此时得到的是图片对象，还不满足cnn的输入要求，即：张量（tensor）**
**cnn要求输入：（batch,1,32,32）**
- batch : 一次输入多少张图像
- 1：每张图像有几个通道
- 32：图像高度为32
- 32：图像宽度为32
**因此使用torchvision进行处理**：
`from PIL import Image`
`from torchvision import transforms`

`transform = transforms.Compose([`
    `transforms.Grayscale(num_output_channels=1),`
    `transforms.Resize((32, 32)),`
    `transforms.ToTensor()`
`])`

`img = Image.open("tower_crane.jpg")`

`x = transform(img)`

`print(x.shape)`
输出：
torch.Size ([1，32，32]) 


## 问题
### 损失计算需要什么？

假设我们输入一张图像x：

模型输出：
`output = model(x)`
得到**预测分数**（logits）：[2.1，-0.3，4.7，1.2]

**真实类别**`label=torch,tensor([2])`
- 即真实类别为第二类
加入损失函数（eg:交叉熵损失）
	`criterion = nn.CrossEntropyLoss()`
	`loss = criterion(output,label)`
从而得到损失值。

### 得到损失值之后模型如何改进？
完整代码为：
	`# 创建模型`
	`model = SimpleCNN(num_classes=4)`
	`# 定义损失函数`
	`criterion = nn.CrossEntropyLoss()`
	`# 定义优化器`
	`optimizer = torch.optim.Adam(`
    `model.parameters(),`
    `lr=0.001`
	`)`

这里有三个对象:

1. **model：神经网络**
负责根据输入图像计算预测结果。
2. [[criterion]]：创建损失函数**
负责衡量预测结果与真实标签之间的差异。
3. **[[optimizer]]：优化器**
负责根据梯度更新模型中可学习的参数。

## 模型训练
### 训练过程中每个epoch
**每个epoch实际在执行内循环**
	① 取一个batch的图像+标签
	②`feed_forward`得到四个输出分数（logits）
	③`CrossEntropyLoss()`计算损失
	④`optimizer.zero_grad()`梯度清零
	⑤`loss.backward()`反向传播计算梯度
	⑥`optimizer.step()`参数更新,按照梯度调整权重
	⑦验证、统计精度
	⑧重复epoch，提升精度
	⑨训练完成后，跑测试，出混淆矩阵