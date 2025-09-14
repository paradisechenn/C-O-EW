import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import LoadData
from sklearn.model_selection import cross_val_score
from torchvision.models import alexnet  #最简单的模型
from torchvision.models import vgg11, vgg13, vgg16, vgg19   # VGG系列
from torchvision.models import resnet18, resnet34,resnet50, resnet101, resnet152    # ResNet系列
from torchvision.models import inception_v3     # Inception 系列
import numpy as np
import matplotlib.pyplot as plt


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    # 从数据加载器中读取batch（一次读取多少张，即批次数），X(图片数据)，y（图片真实标签）。
    for batch, (X, y) in enumerate(dataloader):
        # 将数据移动到同一设备
        X, y = X.to(device), y.to(device)

        # 得到预测的结果pred
        pred = model(X)

        # 计算预测的误差
        loss = loss_fn(pred, y)

        # 反向传播，更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每训练10次，输出一次当前信息
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()  # 设置为评估模式
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # 将数据移动到同一设备
            X, y = X.to(device), y.to(device)

            # 进行预测
            pred = model(X)

            # 计算损失
            test_loss += loss_fn(pred, y).item()

            # 统计正确的预测
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    accuracy = correct / size
    print(f"Validation Accuracy: {100 * accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy


if __name__ == '__main__':
    batch_size = 4

    # 给训练集、验证集和测试集分别创建一个数据集加载器
    train_data = LoadData("train.txt", True)
    val_data = LoadData("val.txt", False)
    test_data = LoadData("test.txt", False)

    train_dataloader = DataLoader(dataset=train_data, num_workers=4, pin_memory=True, batch_size=batch_size,
                          shuffle=True)
    val_dataloader = DataLoader(dataset=val_data, num_workers=4, pin_memory=True, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test_data, num_workers=4, pin_memory=True, batch_size=batch_size)

    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # 选择模型
    # model = alexnet(pretrained=False, num_classes=27).to(device)

    '''        VGG系列    '''
    # model = vgg11(pretrained=False, num_classes=27).to(device)   #  23.1%
    # model = vgg13(pretrained=False, num_classes=27).to(device)   # 30.0%
    # model = vgg16(pretrained=False, num_classes=27).to(device)
    # model = vgg19(pretrained=False, num_classes=27).to(device)

    '''        ResNet系列    '''
    # model = resnet18(pretrained=False, num_classes=27).to(device)    # 43.6%
    # model = resnet34(pretrained=False, num_classes=27).to(device)
    # model = resnet50(pretrained= False, num_classes=27).to(device)
    model = resnet101(pretrained=False, num_classes=27).to(device)   #  26.2%
    # model = resnet152(pretrained=False, num_classes=27).to(device)


    print(model)
    # 定义损失函数，计算相差多少，交叉熵
    loss_fn = nn.CrossEntropyLoss()

    # 定义优化器，用来训练时优化模型参数，随机梯度下降法
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,momentum=0.9)

    # 一共训练N次
    epochs = 100  
    train_loss = []
    train_acc = []
    val_acc = []  # 保存每次epoch后的验证集准确率

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        time_start = time.time()
        train(train_dataloader, model, loss_fn, optimizer, device)
        time_end = time.time()
        print(f"train time: {(time_end - time_start)}")

        # 评估验证集准确率
        val_accuracy = test(val_dataloader, model, loss_fn, device)
        val_acc.append(val_accuracy)  # 保存每个epoch的验证集准确率

    print("Done!")
    # 保存训练好的模型
    torch.save(model.state_dict(), "model_alexnet.pth")
    print("Saved PyTorch Model Success!")

    with open("./train_loss.txt", 'w') as train_los:
            train_los.write(str(train_loss))

    with open("./val_acc.txt", 'w') as val_ac:
            val_ac.write(str(val_acc))


        # 读取存储为txt文件的数据
    def data_read(dir_path):
        with open(dir_path, "r") as f:
            raw_data = f.read()
            data = raw_data[1:-1].split(", ")  # [-1:1]是为了去除文件中的前后中括号"[]"

            # 过滤掉空字符串和非数字的项
            filtered_data = []
            for item in data:
                try:
                    filtered_data.append(float(item))  # 尝试将每项转换为浮点数
                except ValueError:
                    continue  # 如果转换失败，跳过该项

        return np.array(filtered_data)

    # 绘制损失曲线
    train_loss_path = r"D:"  # 存数据集路径
    y_train_loss = data_read(train_loss_path)  # loss值，即y轴
    x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epochs')  # x轴标签
    plt.ylabel('loss')  # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.legend()
    plt.title('Loss curve')
    plt.show()

    # 绘制准确率曲线
    val_acc_path = r"D:"  # 数据集路径
    y_val_acc = data_read(val_acc_path)  # 验证集准确率值，即y轴
    x_val_acc = range(len(y_val_acc))  # 验证集阶段准确率的数量，即x轴
    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epochs')  # x轴标签
    plt.ylabel('accuracy')  # y轴标签

    # 以x_val_acc为横坐标，y_val_acc为纵坐标，曲线宽度为1，实线，增加标签，验证集准确率
    plt.plot(x_val_acc, y_val_acc, color='red', linewidth=1, linestyle="solid", label="validation accuracy")
    plt.legend()
    plt.title('Validation Accuracy curve')
    plt.show()



