import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from createmodel import create

def main():
    # 加载数据集
    # 加载Fashion-MNIST数据集

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    #
    # testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))  # 对于MNIST，我们只需要一个均值和标准差来归一化灰度图像
    # ])

    # 下载并加载训练集
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True,num_workers = 2)

    # 下载并加载测试集
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False,num_workers =2)

    net = create()
    # 初始化模型
    # Xavier初始化：
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d: #对全连接层和卷积层初始化
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.to(device)
    print(torch.cuda.is_available())
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 训练模型
    num_epochs = 1
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        test_loss = 0.0
        correct = 0
        total = 0

        # 训练模型
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 测试模型
        model.eval()
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(trainloader)
        avg_test_loss = test_loss / len(testloader)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Acc: {correct/total*100:.2f}%")


    torch.save(net.state_dict(), "mnist.pkl")

    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
if __name__ =='__main__':
    main()