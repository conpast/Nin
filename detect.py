import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models, utils
from createmodel import create
# 定义图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(132),  # 调整图像大小（ResNet的输入通常是224x224，但我们需要先放大再裁剪）
    transforms.CenterCrop(128),  # 中心裁剪
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5,), (0.5,))# 归一化（对于预训练模型）
])

# 加载并预处理图像
img_path = r'/img_1.png'  # 替换为你的图像路径
img = Image.open(img_path).convert('L')  # 打开图像并转换为灰度图
img_tensor = preprocess(img)
img_tensor = img_tensor.unsqueeze(0)  # 增加一个batch维度（模型期望输入是batch x channels x height x width）

# 加载.pkl模型
def load_model(model_path):
    model = create()
    state_dic = torch.load(model_path)
    model.load_state_dict(state_dic)
    model.eval()  # 设置为评估模式
    return model
model = load_model(r"D:\CODE\python\network\mnist.pkl")
with torch.no_grad():  # 禁用梯度计算，因为我们只进行推理
    output = model(img_tensor)

# 处理预测结果（这里假设是分类任务，输出是logits）
_, predicted = torch.max(output, 1)
print(f'Predicted class: {predicted.item()}')