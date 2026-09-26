import torch
import torchvision.transforms as transforms
from PIL import Image
from models import *

# CIFAR10 10个类别
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# 预处理，和训练保持一致
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化 SimpleDLA
model = SimpleDLA().to(device)
# 权重文件路径
checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=device)
state_dict = checkpoint['net']
# 去掉所有key前面的 module.
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # 去掉前面7个字符 "module."
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()


def predict_image(img_path):
    # 打开图片
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_index = torch.argmax(output, dim=1).item()
    return classes[pred_index]


if __name__ == '__main__':
    print("==== CIFAR10图片分类预测（SimpleDLA）====")
    img_path = input("请输入图片文件路径：")
    try:
        res = predict_image(img_path)
        print(f"✅ 预测结果：{res}")
    except Exception as e:
        print(f"❌ 出错了：{e}")
