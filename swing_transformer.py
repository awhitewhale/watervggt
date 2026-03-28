import torch
from PIL import Image
from transformers import AutoFeatureExtractor, SwinForImageClassification
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import os

# 加载模型
model_name = "microsoft/swin-tiny-patch4-window7-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = SwinForImageClassification.from_pretrained(model_name)
model.eval().cuda()

# 加载你自己的两张图片
image_paths = ["self_collected_examples/room/bedroom/images/bedroom_left_Infrared.png", "self_collected_examples/room/bedroom/images/bedroom_right_Infrared.png"]  # 替换为你的路径
images = [Image.open(p).convert("RGB") for p in image_paths]

# 转换为模型输入
inputs = feature_extractor(images=images, return_tensors="pt")
inputs = {k: v.cuda() for k, v in inputs.items()}

# 计算FLOPs和参数量
with torch.no_grad():
    flops = FlopCountAnalysis(model, inputs["pixel_values"])
    print(f"[FLOPs] {flops.total() / 1e9:.2f} GFLOPs")

print(parameter_count_table(model))