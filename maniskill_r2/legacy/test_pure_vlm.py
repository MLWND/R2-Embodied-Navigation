"""纯 VLM 神经网络多模态推理验证脚本 (无任何规则或关键词兜底)

验证流程:
1. 在 GPU 1 上加载真实的 Qwen-VL 视觉大模型神经网络权重
2. 构造一张机载相机拍摄的多物体 RGB 图像 (包含桌子、冰箱、沙发)
3. 将 (图像像素 Tensor + 自然语言提示词) 一同输入多模态大模型
4. 打印大模型神经网络输出的原始 Token、JSON 结果与预测框
"""
import os
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# 模型权重本地路径
MODEL_PATH = os.path.expanduser("~/.cache/modelscope/models/Qwen--Qwen2.5-VL-3B-Instruct/snapshots/master")

print("="*70)
print(f"🧠 正在 GPU 1 上加载 Qwen-VL 视觉大模型神经网络权重...")
print(f"   模型路径: {MODEL_PATH}")
print("="*70)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="cuda:0",
    trust_remote_code=True
).eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("✅ Qwen-VL 神经网络加载完成！\n")

# 构造机载相机拍摄的场景图像
img = Image.new("RGB", (640, 480), color=(225, 230, 235))
draw = ImageDraw.Draw(img)
# 地面
draw.rectangle([0, 192, 640, 480], fill=(210, 215, 220))
# 桌子 (左侧)
draw.rectangle([130, 120, 280, 220], fill=(180, 115, 55), outline=(30, 30, 30), width=2)
# 沙发 (中间)
draw.rectangle([250, 160, 420, 250], fill=(70, 120, 180), outline=(30, 30, 30), width=2)
# 冰箱 (右侧)
draw.rectangle([460, 100, 560, 330], fill=(230, 230, 240), outline=(30, 30, 30), width=2)

test_instructions = [
    "请帮我找到冰箱，我想去它前面",
    "走到沙发附近休息一下",
    "去桌子旁边放个杯子"
]

for idx, user_cmd in enumerate(test_instructions):
    print(f"\n" + "-"*60)
    print(f"【测试案例 {idx+1}】输入指令: 「{user_cmd}」")
    print(f"-"*60)
    
    prompt = (
        f"You are a robot vision system. User instruction: '{user_cmd}'.\n"
        f"Identify the target object requested in the instruction in the image.\n"
        f"Return JSON format:\n"
        f'{{"target_name": "...", "bbox_2d": [ymin, xmin, ymax, xmax], "spatial_relation": "front|side|near"}}\n'
        f"where bbox_2d is normalized to 0-1000 range."
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda:0")
    
    # 神经网络前向推理生成
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=128)
        out_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        response_text = processor.batch_decode(out_trimmed, skip_special_tokens=True)[0]
        
    print(f"🤖 Qwen-VL 神经网络原生输出 (Raw Model Response):")
    print(response_text)
