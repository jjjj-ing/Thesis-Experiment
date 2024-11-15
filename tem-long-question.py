import os
import io
from PIL import Image
from transformers import pipeline
from diffusers import PixArtSigmaPipeline
import torch
from skimage.metrics import mean_squared_error
import numpy as np

# 加载 LLaVA 模型进行图像描述并使用 I/O 流
def image_description_stream(image, user_questions, output_description_path):
    # 加载 LLaVA 模型，使用 GPU
    model_id = "xtuner/llava-phi-3-mini-hf"
    llava_pipe = pipeline("image-to-text", model=model_id, device=0)

    # 清理显存
    torch.cuda.empty_cache()

    # 使用 StringIO 来代替文件写入
    description_stream = io.StringIO()

    # 对每个问题进行描述生成
    for idx, user_text in enumerate(user_questions):
        # 生成对话模板并创建提示
        prompt = f"<|user|>\n<image>\n{user_text}<|end|>\n<|assistant|>\n"

        # 生成文本描述，不限制生成的新标记数量
        max_tokens = 250  # 根据需要调整生成的新标记数量
        with torch.no_grad():
            outputs = llava_pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": max_tokens})


        # 获取完整的描述
        description = outputs[0]['generated_text']

        # 写入到内存中的 StringIO
        description_stream.write(f"Question {idx + 1}: {description}\n")

        # 输出描述到控制台，方便检查生成的质量
        print(f"Question {idx + 1}: {description}")

    # 卸载 LLaVA 模型并释放显存
    del llava_pipe
    torch.cuda.empty_cache()

    # 将描述写入磁盘文件
    with open(output_description_path, "w", encoding="utf-8") as desc_file:
        desc_file.write(description_stream.getvalue())

    # 获取最终的描述内容
    return description_stream.getvalue()

# 使用 PixArt Sigma 模型生成一张图像并使用 I/O 流
def create_new_image_stream(prompt, output_path):
    # 使用 GPU 加载 PixArt Sigma Pipeline，启用低 CPU 内存模式
    sigma_pipeline = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to("cuda")

    print("Sigma pipeline loaded.")

    # 清理显存
    torch.cuda.empty_cache()

    # 生成图像
    generator = torch.Generator(device="cpu").manual_seed(0)

    with torch.no_grad():
        image = sigma_pipeline(prompt, height=256, width=384, guidance_scale=2.5, generator=generator).images[0]

    # 使用 BytesIO 代替文件保存
    image_stream = io.BytesIO()
    image.save(image_stream, format='PNG')
    image_stream.seek(0)

    # 将生成的图像保存到磁盘
    image.save(output_path, format='PNG')
    print(f"Generated image saved at: {output_path}")

    # 卸载模型并释放显存
    del sigma_pipeline
    torch.cuda.empty_cache()

    return image_stream

# 批量处理文件夹中的图像
def process_images_stream(input_folder, output_folder, user_questions):
    mse_values = []

    # 遍历文件夹中的所有图像文件
    for image_name in os.listdir(input_folder):
        if image_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', 'jfif')):
            image_path = os.path.join(input_folder, image_name)
            print(f"Processing image: {image_path}")

            # 加载图像
            original_image = Image.open(image_path).convert("RGB")

            # 定义描述文件的输出路径
            output_description_path = os.path.join(output_folder, f"description_{os.path.splitext(image_name)[0]}.txt")

            # 生成图像描述并保存到文件
            description = image_description_stream(original_image, user_questions, output_description_path)

            # 定义生成图像的输出路径
            output_image_path = os.path.join(output_folder, f"generated_{os.path.splitext(image_name)[0]}.png")

            # 使用描述生成新图像（通过内存流方式）并保存到指定文件夹中
            generated_image_stream = create_new_image_stream(description, output_image_path)
            generated_image = Image.open(generated_image_stream)

            # 计算原始图像和生成图像之间的 MSE
            original_array = np.array(original_image)
            generated_array = np.array(generated_image)

            # 调整生成图像的大小以匹配原始图像
            if original_array.shape != generated_array.shape:
                generated_image = generated_image.resize(original_image.size)
                generated_array = np.array(generated_image)

            mse_value = mean_squared_error(original_array, generated_array)
            mse_values.append({
                "image_name": image_name,
                "mse": mse_value
            })

    # 汇总并展示所有的 MSE 值
    print("\nSummary of MSE values:")
    for mse_info in mse_values:
        print(f"Original Image: {mse_info['image_name']}, MSE: {mse_info['mse']}")

# 输入文件夹路径
input_folder = 'D:\\vene\\exp\\input'  
output_folder = 'D:\\vene\\exp\\output\\long'  
os.makedirs(output_folder, exist_ok=True)

# 将问题分成多个较小的部分
user_questions = [
    "Describe the color, model, and time of the accident.",
    "Describe the casualties and road facility damage caused by the accident.",
    "Describe the surroundings of the vehicle involved.",
    "Describe the emergency vehicles and traffic police at the scene of the accident.",
    "Describe the impact angle of the accident vehicle.",
    "Describe the externally damaged parts of the vehicle."
]

# 批量处理所有图像
process_images_stream(input_folder, output_folder, user_questions)
