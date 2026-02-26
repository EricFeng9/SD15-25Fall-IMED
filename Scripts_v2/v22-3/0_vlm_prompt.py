# -*- coding: utf-8 -*-
"""
VLM Prompt 生成脚本（DashScope 官方 SDK 版本）
-------------------
功能：
- 读取 CFFA 数据集中的所有 CF 图像
- 调用阿里云 DashScope VLM API（qwen-vl-plus / qwen3.5-plus）为每张图片生成详细的医学描述
- 保存为 JSON 格式，key 为图片原始名称（子文件夹名_图片ID，如 '001_01_aug1_001_01'）

使用方法：
1. 安装依赖: pip install dashscope tqdm
2. 在下方配置区填入你的 API_KEY
3. 运行脚本: python 0_vlm_prompt.py
4. 生成的 JSON 文件将保存在当前目录下: cf_captions.json
"""

import os
import json
import dashscope
from dashscope import MultiModalConversation
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import glob

# ============ 配置区 ============
API_KEY = "sk-75c9e66533704087b9ae0e85b05e799b"  # 替换为你的真实 API Key
MODEL_NAME = "qwen-vl-plus"  # 可选: qwen-vl-plus, qwen-vl-max, qwen3.5-plus

# 设置 DashScope API Key
dashscope.api_key = API_KEY
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

# 数据集根目录（与训练脚本一致）
DATA_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/data/operation_pre_filtered_cffa_augmented"

# 输出 JSON 文件路径
OUTPUT_JSON = "/data/student/Fengjunming/SDXL_ControlNet/Scripts_v2/v22-2/cf_captions.json"

# 并发线程数（根据 API 限流调整，建议 5-10）
MAX_WORKERS = 8

# ============ VLM Prompt（专为眼底图优化） ============
SYSTEM_PROMPT = """
你是一个专业的眼底图像解剖学分析助手。
请观察这张彩色眼底图（CF），并提供一句高度浓缩的英文图像描述（Caption）。
描述必须包含以下结构信息：
1. 视盘（Optic Disc）的位置（左侧、右侧、中间，或者未见视盘）。
2. 黄斑（Macula）的位置。
3. 血管（Vessels）的连贯性。
4. 整体图像亮度和质量。

请严格返回 JSON 格式，不要输出任何其他废话。格式如下：
{"caption": "A color fundus photography, the bright optic disc is located on the [left/right/center], the macula is dark and centrally located, major blood vessels are continuous..."}
"""

USER_PROMPT = "请按照系统指示分析这张眼底图像，并返回 JSON 格式的描述。"


# ============ 核心功能函数 ============
def get_image_caption_api(image_path, retry=3):
    """
    调用 DashScope VLM API 获取单张图片的描述
    
    Args:
        image_path: 图片路径
        retry: 失败重试次数
    
    Returns:
        str: 图片描述文本（失败时返回默认 fallback）
    """
    for attempt in range(retry):
        try:
            # 构建消息（DashScope 官方格式）
            messages = [
                {
                    "role": "system",
                    "content": [{"text": SYSTEM_PROMPT}]
                },
                {
                    "role": "user",
                    "content": [
                        {"image": f"file://{image_path}"},  # 使用本地文件路径
                        {"text": USER_PROMPT}
                    ]
                }
            ]
            
            # 调用 API
            response = MultiModalConversation.call(
                model=MODEL_NAME,
                messages=messages
            )
            
            # 检查响应状态
            if response.status_code == 200:
                content = response.output.choices[0].message.content[0]["text"]
                
                # [修复] 暴力清洗 VLM 返回的 Markdown 标记
                content = content.replace("```json", "").replace("```", "").strip()
                
                # 尝试解析 JSON
                try:
                    caption_dict = json.loads(content)
                    caption = caption_dict.get("caption", content)
                    return caption
                except json.JSONDecodeError:
                    # 如果不是 JSON 格式，直接返回文本
                    return content.strip()
            else:
                raise Exception(f"API Error: {response.code} - {response.message}")
        
        except Exception as e:
            if attempt < retry - 1:
                print(f"  ⚠️  重试 {attempt + 1}/{retry} - {os.path.basename(image_path)}: {str(e)[:100]}")
                continue
            else:
                print(f"  ❌ 失败 - {os.path.basename(image_path)}: {str(e)[:100]}")
                # 失败时返回默认描述
                return "A color fundus photography, retinal image showing optic disc and blood vessels, medical imaging."


def collect_all_cf_images(root_dir):
    """
    收集所有 CF 图像路径
    
    Returns:
        list of dict: [{'key': '001_01_aug1_001_01', 'path': '/path/to/001_01.png'}, ...]
    """
    all_samples = []
    
    # 遍历所有子目录
    subdirs = sorted(os.listdir(root_dir))
    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        # 寻找所有 CF 图像 (命名格式: XXX_01.png)
        cf_files = glob.glob(os.path.join(subdir_path, "*_01.png"))
        for cf_path in cf_files:
            cf_filename = os.path.basename(cf_path).replace('.png', '')  # 如: '002_01'
            # 生成唯一 key: 子文件夹名/CF文件名（不含扩展名）
            unique_key = f"{subdir}/{cf_filename}"  # 如: '002_01_aug3/002_01'
            all_samples.append({
                'key': unique_key,
                'path': cf_path
            })
    
    return all_samples


def process_single_image(sample):
    """处理单张图片（用于多线程）"""
    key = sample['key']
    path = sample['path']
    caption = get_image_caption_api(path)
    return key, caption


# ============ 主函数 ============
def main():
    print("\n" + "="*60)
    print("  🔬 VLM Prompt 生成器（眼底图专用 - DashScope SDK）")
    print("="*60)
    print(f"✓ 数据集根目录: {DATA_ROOT}")
    print(f"✓ API 模型: {MODEL_NAME}")
    print(f"✓ 并发线程数: {MAX_WORKERS}")
    print(f"✓ 输出文件: {OUTPUT_JSON}")
    print("="*60 + "\n")
    
    # 1. 收集所有 CF 图像
    print("📂 正在扫描数据集...")
    all_samples = collect_all_cf_images(DATA_ROOT)
    total_images = len(all_samples)
    print(f"✓ 找到 {total_images} 张 CF 图像\n")
    
    if total_images == 0:
        print("❌ 未找到任何图像，请检查数据集路径！")
        return
    
    # 2. 检查是否已有缓存结果（断点续传）
    existing_results = {}
    if os.path.exists(OUTPUT_JSON):
        print(f"📥 发现已有 JSON 文件，加载中...")
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
        print(f"✓ 已加载 {len(existing_results)} 条缓存结果\n")
    
    # 筛选出尚未处理的图像
    pending_samples = [s for s in all_samples if s['key'] not in existing_results]
    print(f"📊 待处理图像: {len(pending_samples)} 张")
    print(f"📊 已缓存图像: {len(existing_results)} 张\n")
    
    if len(pending_samples) == 0:
        print("✅ 所有图像均已处理完成！")
        return
    
    # 3. 多线程调用 API
    print("🚀 开始调用 VLM API...\n")
    results = existing_results.copy()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_sample = {executor.submit(process_single_image, s): s for s in pending_samples}
        
        # 使用 tqdm 显示进度条
        with tqdm(total=len(pending_samples), desc="处理进度", ncols=100) as pbar:
            for future in as_completed(future_to_sample):
                try:
                    key, caption = future.result()
                    results[key] = caption
                    
                    # 每处理 50 张图，自动保存一次（防止中途崩溃）
                    if len(results) % 50 == 0:
                        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                            json.dump(results, f, indent=4, ensure_ascii=False)
                
                except Exception as e:
                    sample = future_to_sample[future]
                    print(f"\n⚠️  处理失败: {sample['key']} - {str(e)[:100]}")
                
                finally:
                    pbar.update(1)
    
    # 4. 保存最终结果
    print(f"\n💾 保存结果到 {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("  ✅ 任务完成！")
    print("="*60)
    print(f"✓ 总共处理: {len(results)} 张图像")
    print(f"✓ JSON 文件已保存: {OUTPUT_JSON}")
    print(f"✓ 预估成本: ¥{len(results) * 0.002:.2f} - ¥{len(results) * 0.005:.2f} 元")
    print("="*60 + "\n")
    
    # 5. 显示几个示例
    print("📝 随机示例（前 3 条）:")
    for i, (key, caption) in enumerate(list(results.items())[:3]):
        print(f"  {i+1}. {key}")
        print(f"     {caption}\n")


if __name__ == "__main__":
    main()
