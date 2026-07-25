"""
Multi-Scale Underwater Image Enhancement - CLI Batch Mode
一键运行：自动读取 tests/test_images/original/ 中 1-20 序号的图像，批量处理
支持自动识别水体类型（偏黄/偏绿/中性），智能选择参数
"""

import os
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from enhancer import UnderwaterEnhancer
from utils import (
    read_image, save_image, bgr2rgb,
    evaluate_uiqm, evaluate_uciqe,
)
from fusion import fuse_underwater, create_version_configs, FUSION_CONFIGS

# ==================== 预设参数（3种水体类型） ====================

PRESETS = {
    # 偏黄水体：蓝光衰减、暖色调、可能浑浊
    "yellow": {
        "a_shift": 0,           # 中性
        "b_shift": -10,         # 压低B通道（黄→蓝校正）
        "red_strength": 0.25,   # 红色衰减不严重
        "clahe_limit": 2.0,
        "clahe_grid": 8,
        "dehaze_omega": 0.82,   # 较强去雾（黄水体通常浑浊）
        "dehaze_tmin": 0.35,
        "use_guided_filter": True,
        "sharpen_amount": 1.2,
        "sharpen_radius": 0.5,
        "gamma": 1.05,          # 轻微提亮
    },
    # 偏绿水体：红色严重衰减、整体偏暗、对比度低
    "green": {
        "a_shift": -10,         # 压低A通道（绿→品红校正）
        "b_shift": 0,           # 中性
        "red_strength": 0.40,   # 红色通道需要强力恢复
        "clahe_limit": 3.0,     # 更强对比度增强
        "clahe_grid": 8,
        "dehaze_omega": 0.72,   # 中等去雾
        "dehaze_tmin": 0.28,    # 更低透射率下限（偏暗）
        "use_guided_filter": True,
        "sharpen_amount": 1.3,
        "sharpen_radius": 0.5,
        "gamma": 1.15,          # 较强提亮
    },
    # 中性/偏蓝水体
    "neutral": {
        "a_shift": 0,
        "b_shift": 0,
        "red_strength": 0.30,
        "clahe_limit": 2.0,
        "clahe_grid": 8,
        "dehaze_omega": 0.75,
        "dehaze_tmin": 0.35,
        "use_guided_filter": True,
        "sharpen_amount": 1.2,
        "sharpen_radius": 0.5,
        "gamma": 1.1,
    },
}

# ==================== 手动参数（覆盖预设值） ====================
# 修改此处参数会覆盖自动检测的预设值
# auto_detect = True  → 自动识别水体类型 + 手动参数覆盖
# auto_detect = False → 完全使用下方手动参数

CONFIG = {
    # 图像I/O
    "input_dir": "tests/test_images/original",
    "output_dir": "tests/test_images/results",          # 单参数增强输出目录
    "output_dir_fusion": "tests/test_images/results_mul", # 多尺度融合输出目录
    "max_width": 1200,
    "start_num": 1,
    "end_num": 20,
    # 自动区分水体识别开关
    "auto_detect": True,        # True=自动识别水体类型, False=纯手动
    # 多尺度金字塔融合模式
    "use_fusion": True,         # True=多参数融合, False=单参数增强
    # 手动参数（当 auto_detect=True 时，非 None 的值会覆盖预设）
    "a_shift": None,            # None = 使用预设值; 设为数字则覆盖
    "b_shift": None,
    "red_strength": None,
    "clahe_limit": None,
    "clahe_grid": None,
    "dehaze_omega": None,
    "dehaze_tmin": None,
    "use_guided_filter": None,
    "sharpen_amount": None,
    "sharpen_radius": None,
    "gamma": None,
}


# ==================== 水体类型识别 ====================

def classify_water_type(img):
    """
    基于LAB颜色空间识别水体类型
    返回: "yellow" / "green" / "neutral"
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    _, a, b = cv2.split(lab)

    # 归一化到 [-128, 128] 范围
    mean_a = np.mean(a) - 128.0
    mean_b = np.mean(b) - 128.0

    threshold = 5.0  # 判定阈值

    if mean_b > threshold and mean_b > abs(mean_a):
        return "yellow"
    elif mean_a < -threshold and abs(mean_a) > abs(mean_b):
        return "green"
    else:
        return "neutral"


def get_effective_params(img, config):
    """
    获取当前图像的有效参数：
    1. 自动识别水体类型 → 选择预设
    2. 用 config 中非 None 的值覆盖预设
    3. 返回最终参数字典
    """
    if config.get("auto_detect", True):
        water_type = classify_water_type(img)
        params = dict(PRESETS[water_type])  # 复制预设
        print(f"  [检测] 水体类型: {water_type}")
    else:
        # 纯手动模式，从 neutral 预设出发，全部用 config 覆盖
        params = dict(PRESETS["neutral"])
        water_type = "manual"

    # config 中的非 None 值覆盖预设
    override_keys = [
        "a_shift", "b_shift", "red_strength",
        "clahe_limit", "clahe_grid",
        "dehaze_omega", "dehaze_tmin", "use_guided_filter",
        "sharpen_amount", "sharpen_radius", "gamma",
    ]
    overrides = []
    for k in override_keys:
        if k in config and config[k] is not None:
            params[k] = config[k]
            overrides.append(f"{k}={config[k]}")

    if overrides:
        print(f"  [覆盖] 手动参数: {', '.join(overrides)}")

    return params, water_type


# ==================== 核心逻辑 ====================

def find_image(num, input_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """根据编号查找图像文件，自动匹配扩展名"""
    for ext in extensions:
        path = os.path.join(input_dir, f"{num}{ext}")
        if os.path.isfile(path):
            return path
    return None


def process_single(num, config, img_count):
    """处理单张图像"""
    img_path = find_image(num, config["input_dir"])
    if img_path is None:
        return None

    print(f"\n{'─'*50}")
    print(f"  [{num}/{img_count}] 处理: {os.path.basename(img_path)}")
    print(f"{'─'*50}")

    # 读取
    img = read_image(img_path)

    # 尺寸调整
    h, w = img.shape[:2]
    if w > config["max_width"]:
        scale = config["max_width"] / w
        img = cv2.resize(img, (config["max_width"], int(h * scale)),
                         interpolation=cv2.INTER_AREA)
        print(f"  [IO] 缩放: {w}x{h} -> {config['max_width']}x{int(h * scale)}")

    use_fusion = config.get("use_fusion", False)

    if use_fusion:
        enhanced, intermediate = _process_fusion(num, img, config)
    else:
        enhanced, intermediate = _process_single_pipeline(num, img, config)

    # 根据模式选输出目录
    out_dir = config["output_dir_fusion"] if use_fusion else config["output_dir"]

    # 质量评估（无参考）
    uiqm = evaluate_uiqm(enhanced)
    uciqe = evaluate_uciqe(enhanced)

    # 保存增强结果
    os.makedirs(out_dir, exist_ok=True)
    save_image(enhanced, os.path.join(out_dir, f"{num}_enhanced.png"))

    # 保存中间步骤
    for step_name, step_img in intermediate.items():
        step_filename = f"{num}_{step_name}.png"
        cv2.imwrite(os.path.join(out_dir, step_filename), step_img)

    # 生成汇总示意图
    water_type = getattr(process_single, "_last_water_type", None)
    summary_path = os.path.join(out_dir, f"{num}_summary.png")
    _generate_summary(img, enhanced, intermediate, summary_path,
                       water_type=water_type)

    print(f"  UIQM: {uiqm['UIQM']}   UCIQE: {uciqe['UCIQE']}")

    return {"image": num, "uiqm": uiqm["UIQM"], "uciqe": uciqe["UCIQE"]}


def _process_single_pipeline(num, img, config):
    """单参数增强模式"""
    t_start = time.time()

    params, water_type = get_effective_params(img, config)
    process_single._last_water_type = water_type

    enhancer = UnderwaterEnhancer()
    enhancer.set_all_params(
        white_balance={"a_shift": params["a_shift"],
                       "b_shift": params["b_shift"]},
        red_channel={"strength": params["red_strength"]},
        clahe={"clip_limit": params["clahe_limit"],
               "grid_size": (params["clahe_grid"], params["clahe_grid"])},
        dehaze={"omega": params["dehaze_omega"],
                "t_min": params["dehaze_tmin"],
                "use_guided_filter": params["use_guided_filter"]},
        unsharp_mask={"amount": params["sharpen_amount"],
                      "radius": params["sharpen_radius"]},
        gamma={"gamma": params["gamma"]},
    )

    enhanced = enhancer.process(img)
    intermediate = enhancer.get_intermediate()

    t_elapsed = time.time() - t_start
    print(f"  耗时: {t_elapsed:.2f}s")
    return enhanced, intermediate


def _process_fusion(num, img, config):
    """多尺度金字塔融合模式"""
    t_start = time.time()

    # 水体类型检测
    water_type = classify_water_type(img)
    process_single._last_water_type = water_type
    print(f"  [检测] 水体类型: {water_type}")

    # 选择融合配置
    fusion_cfgs = create_version_configs(water_type)

    # 执行多尺度金字塔融合
    result, versions = fuse_underwater(
        img, fusion_configs=fusion_cfgs,
        collect_intermediate=True
    )

    # 构建 intermediate 字典用于保存
    intermediate = {}
    intermediate["01_Original"] = img
    for i, (cfg, ver) in enumerate(zip(fusion_cfgs, versions)):
        intermediate[f"V{i+1}_{cfg['name']}"] = ver
    intermediate["Fused"] = result

    t_elapsed = time.time() - t_start
    print(f"  融合耗时: {t_elapsed:.2f}s")
    return result, intermediate


def _generate_summary(original, enhanced, intermediate, save_path,
                       water_type=None):
    """生成每张图像的汇总示意图"""
    step_names = list(intermediate.keys())

    images = [(step_names[0], bgr2rgb(original))]
    for name in step_names[1:]:
        if name in intermediate:
            images.append((name, bgr2rgb(intermediate[name])))
    images.append(("Enh_Final", bgr2rgb(enhanced)))

    n_total = len(images)
    cols = 4
    rows = (n_total + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = axes.flatten() if n_total > 1 else [axes]

    for idx, (title, img) in enumerate(images):
        axes[idx].imshow(img)
        axes[idx].set_title(title, fontsize=9)
        axes[idx].axis("off")

    for idx in range(n_total, len(axes)):
        axes[idx].axis("off")

    # 在图上标注水体类型
    if water_type:
        fig.suptitle(f"Water Type: {water_type.upper()}", fontsize=12,
                     fontweight="bold", y=0.99)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  汇总图已保存: {os.path.basename(save_path)}")


def main():
    print(f"\n{'='*60}")
    print(f"  水下图像增强系统")
    print(f"{'='*60}")

    # 检查输入目录
    input_dir = os.path.join(os.path.dirname(__file__) or ".", CONFIG["input_dir"])
    output_dir = os.path.join(os.path.dirname(__file__) or ".", CONFIG["output_dir"])
    output_dir_fusion = os.path.join(os.path.dirname(__file__) or ".", CONFIG["output_dir_fusion"])
    CONFIG["input_dir"] = input_dir
    CONFIG["output_dir"] = output_dir
    CONFIG["output_dir_fusion"] = output_dir_fusion

    if CONFIG.get("auto_detect", True):
        print(f"  [模式] 智能识别 (自动检测水体类型)")
    else:
        print(f"  [模式] 手动参数")
    if CONFIG.get("use_fusion", False):
        print(f"  [模式] 多尺度金字塔融合 (Multi-scale Fusion)")
    else:
        print(f"  [模式] 单参数增强")

    if not os.path.isdir(input_dir):
        print(f"\n[错误] 输入目录不存在: {input_dir}")
        print(f"请创建目录并将测试图像（1.jpg ~ 20.jpg）放入其中。")
        return

    # 扫描图像
    image_list = []
    for num in range(CONFIG["start_num"], CONFIG["end_num"] + 1):
        if find_image(num, input_dir):
            image_list.append(num)

    if not image_list:
        print(f"\n[错误] 在 {input_dir} 中没有找到 1~20 序号的图像。")
        print(f"支持的格式: .jpg .jpeg .png .bmp")
        return

    use_fusion = CONFIG.get("use_fusion", False)
    print(f"\n[扫描] 找到 {len(image_list)} 张图像: {image_list}")
    print(f"[输出] 结果保存到: {output_dir_fusion if use_fusion else output_dir}")

    # 批量处理
    results = []
    for num in image_list:
        result = process_single(num, CONFIG, len(image_list))
        if result:
            results.append(result)

    # 汇总
    print(f"\n{'='*60}")
    print(f"  处理完成 ({len(results)}/{len(image_list)} 张)")
    print(f"{'='*60}")

    if results:
        avg_uiqm = sum(r["uiqm"] for r in results) / len(results)
        avg_uciqe = sum(r["uciqe"] for r in results) / len(results)
        print(f"  平均 UIQM:  {avg_uiqm:.4f}")
        print(f"  平均 UCIQE: {avg_uciqe:.4f}")

    print(f"\n完成！所有结果已保存到: {output_dir}\n")


if __name__ == "__main__":
    main()
