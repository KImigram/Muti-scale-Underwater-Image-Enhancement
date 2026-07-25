"""
水下图像增强系统 - 多尺度金字塔融合模块
对同一输入图像运行多次增强流水线（不同参数），通过金字塔融合生成最终结果。
"""

import cv2
import numpy as np

from enhancer import UnderwaterEnhancer


# ==================== 多参数配置（5组） ====================

FUSION_CONFIGS = [
    # 配置1: 默认均衡
    {
        "name": "Default",
        "white_balance": {"a_shift": 0, "b_shift": 0},
        "red_channel": {"strength": 0.30},
        "clahe": {"clip_limit": 2.0, "grid_size": (8, 8)},
        "dehaze": {"omega": 0.75, "t_min": 0.35,
                   "use_guided_filter": True, "kernel_size": 15},
        "unsharp_mask": {"amount": 1.2, "radius": 0.5},
        "gamma": {"gamma": 1.10},
    },
    # 配置2: 偏黄水体校正（强蓝恢复）
    {
        "name": "Anti-Yellow",
        "white_balance": {"a_shift": 0, "b_shift": -12},
        "red_channel": {"strength": 0.25},
        "clahe": {"clip_limit": 2.0, "grid_size": (8, 8)},
        "dehaze": {"omega": 0.82, "t_min": 0.35,
                   "use_guided_filter": True, "kernel_size": 15},
        "unsharp_mask": {"amount": 1.2, "radius": 0.5},
        "gamma": {"gamma": 1.05},
    },
    # 配置3: 偏绿水体校正（强红恢复 + 提亮）
    {
        "name": "Anti-Green",
        "white_balance": {"a_shift": -12, "b_shift": 0},
        "red_channel": {"strength": 0.42},
        "clahe": {"clip_limit": 3.0, "grid_size": (8, 8)},
        "dehaze": {"omega": 0.70, "t_min": 0.28,
                   "use_guided_filter": True, "kernel_size": 15},
        "unsharp_mask": {"amount": 1.3, "radius": 0.5},
        "gamma": {"gamma": 1.18},
    },
    # 配置4: 强去雾 + 高对比度（浑浊水体）
    {
        "name": "Dehaze-Plus",
        "white_balance": {"a_shift": 0, "b_shift": -5},
        "red_channel": {"strength": 0.30},
        "clahe": {"clip_limit": 3.5, "grid_size": (8, 8)},
        "dehaze": {"omega": 0.85, "t_min": 0.25,
                   "use_guided_filter": True, "kernel_size": 15},
        "unsharp_mask": {"amount": 1.4, "radius": 0.6},
        "gamma": {"gamma": 1.10},
    },
    # 配置5: 柔和增强（保留原始色调）
    {
        "name": "Mild",
        "white_balance": {"a_shift": 0, "b_shift": -3},
        "red_channel": {"strength": 0.20},
        "clahe": {"clip_limit": 1.5, "grid_size": (8, 8)},
        "dehaze": {"omega": 0.60, "t_min": 0.40,
                   "use_guided_filter": True, "kernel_size": 15},
        "unsharp_mask": {"amount": 1.1, "radius": 0.4},
        "gamma": {"gamma": 1.05},
    },
]


# ==================== 权重图计算 ====================

def _laplacian_contrast(img):
    """计算拉普拉斯能量（局部对比度）"""
    img_f = img.astype(np.float32) / 255.0
    gray = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
    return np.abs(lap)


def _saturation_measure(img):
    """计算饱和度（RGB三通道标准差）"""
    img_f = img.astype(np.float32) / 255.0
    mean = np.mean(img_f, axis=2)
    diff = img_f - np.stack([mean] * 3, axis=2)
    sat = np.sqrt(np.mean(diff ** 2, axis=2))
    return sat


def _well_exposedness(img):
    """计算曝光良好度（接近0.5为最佳）"""
    img_f = img.astype(np.float32) / 255.0
    sigma = 0.2
    expo = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    for c in range(3):
        channel = img_f[:, :, c]
        expo *= np.exp(-0.5 * ((channel - 0.5) / sigma) ** 2)
    return expo


def compute_weight_map(img):
    """
    计算像素级融合权重图
    组合: 对比度 × 饱和度 × 曝光良好度
    """
    contrast = _laplacian_contrast(img)
    saturation = _saturation_measure(img)
    exposed = _well_exposedness(img)

    weight = contrast * saturation * exposed
    weight += 1e-6  # 防止除零
    return weight


# ==================== 金字塔构建 ====================

def build_gaussian_pyramid(img, levels):
    """构建高斯金字塔"""
    pyramid = [img.astype(np.float32)]
    for _ in range(levels - 1):
        img = cv2.pyrDown(pyramid[-1])
        pyramid.append(img)
    return pyramid

def build_laplacian_pyramid(img, levels):
    """构建拉普拉斯金字塔"""
    gaussian = [img.astype(np.float32)]
    for _ in range(levels - 1):
        img = cv2.pyrDown(gaussian[-1])
        gaussian.append(img)

    laplacian = []
    for i in range(levels - 1):
        size = (gaussian[i].shape[1], gaussian[i].shape[0])
        up = cv2.pyrUp(gaussian[i + 1], dstsize=size)
        lap = gaussian[i] - up
        laplacian.append(lap)
    laplacian.append(gaussian[-1])
    return laplacian


def _pyramid_reconstruct(laplacian_pyr):
    """从拉普拉斯金字塔重建图像"""
    result = laplacian_pyr[-1]
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        size = (laplacian_pyr[i].shape[1], laplacian_pyr[i].shape[0])
        up = cv2.pyrUp(result, dstsize=size)
        result = up + laplacian_pyr[i]
    return result


# ==================== 多尺度融合主函数 ====================

def fuse_underwater(img, fusion_configs=None, pyramid_levels=5,
                    collect_intermediate=False):
    """
    多尺度金字塔融合主入口

    流程：
    1. 对同一输入图像，用多组不同参数运行完整增强流水线
    2. 为每个增强版本计算权重图（对比度 x 饱和度 x 曝光）
    3. 权重图 → 高斯金字塔
    4. 增强版本 → 拉普拉斯金字塔
    5. 逐层加权融合
    6. 金字塔重建 → 最终融合图像

    参数:
        img: 输入 BGR 图像
        fusion_configs: 多参数配置列表，默认使用内置5组
        pyramid_levels: 金字塔层数，默认5
        collect_intermediate: 是否返回各增强版本
    返回:
        result: 融合后 BGR 图像
        versions: 各增强版本列表（仅 collect_intermediate=True 时）
    """
    if fusion_configs is None:
        fusion_configs = FUSION_CONFIGS

    h, w = img.shape[:2]
    max_levels = int(np.log2(min(h, w))) - 2
    levels = min(pyramid_levels, max_levels)
    levels = max(2, levels)

    print(f"\n  [Fusion] 使用 {len(fusion_configs)} 组参数, {levels} 层金字塔")

    # 第一步：多参数增强生成多个版本
    versions = []
    for cfg in fusion_configs:
        enhancer = UnderwaterEnhancer()
        enhancer.set_all_params(
            white_balance=cfg.get("white_balance", {}),
            red_channel=cfg.get("red_channel", {}),
            clahe=cfg.get("clahe", {}),
            dehaze=cfg.get("dehaze", {}),
            unsharp_mask=cfg.get("unsharp_mask", {}),
            gamma=cfg.get("gamma", {}),
        )
        enhanced = enhancer.process(img.copy(), collect_intermediate=False)
        versions.append(enhanced.astype(np.float32))
        print(f"    [{cfg['name']}] 完成")

    # 第二步：计算每个版本的权重图
    weight_maps = []
    for i, ver in enumerate(versions):
        w = compute_weight_map(ver.astype(np.uint8))
        weight_maps.append(w)

    # 第三步：权重归一化（每个像素位置所有版本权重和为1）
    weight_sum = sum(weight_maps)
    weight_maps = [w / (weight_sum + 1e-8) for w in weight_maps]

    # 第四步：构建高斯金字塔（权重）+ 拉普拉斯金字塔（增强版本）
    weight_pyrs = [build_gaussian_pyramid(w, levels) for w in weight_maps]
    version_pyrs = [build_laplacian_pyramid(v, levels) for v in versions]

    # 第五步：逐层融合
    fused_lap_pyr = []
    for lev in range(levels):
        h_lev, w_lev = weight_pyrs[0][lev].shape[:2]
        fused_layer = np.zeros_like(version_pyrs[0][lev])
        for i in range(len(versions)):
            # 权重广播到3通道
            w_3ch = np.repeat(weight_pyrs[i][lev][:, :, np.newaxis], 3, axis=2)
            fused_layer += w_3ch * version_pyrs[i][lev]
        fused_lap_pyr.append(fused_layer)

    # 第六步：重建
    result_f = _pyramid_reconstruct(fused_lap_pyr)
    result = np.clip(result_f, 0, 255).astype(np.uint8)

    if collect_intermediate:
        return result, [v.astype(np.uint8) for v in versions]
    return result


def create_version_configs(water_type=None):
    """
    根据水体类型创建定制参数配置
    可用: "yellow" / "green" / "neutral" / None(全用)
    """
    if water_type == "yellow":
        return [
            FUSION_CONFIGS[0],  # Default
            FUSION_CONFIGS[1],  # Anti-Yellow (重点配置，放在前面)
            FUSION_CONFIGS[4],  # Mild
        ]
    elif water_type == "green":
        return [
            FUSION_CONFIGS[0],  # Default
            FUSION_CONFIGS[2],  # Anti-Green
            FUSION_CONFIGS[3],  # Dehaze-Plus
        ]
    elif water_type == "neutral":
        return FUSION_CONFIGS[:3]  # 前3组
    else:
        # 全部5组
        return FUSION_CONFIGS
