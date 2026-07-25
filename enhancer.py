"""
水下图像增强系统 - 增强处理模块
包含6个级联物理增强算法 + 流水线控制
"""

import cv2
import numpy as np
# ==================== 模块2: 白平衡校正 ====================

def white_balance(img, a_shift=0, b_shift=0):
    """
    LAB颜色空间白平衡校正
    将A/B通道均值归一化到128
    参数:
        img: BGR图像 (np.ndarray)
        a_shift: A通道偏移 (范围-50~50), 默认0
        b_shift: B通道偏移 (范围-50~50), 默认0
    返回:
        白平衡校正后BGR图像
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    l, a, b = cv2.split(lab)

    # A/B通道均值归一化到128
    a_mean = np.mean(a) + a_shift
    b_mean = np.mean(b) + b_shift

    a = a - a_mean + 128.0
    b = b - b_mean + 128.0

    # 裁剪到有效范围
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    l = np.clip(l, 0, 255)

    lab_corrected = cv2.merge([l, a, b]).astype(np.uint8)
    result = cv2.cvtColor(lab_corrected, cv2.COLOR_Lab2BGR)

    return np.clip(result, 0, 255).astype(np.uint8)


# ==================== 模块3: 红色通道恢复 ====================

def red_channel_recovery(img, strength=0.3):
    """
    红色通道恢复：直方图均衡化 + 加权融合
    参数:
        img: BGR图像
        strength: 融合强度 (范围0.1~0.5), 默认0.3
    返回:
        红色恢复后BGR图像
    """
    b, g, r = cv2.split(img)
    # 对红色通道做直方图均衡化
    r_eq = cv2.equalizeHist(r)
    # 加权融合原红色通道与均衡化结果
    r_recovered = cv2.addWeighted(r, 1.0 - strength, r_eq, strength, 0)
    result = cv2.merge([b, g, r_recovered])
    return result

# ==================== 模块4: CLAHE增强 ====================

def clahe_enhance(img, clip_limit=2.0, grid_size=(8, 8)):
    """
    CLAHE增强：在LAB空间对亮度通道做自适应直方图均衡化
    参数:
        img: BGR图像
        clip_limit: 对比度限制阈值 (范围0.5~4.0), 默认2.0
        grid_size: 网格大小, 默认(8, 8)
    返回:
        CLAHE增强后BGR图像
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    # 创建CLAHE对象并处理L通道
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)
    return result


# ==================== 模块5: 暗通道去雾 ====================

def _dark_channel(img, kernel_size=15):
    """计算暗通道图"""
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dark = cv2.erode(min_channel, kernel)
    return dark


def _atmospheric_light(img, dark_channel, top_percent=0.001):
    """估计大气光值（最亮0.1%像素）"""
    h, w = dark_channel.shape
    num_pixels = h * w
    num_top = int(max(num_pixels * top_percent, 1))

    # 暗通道中最亮像素的位置
    dark_flat = dark_channel.flatten()
    indices = np.argpartition(dark_flat, -num_top)[-num_top:]

    # 在这些位置取原始图像中的最大值作为大气光
    img_flat = img.reshape(-1, 3)
    top_pixels = img_flat[indices]
    # 取暗通道值最大的像素在原图中对应的最亮值
    best_idx = np.argmax(dark_flat[indices])
    return top_pixels[best_idx].astype(np.float32)


def dark_channel_dehaze(img, omega=0.75, t_min=0.35,
                        use_guided_filter=True, kernel_size=15):
    """
    暗通道去雾算法

    参数:
        img: BGR图像
        omega: 保留雾度的系数 (范围0.5~0.95), 默认0.75
        t_min: 最小透射率 (范围0.2~0.5), 默认0.35
        use_guided_filter: 是否使用导向滤波, 默认True
        kernel_size: 暗通道核大小, 默认15
    返回:
        去雾后BGR图像
    """
    img_f = img.astype(np.float32) / 255.0

    # 1. 暗通道计算
    dark = _dark_channel(img_f, kernel_size)

    # 2. 大气光估计
    A = _atmospheric_light(img_f, dark)

    # 3. 透射率估计
    dark_norm = _dark_channel(img_f / A, kernel_size)
    t = 1.0 - omega * dark_norm

    # 4. 导向滤波优化透射率
    if use_guided_filter:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        try:
            t = cv2.ximgproc.guidedFilter(gray, t, kernel_size * 2 + 1, 1e-4)
        except AttributeError:
            # 如果cv2.ximgproc不可用，使用双边滤波替代
            t = cv2.bilateralFilter(t.astype(np.float32), 9, 75, 75)

    # 5. 透射率裁剪
    t = np.maximum(t, t_min)

    # 6. 场景辐射恢复
    result = np.empty_like(img_f)
    for i in range(3):
        result[:, :, i] = (img_f[:, :, i] - A[i]) / t + A[i]

    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result


# ==================== 模块6: 反锐化掩模锐化 ====================

def unsharp_mask(img, amount=1.2, radius=0.5):
    """
    反锐化掩模锐化：高斯模糊 + 加权叠加

    参数:
        img: BGR图像
        amount: 锐化强度 (范围1.0~2.0), 默认1.2
        radius: 模糊半径 (范围0.3~1.0), 默认0.5
    返回:
        锐化后BGR图像
    """
    # 根据radius计算高斯核大小和sigma
    ksize = int(radius * 4) | 1  # 确保为奇数
    ksize = max(3, ksize)

    blurred = cv2.GaussianBlur(img, (ksize, ksize), radius)
    result = cv2.addWeighted(img, amount, blurred, 1.0 - amount, 0)

    return np.clip(result, 0, 255).astype(np.uint8)


# ==================== 模块7: Gamma校正 ====================

def gamma_correction(img, gamma=1.1):
    """
    Gamma校正：通过查找表实现非线性映射
    参数:
        img: BGR图像
        gamma: Gamma值 (范围0.8~1.5), 默认1.1
    返回:
        Gamma校正后BGR图像
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in range(256)]).astype(np.uint8)
    result = cv2.LUT(img, table)
    return result


# ==================== 模块8: 流水线控制 ====================

class UnderwaterEnhancer:
    """水下图像增强流水线控制器"""

    def __init__(self):
        # 默认参数配置
        self.params = {
            "white_balance": {"a_shift": 0, "b_shift": 0},
            "red_channel": {"strength": 0.3},
            "clahe": {"clip_limit": 2.0, "grid_size": (8, 8)},
            "dehaze": {"omega": 0.75, "t_min": 0.35,
                       "use_guided_filter": True, "kernel_size": 15},
            "unsharp_mask": {"amount": 1.2, "radius": 0.5},
            "gamma": {"gamma": 1.1},
        }
        self._intermediate = {}
        self._step_names = [
            "01_Original", "02_White_Balance", "03_Red_Channel",
            "04_CLAHE", "05_Dehaze", "06_Unsharp_Mask", "07_Gamma"
        ]

    def set_params(self, module_name, **kwargs):
        """批量设置某模块的参数"""
        if module_name in self.params:
            self.params[module_name].update(kwargs)
            print(f"[Pipeline] {module_name} 参数已更新: {self.params[module_name]}")
        else:
            available = list(self.params.keys())
            raise ValueError(f"未知模块 '{module_name}'，可用模块: {available}")

    def set_all_params(self, **module_params):
        """批量设置所有模块参数
        用法: set_all_params(white_balance={'a_shift': 5}, gamma={'gamma': 1.2})
        """
        for module_name, params in module_params.items():
            self.set_params(module_name, **params)

    def get_intermediate(self):
        """返回所有中间步骤结果"""
        return dict(self._intermediate)

    def process(self, img, collect_intermediate=True):
        """
        完整处理流程：按顺序调用各增强模块
        参数:
            img: 输入BGR图像
            collect_intermediate: 是否收集中间结果
        返回:
            增强后BGR图像
        """
        self._intermediate = {}

        current = img.copy()
        if collect_intermediate:
            self._intermediate[self._step_names[0]] = current

        # 步骤1: 白平衡校正
        p = self.params["white_balance"]
        current = white_balance(current, **p)
        if collect_intermediate:
            self._intermediate[self._step_names[1]] = current

        # 步骤2: 红色通道恢复
        p = self.params["red_channel"]
        current = red_channel_recovery(current, **p)
        if collect_intermediate:
            self._intermediate[self._step_names[2]] = current

        # 步骤3: CLAHE增强
        p = self.params["clahe"]
        current = clahe_enhance(current, **p)
        if collect_intermediate:
            self._intermediate[self._step_names[3]] = current

        # 步骤4: 暗通道去雾
        p = self.params["dehaze"]
        current = dark_channel_dehaze(current, **p)
        if collect_intermediate:
            self._intermediate[self._step_names[4]] = current

        # 步骤5: 反锐化掩模
        p = self.params["unsharp_mask"]
        current = unsharp_mask(current, **p)
        if collect_intermediate:
            self._intermediate[self._step_names[5]] = current

        # 步骤6: Gamma校正
        p = self.params["gamma"]
        current = gamma_correction(current, **p)
        if collect_intermediate:
            self._intermediate[self._step_names[6]] = current

        return current
