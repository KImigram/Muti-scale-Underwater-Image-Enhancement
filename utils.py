"""
水下图像增强系统 - 工具模块
包含图像I/O、质量评估、可视化功能
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# ==================== 图像I/O ====================

def read_image(path):
    """读取图像，支持jpg/png"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def resize_image(img, max_width=1200):
    """自动尺寸调整：宽>max_width时等比缩放"""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"[IO] 图像已缩放: {w}x{h} -> {new_w}x{new_h}")
    return img


def save_image(img, path):
    """保存图像"""
    cv2.imwrite(path, img)
    print(f"[IO] 图像已保存: {path}")


def get_image_info(img):
    """显示图像信息"""
    h, w = img.shape[:2]
    info = {
        "size": f"{w}x{h}",
        "channels": img.shape[2] if len(img.shape) == 3 else 1,
        "dtype": str(img.dtype),
        "value_range": (img.min(), img.max())
    }
    print(f"[IO] 图像信息: {info}")
    return info


# ==================== 质量评估 ====================

def _safe_float32(img):
    """确保图像为float32格式，值域[0,1]"""
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def _rgb_channels(img):
    """将BGR图像分解为R, G, B三个通道 (float32)"""
    img_f = _safe_float32(img)
    return img_f[:, :, 2], img_f[:, :, 1], img_f[:, :, 0]


def _rgb2gray(img):
    """RGB转灰度图"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def evaluate_uicm(r, g, b):
    """UICM: 水下图像色彩评估 - 计算色彩度"""
    rg = r - g
    yb = 0.5 * (r + g) - b

    mu_rg = np.mean(rg)
    mu_yb = np.mean(yb)
    sigma_rg = np.sqrt(np.mean((rg - mu_rg) ** 2))
    sigma_yb = np.sqrt(np.mean((yb - mu_yb) ** 2))

    return -0.0268 * np.sqrt(sigma_rg ** 2 + sigma_yb ** 2) \
           + 0.1586 * np.sqrt(mu_rg ** 2 + mu_yb ** 2)


def evaluate_uism(r, g, b):
    """UISM: 水下图像清晰度评估 - 利用Sobel边缘检测"""
    def _eme(img_channel):
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 1)
        sobel = np.abs(sobel)
        # 分块计算EME
        h, w = sobel.shape
        block_h, block_w = max(h // 8, 1), max(w // 8, 1)
        eme_vals = []
        for i in range(0, h - block_h + 1, block_h):
            for j in range(0, w - block_w + 1, block_w):
                block = sobel[i:i + block_h, j:j + block_w]
                block = block.flatten()
                valid = block[block > 0]
                if len(valid) > 1:
                    imax = valid.max()
                    imin = valid.min()
                    if imin > 0:
                        eme_vals.append(20 * np.log(imax / imin))
        return np.mean(eme_vals) if eme_vals else 0.0

    return 0.299 * _eme(r) + 0.587 * _eme(g) + 0.114 * _eme(b)


def evaluate_uiconm(r, g, b):
    """UIConM: 水下图像对比度评估"""
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray_scaled = (gray * 255).astype(np.uint8)

    # 局部标准差作为对比度度量
    h, w = gray.shape
    block_h, block_w = max(h // 8, 1), max(w // 8, 1)
    contrast_vals = []
    for i in range(0, h - block_h + 1, block_h):
        for j in range(0, w - block_w + 1, block_w):
            block = gray[i:i + block_h, j:j + block_w]
            contrast_vals.append(np.std(block))
    return np.mean(contrast_vals)


def evaluate_uiqm(img):
    """UIQM: 水下图像质量度量
    返回: {uicm, uism, uiconm, uiqm}
    """
    r, g, b = _rgb_channels(img)
    uicm = evaluate_uicm(r, g, b)
    uism = evaluate_uism(r, g, b)
    uiconm = evaluate_uiconm(r, g, b)
    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    return {
        "UICM": round(uicm, 4),
        "UISM": round(uism, 4),
        "UIConM": round(uiconm, 4),
        "UIQM": round(uiqm, 4)
    }


def evaluate_uciqe(img):
    """UCIQE: 水下彩色图像质量评估
    返回: {uciqe}
    """
    img_f = _safe_float32(img)
    img_lab = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(img_lab)
    l = l.astype(np.float32) / 255.0
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0

    # 色度标准差
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = np.std(chroma)

    # 亮度对比度
    l_sorted = np.sort(l.flatten())
    top_1pct = int(len(l_sorted) * 0.99)
    bottom_1pct = int(len(l_sorted) * 0.01)
    con_l = l_sorted[top_1pct] - l_sorted[bottom_1pct]

    # 饱和度均值
    sat = chroma / (l + 1e-10)
    mu_s = np.mean(sat)

    uciqe = 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s
    return {"UCIQE": round(uciqe, 4)}


def evaluate_psnr(img_orig, img_enhanced):
    """PSNR: 峰值信噪比"""
    psnr_val = cv2.PSNR(img_orig, img_enhanced)
    return {"PSNR": round(psnr_val, 4)}


def evaluate_ssim(img_orig, img_enhanced):
    """SSIM: 结构相似性"""
    gray_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    gray_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)
    ssim_val = ssim(gray_orig, gray_enhanced, data_range=255)
    return {"SSIM": round(ssim_val, 4)}


def evaluate_full(original, enhanced):
    """完整质量评估（需提供原图）
    返回包含所有指标的字典
    """
    results = {}
    results.update(evaluate_uiqm(enhanced))
    results.update(evaluate_uciqe(enhanced))
    results.update(evaluate_psnr(original, enhanced))
    results.update(evaluate_ssim(original, enhanced))
    return results


def evaluate_no_ref(enhanced):
    """无参考质量评估
    返回UIQM和UCIQE
    """
    results = {}
    results.update(evaluate_uiqm(enhanced))
    results.update(evaluate_uciqe(enhanced))
    return results


# ==================== 可视化 ====================

def bgr2rgb(img):
    """BGR转RGB用于matplotlib显示"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_comparison(original, enhanced, title="Original vs Enhanced", save_path=None):
    """并排显示原图与增强图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(bgr2rgb(original))
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(bgr2rgb(enhanced))
    axes[1].set_title("Enhanced", fontsize=12)
    axes[1].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Vis] 对比图已保存: {save_path}")
    plt.show()


def show_pipeline_steps(intermediate_results, step_names=None, save_path=None):
    """显示处理流程各步骤对比"""
    if step_names is None:
        step_names = list(intermediate_results.keys())

    n_steps = len(intermediate_results)
    cols = min(4, n_steps)
    rows = (n_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if n_steps > 1 else [axes]

    for idx, name in enumerate(step_names):
        if name in intermediate_results:
            axes[idx].imshow(bgr2rgb(intermediate_results[name]))
            axes[idx].set_title(name, fontsize=10)
            axes[idx].axis("off")

    # 隐藏多余的子图
    for idx in range(n_steps, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Vis] 流程步骤图已保存: {save_path}")
    plt.show()


def generate_report_image(original, enhanced, intermediate_results,
                          step_names=None, save_path=None):
    """生成完整的报告对比图"""
    if step_names is None:
        step_names = list(intermediate_results.keys())

    n_steps = len(intermediate_results) + 2  # +原图 +最终结果
    cols = 4
    rows = (n_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = axes.flatten()

    images = [("Original", bgr2rgb(original))]
    for name in step_names:
        if name in intermediate_results:
            images.append((name, bgr2rgb(intermediate_results[name])))
    images.append(("Enhanced", bgr2rgb(enhanced)))

    for idx, (title, img) in enumerate(images):
        axes[idx].imshow(img)
        axes[idx].set_title(title, fontsize=9)
        axes[idx].axis("off")

    for idx in range(len(images), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Vis] 报告图已保存: {save_path}")
    plt.show()
