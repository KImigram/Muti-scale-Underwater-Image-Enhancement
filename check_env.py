"""
环境检测脚本 - 检查所有依赖是否正常可用
"""

import sys

print("=" * 50)
print("  水下图像增强系统 - 环境检测")
print("=" * 50)

# 1. Python 版本
print(f"\n[1] Python 版本: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
if sys.version_info < (3, 7):
    print("    警告: 建议使用 Python >= 3.7")
else:
    print("    ✓ OK")

# 2. 基础科学计算库
modules_basic = [
    ("numpy", "np"),
    ("matplotlib", None),
]

for mod_name, alias in modules_basic:
    try:
        mod = __import__(mod_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"[ ] {mod_name}: {version}  ✓ OK")
    except ImportError as e:
        print(f"[ ] {mod_name}: ✗ 缺失 ({e})")

# 3. OpenCV (核心依赖)
try:
    import cv2
    print(f"[ ] cv2: {cv2.__version__}  ✓ OK")

    # 检查 contrib 模块（导向滤波需要）
    try:
        gf = cv2.ximgproc.guidedFilter
        print("    ximgproc.guidedFilter: 可用  ✓ OK")
    except AttributeError:
        print("    ximgproc.guidedFilter: ✗ 不可用（将使用双边滤波替代）")

except ImportError as e:
    print(f"[ ] cv2: ✗ 缺失 ({e})")

# 4. scikit-image (SSIM 需要)
try:
    import skimage
    from skimage.metrics import structural_similarity
    print(f"[ ] scikit-image: {skimage.__version__}  ✓ OK")
    print("    structural_similarity: 可用  ✓ OK")
except ImportError as e:
    print(f"[ ] scikit-image: ✗ 缺失 ({e})")

# 5. PyYAML (配置文件需要)
try:
    import yaml
    print(f"[ ] PyYAML: 可用  ✓ OK")
except ImportError:
    print(f"[ ] PyYAML: ✗ 缺失（config.yaml 加载将不可用）")

# 6. 功能验证
print(f"\n{'='*50}")
print("  功能验证")
print(f"{'='*50}")

# 6.1 NumPy 基础运算
try:
    import numpy as np
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr = np.clip(arr, 0, 255)
    print("[ ] np.ndarray + np.clip: ✓ OK")
except Exception as e:
    print(f"[ ] np.ndarray + np.clip: ✗ ({e})")

# 6.2 OpenCV 基础功能
try:
    import cv2
    dummy = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # 颜色空间转换
    lab = cv2.cvtColor(dummy, cv2.COLOR_BGR2Lab)
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    print("[ ] cv2.cvtColor (BGR↔Lab): ✓ OK")

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(np.zeros((100, 100), dtype=np.uint8))
    print("[ ] cv2.createCLAHE: ✓ OK")

    # 滤波 / 形态学
    blurred = cv2.GaussianBlur(dummy, (3, 3), 0.5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    eroded = cv2.erode(np.zeros((100, 100), dtype=np.uint8), kernel)
    print("[ ] cv2.GaussianBlur + cv2.erode: ✓ OK")

    # addWeighted
    fused = cv2.addWeighted(dummy, 0.7, dummy, 0.3, 0)
    print("[ ] cv2.addWeighted: ✓ OK")

    # equalizeHist
    r_eq = cv2.equalizeHist(np.zeros((100, 100), dtype=np.uint8))
    print("[ ] cv2.equalizeHist: ✓ OK")

    # LUT
    table = np.arange(256, dtype=np.uint8)
    gamma_result = cv2.LUT(dummy, table)
    print("[ ] cv2.LUT: ✓ OK")

    # PSNR
    psnr = cv2.PSNR(dummy, dummy)
    print("[ ] cv2.PSNR: ✓ OK")

except Exception as e:
    print(f"[ ] OpenCV 功能验证失败: {e}")

# 6.3 SSIM 验证
try:
    from skimage.metrics import structural_similarity as ssim
    gray1 = np.random.rand(100, 100).astype(np.float32)
    gray2 = gray1.copy()
    val = ssim(gray1, gray2, data_range=1.0)
    print(f"[ ] SSIM 计算: ✓ OK (测试值={val:.2f})")
except Exception as e:
    print(f"[ ] SSIM 验证失败: {e}")

# 6.4 matplotlib 可视化
try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互模式，防止弹出窗口
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((10, 10, 3)))
    plt.close(fig)
    print("[ ] matplotlib 绘图: ✓ OK")
except Exception as e:
    print(f"[ ] matplotlib 验证失败: {e}")

# 7. 总结
print(f"\n{'='*50}")
print("  环境检测完成")
print(f"{'='*50}\n")
