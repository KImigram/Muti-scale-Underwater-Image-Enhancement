"""
Multi-Scale Underwater Image Enhancement - GUI
预览 + 处理控制 + 实时显示
"""

import os
import time
import threading
import queue

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from enhancer import (
    white_balance, red_channel_recovery, clahe_enhance,
    dark_channel_dehaze, unsharp_mask, gamma_correction,
)
from fusion import fuse_underwater, FUSION_CONFIGS, create_version_configs


# ==================== 辅助函数 ====================

def _classify_water_type(img):
    """基于LAB颜色空间识别水体类型"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    _, a, b = cv2.split(lab)
    mean_a = np.mean(a) - 128.0
    mean_b = np.mean(b) - 128.0
    threshold = 5.0
    if mean_b > threshold and mean_b > abs(mean_a):
        return "yellow"
    elif mean_a < -threshold and abs(mean_a) > abs(mean_b):
        return "green"
    return "neutral"


def _cv2_to_tk(img_bgr, max_w=600, max_h=400):
    """OpenCV BGR图像 -> 缩放后的 ImageTk.PhotoImage"""
    if not PIL_AVAILABLE:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pil_img = Image.fromarray(img_resized)
    return ImageTk.PhotoImage(pil_img)


# ==================== 图像管理窗口 ====================

class ImageManagerWindow(tk.Toplevel):
    """管理待处理图像：上传、删除、预览"""

    def __init__(self, parent, input_dir, on_select_callback=None):
        super().__init__(parent)
        self.title("Image Manager - 图像管理")
        self.geometry("700x500")
        self.parent = parent
        self.input_dir = input_dir
        self.on_select_callback = on_select_callback
        self._current_image = None

        os.makedirs(input_dir, exist_ok=True)

        self._build_ui()
        self._refresh_list()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build_ui(self):
        # 顶部按钮栏
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_frame, text="+ Upload Image", command=self._upload).pack(
            side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="- Delete Selected", command=self._delete).pack(
            side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Refresh", command=self._refresh_list).pack(
            side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Select for Processing",
                   command=self._select_for_processing).pack(side=tk.RIGHT, padx=3)

        # 中间区域：列表 + 预览
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧：文件列表
        list_frame = ttk.LabelFrame(paned, text="Image List")
        paned.add(list_frame, weight=1)

        self.listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        # 右侧：预览
        preview_frame = ttk.LabelFrame(paned, text="Preview")
        paned.add(preview_frame, weight=2)

        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # 底部信息
        self.info_label = ttk.Label(
            self, text=f"Directory: {self.input_dir}")
        self.info_label.pack(fill=tk.X, padx=5, pady=2)

    def _refresh_list(self):
        self.listbox.delete(0, tk.END)
        if not os.path.isdir(self.input_dir):
            return
        supported = (".jpg", ".jpeg", ".png", ".bmp")
        files = sorted([f for f in os.listdir(self.input_dir)
                        if f.lower().endswith(supported)])
        for f in files:
            self.listbox.insert(tk.END, f)
        self.info_label.config(
            text=f"Directory: {self.input_dir}  ({len(files)} images)")

    def _on_select(self, event):
        sel = self.listbox.curselection()
        if not sel:
            return
        filename = self.listbox.get(sel[0])
        path = os.path.join(self.input_dir, filename)
        img = cv2.imread(path)
        if img is not None:
            tk_img = _cv2_to_tk(img, max_w=400, max_h=350)
            if tk_img:
                self.preview_label.config(image=tk_img)
                self.preview_label.image = tk_img
            self._current_image = path

    def _upload(self):
        paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                       ("All Files", "*.*")])
        if not paths:
            return
        for src in paths:
            dst = os.path.join(self.input_dir, os.path.basename(src))
            img = cv2.imread(src)
            if img is not None:
                cv2.imwrite(dst, img)
            else:
                messagebox.showwarning("Warning",
                                       f"Cannot read: {os.path.basename(src)}")
        self._refresh_list()

    def _delete(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("Info", "Please select an image to delete.")
            return
        filename = self.listbox.get(sel[0])
        if messagebox.askyesno("Confirm", f"Delete '{filename}'?"):
            path = os.path.join(self.input_dir, filename)
            if os.path.isfile(path):
                os.remove(path)
            self._refresh_list()

    def _select_for_processing(self):
        if self._current_image and self.on_select_callback:
            self.on_select_callback(self._current_image)
            self.destroy()


# ==================== 主窗口 ====================

class MainWindow:
    """水下图像增强系统主界面"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Underwater Image Enhancement System")
        self.root.geometry("1050x650")
        self.root.minsize(900, 550)

        self.current_image_path = None
        self.current_image = None
        self.enhanced_image = None
        self.processing = False
        self._update_queue = queue.Queue()
        self._step_names = [
            "01_Original", "02_White_Balance", "03_Red_Channel",
            "04_CLAHE", "05_Dehaze", "06_Unsharp_Mask", "07_Gamma"
        ]

        self._build_ui()
        self._poll_queue()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------------------- UI 构建 --------------------

    def _build_ui(self):
        # 顶部路径栏
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=8, pady=5)

        ttk.Label(top_frame, text="Image:").pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value="(No image selected)")
        ttk.Entry(top_frame, textvariable=self.path_var, state="readonly",
                  width=60).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(top_frame, text="Browse...", command=self._browse_image).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="Image Manager",
                   command=self._open_image_manager).pack(side=tk.LEFT, padx=2)

        # 主体：左侧预览 + 右侧控制
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=5)

        # ---- 左侧：预览区 ----
        preview_frame = ttk.LabelFrame(main_paned, text="Preview")
        main_paned.add(preview_frame, weight=3)

        self.preview_canvas = tk.Canvas(preview_frame, bg="#1a1a2e",
                                         width=600, height=450)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self._canvas_img_id = None
        self._canvas_text_id = None

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(preview_frame, textvariable=self.status_var,
                                       anchor=tk.CENTER, font=("", 10))
        self.status_label.pack(fill=tk.X, padx=3, pady=2)

        # ---- 右侧：控制面板 ----
        ctrl_frame = ttk.Frame(main_paned)
        main_paned.add(ctrl_frame, weight=2)

        ctrl_canvas = tk.Canvas(ctrl_frame, width=280)
        ctrl_scroll = ttk.Scrollbar(ctrl_frame, orient=tk.VERTICAL,
                                     command=ctrl_canvas.yview)
        self.ctrl_inner = ttk.Frame(ctrl_canvas)
        self.ctrl_inner.bind("<Configure>",
                              lambda e: ctrl_canvas.configure(
                                  scrollregion=ctrl_canvas.bbox("all")))
        ctrl_canvas.create_window((0, 0), window=self.ctrl_inner, anchor="nw")
        ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)
        ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ctrl_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_control_panel(self.ctrl_inner)
        self._show_default_preview()

    def _build_control_panel(self, parent):
        """构建右侧控制面板"""
        row = 0

        # -- 模式选择 --
        ttk.Label(parent, text="Enhancement Mode", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(8, 2))
        row += 1

        self.mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(parent, text="Single Pipeline",
                        variable=self.mode_var, value="single",
                        command=self._on_mode_change).grid(
            row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Radiobutton(parent, text="Multi-scale Fusion",
                        variable=self.mode_var, value="fusion",
                        command=self._on_mode_change).grid(
            row=row, column=0, columnspan=2, sticky="w")
        row += 1

        # -- 参数区域（单流水线模式）--
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1

        self.single_params_frame = ttk.LabelFrame(
            parent, text="Single Pipeline Parameters")
        self.single_params_frame.grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=3)
        self._build_single_params(self.single_params_frame)
        row += 1

        # -- 融合模式选项 --
        self.fusion_frame = ttk.LabelFrame(parent, text="Fusion Options")
        self.fusion_frame.grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=3)

        ttk.Label(self.fusion_frame, text="Water Type:").grid(
            row=0, column=0, sticky="w", padx=3, pady=2)
        self.fusion_water_var = tk.StringVar(value="auto")
        ttk.Combobox(self.fusion_frame, textvariable=self.fusion_water_var,
                     values=["auto", "yellow", "green", "neutral", "all"],
                     state="readonly", width=12).grid(
            row=0, column=1, sticky="w", padx=3, pady=2)

        # 默认隐藏融合帧
        self.fusion_frame.grid_remove()

        # -- 开始按钮 --
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row + 1, column=0, columnspan=2, sticky="ew", pady=8)

        self.process_btn = ttk.Button(
            parent, text="Start Processing",
            command=self._start_processing)
        self.process_btn.grid(
            row=row + 2, column=0, columnspan=2, pady=5, ipady=4)

        # -- 质量指标 --
        self.metrics_frame = ttk.LabelFrame(parent, text="Quality Metrics")
        self.metrics_frame.grid(
            row=row + 3, column=0, columnspan=2, sticky="ew", pady=8)

        self.metric_labels = {}
        for i, key in enumerate(["UIQM", "UICM", "UISM", "UIConM", "UCIQE"]):
            ttk.Label(self.metrics_frame, text=f"{key}:").grid(
                row=i, column=0, sticky="w", padx=3, pady=1)
            lbl = ttk.Label(self.metrics_frame, text="--", width=12, anchor="e")
            lbl.grid(row=i, column=1, sticky="e", padx=3, pady=1)
            self.metric_labels[key] = lbl

        # -- 保存按钮 --
        self.save_btn = ttk.Button(parent, text="Save Result",
                                   command=self._save_result, state=tk.DISABLED)
        self.save_btn.grid(
            row=row + 4, column=0, columnspan=2, pady=5, ipady=4)

    def _build_single_params(self, parent):
        """构建单流水线参数控件"""
        row = 0

        ttk.Label(parent, text="a_shift (-50~50):").grid(
            row=row, column=0, sticky="w", padx=3, pady=1)
        self.param_a_shift = tk.Scale(parent, from_=-50, to=50, orient=tk.HORIZONTAL,
                                       length=160, resolution=1)
        self.param_a_shift.set(0)
        self.param_a_shift.grid(row=row, column=1, sticky="w", padx=3)
        row += 1

        ttk.Label(parent, text="b_shift (-50~50):").grid(
            row=row, column=0, sticky="w", padx=3, pady=1)
        self.param_b_shift = tk.Scale(parent, from_=-50, to=50, orient=tk.HORIZONTAL,
                                       length=160, resolution=1)
        self.param_b_shift.set(0)
        self.param_b_shift.grid(row=row, column=1, sticky="w", padx=3)
        row += 1

        ttk.Label(parent, text="red_strength (0.1~0.5):").grid(
            row=row, column=0, sticky="w", padx=3, pady=1)
        self.param_red = tk.Scale(parent, from_=0.1, to=0.5, orient=tk.HORIZONTAL,
                                   length=160, resolution=0.01)
        self.param_red.set(0.3)
        self.param_red.grid(row=row, column=1, sticky="w", padx=3)
        row += 1

        ttk.Label(parent, text="clahe_limit (0.5~4.0):").grid(
            row=row, column=0, sticky="w", padx=3, pady=1)
        self.param_clahe = tk.Scale(parent, from_=0.5, to=4.0, orient=tk.HORIZONTAL,
                                     length=160, resolution=0.5)
        self.param_clahe.set(2.0)
        self.param_clahe.grid(row=row, column=1, sticky="w", padx=3)
        row += 1

        ttk.Label(parent, text="dehaze_omega (0.5~0.95):").grid(
            row=row, column=0, sticky="w", padx=3, pady=1)
        self.param_omega = tk.Scale(parent, from_=0.5, to=0.95, orient=tk.HORIZONTAL,
                                     length=160, resolution=0.01)
        self.param_omega.set(0.75)
        self.param_omega.grid(row=row, column=1, sticky="w", padx=3)
        row += 1

        ttk.Label(parent, text="dehaze_tmin (0.2~0.5):").grid(
            row=row, column=0, sticky="w", padx=3, pady=1)
        self.param_tmin = tk.Scale(parent, from_=0.2, to=0.5, orient=tk.HORIZONTAL,
                                    length=160, resolution=0.01)
        self.param_tmin.set(0.35)
        self.param_tmin.grid(row=row, column=1, sticky="w", padx=3)
        row += 1

        self.param_guided_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="use_guided_filter",
                        variable=self.param_guided_var).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=3, pady=1)
        row += 1

        ttk.Label(parent, text="sharpen_amount (1.0~2.0):").grid(
            row=row, column=0, sticky="w", padx=3, pady=1)
        self.param_sharpen = tk.Scale(parent, from_=1.0, to=2.0, orient=tk.HORIZONTAL,
                                       length=160, resolution=0.1)
        self.param_sharpen.set(1.2)
        self.param_sharpen.grid(row=row, column=1, sticky="w", padx=3)
        row += 1

        ttk.Label(parent, text="gamma (0.8~1.5):").grid(
            row=row, column=0, sticky="w", padx=3, pady=1)
        self.param_gamma = tk.Scale(parent, from_=0.8, to=1.5, orient=tk.HORIZONTAL,
                                     length=160, resolution=0.01)
        self.param_gamma.set(1.1)
        self.param_gamma.grid(row=row, column=1, sticky="w", padx=3)

    # -------------------- 模式切换 --------------------

    def _on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "single":
            self.fusion_frame.grid_remove()
            self.single_params_frame.grid()
        else:
            self.single_params_frame.grid_remove()
            self.fusion_frame.grid()

    # -------------------- 按钮事件 --------------------

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                       ("All Files", "*.*")])
        if path:
            self._load_image(path)

    def _open_image_manager(self):
        input_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__) or ".", "tests", "test_images", "original"))
        ImageManagerWindow(self.root, input_dir,
                            on_select_callback=self._load_image)

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Cannot read image: {path}")
            return
        self.current_image_path = path
        self.current_image = img
        self.path_var.set(path)
        self._display_preview(img, label="Original")
        self.status_var.set("Image loaded. Ready to process.")
        self._clear_metrics()
        for lbl in self.metric_labels.values():
            lbl.config(text="--")
        self.save_btn.config(state=tk.DISABLED)
        self.enhanced_image = None

    # -------------------- 处理逻辑 --------------------

    def _start_processing(self):
        if self.current_image is None:
            messagebox.showinfo("Info", "Please select an image first.")
            return
        if self.processing:
            messagebox.showinfo("Info", "Processing in progress, please wait.")
            return

        self.processing = True
        self.process_btn.config(text="Processing...", state=tk.DISABLED)
        self._clear_metrics()

        mode = self.mode_var.get()
        thread = threading.Thread(
            target=self._process_thread, args=(mode,), daemon=True)
        thread.start()

    def _process_thread(self, mode):
        """后台处理线程"""
        img = self.current_image

        if mode == "single":
            self._process_single(img)
        else:
            self._process_fusion(img)

        self._update_queue.put(("done", None))
        self._update_queue.put(("status", "Processing complete."))

    def _process_single(self, img):
        """单流水线分步处理，每步发送中间结果到队列"""
        params = {
            "a_shift": self.param_a_shift.get(),
            "b_shift": self.param_b_shift.get(),
            "red_strength": self.param_red.get(),
            "clahe_limit": self.param_clahe.get(),
            "dehaze_omega": self.param_omega.get(),
            "dehaze_tmin": self.param_tmin.get(),
            "use_guided_filter": self.param_guided_var.get(),
            "sharpen_amount": self.param_sharpen.get(),
            "gamma_val": self.param_gamma.get(),
        }

        steps = [
            ("01_Original", None),
            ("02_White_Balance", lambda x: white_balance(
                x, a_shift=params["a_shift"], b_shift=params["b_shift"])),
            ("03_Red_Channel", lambda x: red_channel_recovery(
                x, strength=params["red_strength"])),
            ("04_CLAHE", lambda x: clahe_enhance(
                x, clip_limit=params["clahe_limit"], grid_size=(8, 8))),
            ("05_Dehaze", lambda x: dark_channel_dehaze(
                x, omega=params["dehaze_omega"], t_min=params["dehaze_tmin"],
                use_guided_filter=params["use_guided_filter"])),
            ("06_Unsharp_Mask", lambda x: unsharp_mask(
                x, amount=params["sharpen_amount"], radius=0.5)),
            ("07_Gamma", lambda x: gamma_correction(
                x, gamma=params["gamma_val"])),
        ]

        current = img.copy()
        for name, func in steps:
            self._update_queue.put(("step_start", name, current))
            time.sleep(0.1)  # 给 GUI 时间刷新
            if func is not None:
                current = func(current)
                self._update_queue.put(("step_done", name, current))

        # 计算指标
        from utils import evaluate_uiqm, evaluate_uciqe
        uiqm = evaluate_uiqm(current)
        uciqe = evaluate_uciqe(current)
        self._update_queue.put(("metrics", uiqm, uciqe, current))

    def _process_fusion(self, img):
        """多尺度融合处理"""
        self._update_queue.put(("step_start", "Fusion", img))
        self._update_queue.put(("status", "Running multi-scale fusion..."))

        water_sel = self.fusion_water_var.get()
        if water_sel == "auto":
            water_type = _classify_water_type(img)
            cfgs = create_version_configs(water_type)
        elif water_sel == "all":
            cfgs = FUSION_CONFIGS
        else:
            cfgs = create_version_configs(water_sel)

        result = fuse_underwater(
            img, fusion_configs=cfgs,
            pyramid_levels=5, collect_intermediate=False)

        from utils import evaluate_uiqm, evaluate_uciqe
        uiqm = evaluate_uiqm(result)
        uciqe = evaluate_uciqe(result)
        self._update_queue.put(("fusion_result", result, uiqm, uciqe))

    # -------------------- 队列轮询 --------------------

    def _poll_queue(self):
        """在主线程中轮询队列，更新 GUI"""
        try:
            while True:
                msg = self._update_queue.get_nowait()
                msg_type = msg[0]

                if msg_type == "step_start":
                    _, name, img = msg
                    self._display_preview(img, label=name)
                    self.status_var.set(f"Processing: {name}")

                elif msg_type == "step_done":
                    _, name, img = msg
                    self._display_preview(img, label=name)
                    self.status_var.set(f"Done: {name}")

                elif msg_type == "metrics":
                    _, uiqm, uciqe, enhanced = msg
                    self.enhanced_image = enhanced
                    self._update_metrics(uiqm, uciqe)
                    self._display_preview(enhanced, label="Enhanced Result")
                    self.processing = False
                    self.process_btn.config(
                        text="Start Processing", state=tk.NORMAL)
                    self.save_btn.config(state=tk.NORMAL)
                    self.status_var.set("Processing complete.")

                elif msg_type == "fusion_result":
                    _, result, uiqm, uciqe = msg
                    self.enhanced_image = result
                    self._update_metrics(uiqm, uciqe)
                    self._display_preview(result, label="Fusion Result")
                    self.processing = False
                    self.process_btn.config(
                        text="Start Processing", state=tk.NORMAL)
                    self.save_btn.config(state=tk.NORMAL)
                    self.status_var.set("Fusion complete.")

                elif msg_type == "status":
                    self.status_var.set(msg[1])

                elif msg_type == "done":
                    pass

        except queue.Empty:
            pass

        self.root.after(100, self._poll_queue)

    def _update_metrics(self, uiqm, uciqe):
        for key in ["UIQM", "UICM", "UISM", "UIConM", "UCIQE"]:
            val = uiqm.get(key) or uciqe.get(key)
            if val is not None:
                self.metric_labels[key].config(text=f"{val:.4f}")

    def _clear_metrics(self):
        for lbl in self.metric_labels.values():
            lbl.config(text="--")

    # -------------------- 预览显示 --------------------

    def _display_preview(self, img_bgr, label="Preview"):
        """在预览 Canvas 上显示图像"""
        if not PIL_AVAILABLE:
            return
        canvas = self.preview_canvas
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 10:
            cw, ch = 600, 450

        tk_img = _cv2_to_tk(img_bgr, max_w=cw - 10, max_h=ch - 40)

        canvas.delete("all")
        cx, cy = cw // 2, (ch - 20) // 2
        canvas.create_image(cx, cy, image=tk_img, anchor=tk.CENTER)
        canvas.create_text(cx, ch - 10, text=label,
                           fill="white", font=("", 10), anchor=tk.S)
        # 保存引用防止被 GC
        canvas._tk_img_ref = tk_img

    def _show_default_preview(self):
        if PIL_AVAILABLE:
            self.preview_canvas.create_text(
                300, 225, text="Select an image to preview",
                fill="#666666", font=("", 14), anchor=tk.CENTER)

    # -------------------- 保存结果 --------------------

    def _save_result(self):
        if self.enhanced_image is None:
            messagebox.showinfo("Info", "No result to save.")
            return
        default_name = "enhanced_result.png"
        if self.current_image_path:
            base = os.path.splitext(os.path.basename(self.current_image_path))[0]
            default_name = f"{base}_enhanced.png"
        path = filedialog.asksaveasfilename(
            title="Save Enhanced Image",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")])
        if path:
            cv2.imwrite(path, self.enhanced_image)
            messagebox.showinfo("Success", f"Saved to:\n{path}")

    # -------------------- 生命周期 --------------------

    def _on_close(self):
        self.processing = False
        self.root.destroy()

    def run(self):
        if not PIL_AVAILABLE:
            messagebox.showwarning(
                "Warning",
                "Pillow (PIL) is required for image preview.\n"
                "Install it with: pip install Pillow\n"
                "The GUI will run but without image preview.")
        self.root.mainloop()


# ==================== 入口 ====================

if __name__ == "__main__":
    app = MainWindow()
    app.run()
