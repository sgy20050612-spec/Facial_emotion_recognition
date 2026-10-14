"""Chinese desktop interface. Nothing starts capturing until Start is pressed."""
import argparse
import logging
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from realtime_fer.download_models import DEFAULT_MODELS
from realtime_fer.engine import LABELS_EN, LABELS_ZH, Recognizer

ROOT = Path(__file__).resolve().parents[1]


def open_capture(source, is_camera):
    if is_camera:
        indices = range(4) if source == '自动' else [int(source)]
        for index in indices:
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        logging.info('Camera opened: index=%s backend=%s', index, backend)
                        return cap
                cap.release()
        raise RuntimeError('没有找到可读取的摄像头。请选择“自动”或编号 0；关闭其他相机/视频通话程序；在 Windows 设置 → 隐私和安全性 → 相机中允许桌面应用访问相机。')
    else:
        cap = cv2.VideoCapture(str(source))
    return cap


def capture_worker(source, is_camera, mirror, models, stop, paused, frames, events):
    cap = None
    try:
        events.put(('status', '正在加载模型…'))
        engine = Recognizer(models)
        if stop.is_set():
            return
        events.put(('status', '正在打开摄像头…' if is_camera else '正在打开视频…'))
        cap = open_capture(source, is_camera)
        if not cap.isOpened():
            raise RuntimeError('无法打开摄像头。请检查 Windows 相机权限、摄像头编号，并关闭占用摄像头的程序。'
                               if is_camera else '无法读取视频。请检查文件是否存在及编码是否受支持，可尝试 MP4/H.264。')
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and 1 <= fps <= 240 else 25.
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else 0
        index = 0
        events.put(('status', '识别中 · 本地 CPU 推理'))
        while not stop.is_set():
            if paused.is_set():
                stop.wait(.04)
                continue
            start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if is_camera or index == 0:
                    raise RuntimeError('无法获取图像帧。请检查摄像头连接或视频编码。')
                events.put(('status', '视频读取结束'))
                break
            index += 1
            if is_camera and mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if max(h, w) > 960:
                ratio = 960 / max(h, w)
                frame = cv2.resize(frame, (round(w * ratio), round(h * ratio)))
            timestamp = time.monotonic() if is_camera else (index - 1) / fps
            faces = engine.process(frame, timestamp)
            elapsed = time.perf_counter() - start
            item = dict(frame=frame, faces=faces, fps=1 / max(elapsed, .001), index=index,
                        total=frame_total, seconds=(index - 1) / fps, is_camera=is_camera)
            try:
                frames.put_nowait(item)
            except queue.Full:
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
                frames.put_nowait(item)
            # Preserve all video frames; playback slows down if inference is slower than source FPS.
            if not is_camera:
                stop.wait(max(0, 1 / fps - elapsed))
    except Exception as exc:
        logging.exception('Recognition worker failed')
        events.put(('error', str(exc)))
    finally:
        if cap is not None:
            cap.release()
        events.put(('done', ''))


class App:
    def __init__(self, root, models):
        self.root, self.models = root, models
        root.title('实时表情识别 · 摄像头 / 视频')
        root.geometry('1200x780')
        root.minsize(960, 640)
        root.configure(bg='#eef2f7')
        self.worker = None
        self.stop, self.paused = threading.Event(), threading.Event()
        self.frames, self.events = queue.Queue(maxsize=1), queue.Queue()
        self.source = tk.StringVar(value='camera')
        self.video = tk.StringVar()
        self.camera = tk.StringVar(value='自动')
        self.mirror = tk.BooleanVar(value=True)
        self.threshold = tk.DoubleVar(value=.45)
        self.status = tk.StringVar(value='请选择输入来源，然后点击“开始识别”')
        self.stats = tk.StringVar(value='EmotiEffNet-B0 · 8 类表情 · 本地运行')
        self.progress = tk.DoubleVar(value=0)
        self.font = None
        font_path = Path('C:/Windows/Fonts/msyh.ttc')
        if font_path.exists():
            self.font = ImageFont.truetype(str(font_path), 19)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TFrame', background='#eef2f7')
        style.configure('TLabel', background='#eef2f7', font=('Microsoft YaHei UI', 10))
        style.configure('TButton', font=('Microsoft YaHei UI', 10), padding=8)
        style.configure('TRadiobutton', background='#eef2f7', font=('Microsoft YaHei UI', 10))
        style.configure('TCheckbutton', background='#eef2f7', font=('Microsoft YaHei UI', 10))
        shell = ttk.Frame(root, padding=18)
        shell.pack(fill='both', expand=True)
        ttk.Label(shell, text='实时表情识别', font=('Microsoft YaHei UI', 22, 'bold')).pack(anchor='w')
        ttk.Label(shell, text='选择摄像头或本地视频，查看人脸表情与模型分数。', foreground='#64748b').pack(anchor='w', pady=(2, 16))
        body = ttk.Frame(shell)
        body.pack(fill='both', expand=True)
        controls = ttk.Frame(body, width=285)
        controls.pack(side='left', fill='y', padx=(0, 18))
        controls.pack_propagate(False)
        ttk.Label(controls, text='输入来源', font=('Microsoft YaHei UI', 12, 'bold')).pack(anchor='w', pady=(0, 8))
        self.camera_radio = ttk.Radiobutton(controls, text='电脑摄像头', value='camera', variable=self.source, command=self.source_changed)
        self.camera_radio.pack(anchor='w')
        camera_row = ttk.Frame(controls)
        camera_row.pack(fill='x', pady=8)
        ttk.Label(camera_row, text='摄像头编号').pack(side='left')
        self.camera_box = ttk.Combobox(camera_row, textvariable=self.camera, values=['自动', '0', '1', '2', '3'], width=5, state='readonly')
        self.camera_box.pack(side='right')
        self.mirror_box = ttk.Checkbutton(controls, text='镜像预览（仅摄像头）', variable=self.mirror)
        self.mirror_box.pack(anchor='w', pady=(0, 16))
        self.video_radio = ttk.Radiobutton(controls, text='本地视频文件', value='video', variable=self.source, command=self.source_changed)
        self.video_radio.pack(anchor='w')
        self.video_entry = ttk.Entry(controls, textvariable=self.video)
        self.video_entry.pack(fill='x', pady=8)
        self.browse_button = ttk.Button(controls, text='选择视频…', command=self.browse)
        self.browse_button.pack(fill='x')
        ttk.Separator(controls).pack(fill='x', pady=18)
        self.start_button = ttk.Button(controls, text='开始识别', command=self.start)
        self.start_button.pack(fill='x')
        row = ttk.Frame(controls)
        row.pack(fill='x', pady=8)
        self.pause_button = ttk.Button(row, text='暂停', command=self.pause, state='disabled')
        self.pause_button.pack(side='left', fill='x', expand=True, padx=(0, 6))
        self.stop_button = ttk.Button(row, text='停止', command=self.stop_run, state='disabled')
        self.stop_button.pack(side='left', fill='x', expand=True)
        ttk.Label(controls, text='不确定阈值', font=('Microsoft YaHei UI', 10, 'bold')).pack(anchor='w', pady=(16, 0))
        ttk.Scale(controls, from_=.2, to=.8, variable=self.threshold).pack(fill='x')
        ttk.Label(controls, text='最高分低于阈值时显示“不确定”。\n分数不是预测正确率，也不代表内心情绪。', wraplength=270, foreground='#64748b').pack(anchor='w', pady=8)
        preview = ttk.Frame(body)
        preview.pack(side='left', fill='both', expand=True)
        scores = ttk.Frame(preview)
        scores.pack(side='bottom', fill='x', pady=(8, 0))
        ttk.Label(scores, text='最大人脸 · 模型分数').grid(row=0, column=0, columnspan=2, sticky='w')
        scores.columnconfigure((0, 1), weight=1)
        self.bars = []
        for i, name in enumerate(LABELS_ZH):
            bar_row = ttk.Frame(scores)
            bar_row.grid(row=1+i//2, column=i%2, sticky='ew', padx=(0, 12), pady=2)
            ttk.Label(bar_row, text=name, width=5).pack(side='left')
            value = tk.DoubleVar(value=0)
            ttk.Progressbar(bar_row, variable=value, maximum=1, length=150).pack(side='left', fill='x', expand=True)
            text = ttk.Label(bar_row, text='—', width=5, anchor='e')
            text.pack(side='right')
            self.bars.append((value, text))
        self.canvas = tk.Canvas(preview, bg='#101c2e', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.placeholder = self.canvas.create_text(350, 250, text='等待输入画面\n摄像头 / 本地视频', fill='#b6c5da', font=('Microsoft YaHei UI', 18), justify='center')
        self.image_id = self.canvas.create_image(0, 0, anchor='center')
        self.canvas.bind('<Configure>', self.resize)
        ttk.Progressbar(preview, variable=self.progress, maximum=100).pack(fill='x', pady=(10, 6))
        ttk.Label(preview, textvariable=self.stats, foreground='#475569').pack(anchor='w')
        ttk.Label(shell, textvariable=self.status, foreground='#0f766e', wraplength=1100).pack(anchor='w', pady=(12, 0))
        self.source_changed()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(30, self.poll)

    def resize(self, event):
        self.canvas.coords(self.placeholder, event.width / 2, event.height / 2)

    def source_changed(self):
        camera = self.source.get() == 'camera'
        self.camera_box.configure(state='readonly' if camera else 'disabled')
        self.mirror_box.configure(state='normal' if camera else 'disabled')
        self.video_entry.configure(state='disabled' if camera else 'normal')
        self.browse_button.configure(state='disabled' if camera else 'normal')

    def browse(self):
        path = filedialog.askopenfilename(title='选择视频文件', filetypes=[('视频文件', '*.mp4 *.avi *.mov *.mkv *.wmv *.webm'), ('所有文件', '*.*')])
        if path:
            self.video.set(path)

    def set_running(self, running):
        self.start_button.configure(state='disabled' if running else 'normal')
        for control in [self.camera_radio, self.video_radio, self.camera_box, self.video_entry, self.browse_button, self.mirror_box]:
            control.configure(state='disabled' if running else 'normal')
        for control in [self.pause_button, self.stop_button]:
            control.configure(state='normal' if running else 'disabled')
        if not running:
            self.source_changed()
            self.pause_button.configure(text='暂停')

    def start(self):
        if self.worker is not None and self.worker.is_alive():
            return
        camera = self.source.get() == 'camera'
        source = self.camera.get() if camera else self.video.get().strip()
        if not camera and not Path(source).is_file():
            messagebox.showwarning('请选择视频', '请先选择一个存在的视频文件。')
            return
        if not (self.models / 'models.json').exists():
            messagebox.showerror('模型未安装', f'请先安装模型到：\n{self.models}')
            return
        self.stop.clear()
        self.paused.clear()
        self.frames, self.events = queue.Queue(maxsize=1), queue.Queue()
        self.progress.set(0)
        self.set_running(True)
        self.worker = threading.Thread(target=capture_worker, args=(source, camera, self.mirror.get(), self.models,
                                       self.stop, self.paused, self.frames, self.events), daemon=True)
        self.worker.start()

    def pause(self):
        if self.paused.is_set():
            self.paused.clear()
            self.pause_button.configure(text='暂停')
            self.status.set('识别中 · 本地 CPU 推理')
        else:
            self.paused.set()
            self.pause_button.configure(text='继续')
            self.status.set('已暂停')

    def stop_run(self):
        self.stop.set()
        self.status.set('正在停止并释放输入设备…')
        self.pause_button.configure(state='disabled')
        self.stop_button.configure(state='disabled')

    def draw(self, item):
        image = Image.fromarray(cv2.cvtColor(item['frame'], cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        threshold = self.threshold.get()
        for face in item['faces']:
            x1, y1, x2, y2 = face['box']
            prob = face['probabilities']
            index = int(prob.argmax())
            low = prob[index] < threshold
            color = '#fbbf24' if low else '#34d399'
            label = ('不确定' if low else LABELS_ZH[index]) if self.font else ('Uncertain' if low else LABELS_EN[index])
            label += f' {prob[index]:.0%}'
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            top = max(0, y1 - 29)
            rect = draw.textbbox((x1 + 5, top + 3), label, font=self.font)
            draw.rectangle((x1, top, rect[2] + 6, rect[3] + 5), fill='#101c2e')
            draw.text((x1 + 5, top + 3), label, fill=color, font=self.font)
        width, height = max(100, self.canvas.winfo_width()), max(100, self.canvas.winfo_height())
        image.thumbnail((width, height), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfigure(self.placeholder, state='hidden')
        self.canvas.coords(self.image_id, width / 2, height / 2)
        self.canvas.itemconfigure(self.image_id, image=self.photo)
        for index, (value, text) in enumerate(self.bars):
            if item['faces']:
                prob = item['faces'][0]['probabilities'][index]
                value.set(float(prob))
                text.configure(text=f'{prob:.0%}')
            else:
                value.set(0)
                text.configure(text='—')
        prefix = '摄像头' if item['is_camera'] else f"视频 {item['seconds']:.1f} 秒"
        self.stats.set(f"{prefix}  ·  处理速度 {item['fps']:.1f} FPS  ·  人脸 {len(item['faces'])}  ·  未检测到时不作表情判断" if not item['faces'] else
                       f"{prefix}  ·  处理速度 {item['fps']:.1f} FPS  ·  人脸 {len(item['faces'])}  ·  最多处理 4 张脸")
        if item['total'] > 0:
            self.progress.set(min(100, 100 * item['index'] / item['total']))

    def poll(self):
        try:
            while True:
                kind, message = self.events.get_nowait()
                if kind == 'status':
                    self.status.set(message)
                elif kind == 'error':
                    self.status.set(message)
                    messagebox.showerror('识别未完成', message)
                elif kind == 'done':
                    self.set_running(False)
                    if self.stop.is_set():
                        self.status.set('已停止，输入设备已释放。可以切换来源。')
        except queue.Empty:
            pass
        try:
            self.draw(self.frames.get_nowait())
        except queue.Empty:
            pass
        self.root.after(30, self.poll)

    def close(self):
        self.stop.set()
        self.root.withdraw()
        self.wait_close(time.monotonic())

    def wait_close(self, start):
        if self.worker and self.worker.is_alive() and time.monotonic() - start < 4:
            self.root.after(50, lambda: self.wait_close(start))
        else:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default=DEFAULT_MODELS)
    args = parser.parse_args()
    logs = ROOT / 'realtime_output'
    logs.mkdir(exist_ok=True)
    logging.basicConfig(filename=str(logs / 'app.log'), encoding='utf-8', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    root = tk.Tk()
    App(root, args.models)
    root.mainloop()


if __name__ == '__main__':
    main()
