from __future__ import annotations

import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch

from .config import OrbitConfig
from .jobs import create_job_bundle
from .model import OrbitForCausalLM
from .train import run_training
from .training_config import TrainingConfig


class OrbitApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Orbit AI")
        self.root.geometry("900x680")
        self.model = None
        self.stop_event = threading.Event()
        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.configure("Title.TLabel", font=("SF Pro Display", 24, "bold"))
        style.configure("Hint.TLabel", foreground="#666666")
        header = ttk.Frame(self.root, padding=20)
        header.pack(fill="x")
        ttk.Label(header, text="Orbit", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="人人都能训练和使用的本地 AI", style="Hint.TLabel").pack(anchor="w", pady=(4, 0))
        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.chat_tab, self.train_tab = ttk.Frame(tabs, padding=16), ttk.Frame(tabs, padding=16)
        tabs.add(self.chat_tab, text="对话")
        tabs.add(self.train_tab, text="图形化训练")
        self._build_chat()
        self._build_training()

    def _build_chat(self):
        self.chat_output = tk.Text(self.chat_tab, height=25, wrap="word", state="disabled")
        self.chat_output.pack(fill="both", expand=True)
        row = ttk.Frame(self.chat_tab)
        row.pack(fill="x", pady=(12, 0))
        self.prompt = ttk.Entry(row)
        self.prompt.pack(side="left", fill="x", expand=True)
        self.prompt.bind("<Return>", lambda _: self._send())
        ttk.Button(row, text="发送", command=self._send).pack(side="left", padx=(8, 0))
        ttk.Button(row, text="加载模型", command=self._load_model).pack(side="left", padx=(8, 0))

    def _build_training(self):
        form = ttk.Frame(self.train_tab)
        form.pack(fill="x")
        self.preset = tk.StringVar(value="300m")
        defaults = TrainingConfig.for_model("300m")
        self.steps = tk.StringVar(value="100")
        self.batch = tk.StringVar(value=str(defaults.batch_size))
        self.seq_len = tk.StringVar(value="256")
        self.grad_accum = tk.StringVar(value=str(defaults.grad_accum))
        self.lr = tk.StringVar(value=str(defaults.learning_rate))
        self.warmup = tk.StringVar(value=str(defaults.warmup_steps))
        self.weight_decay = tk.StringVar(value=str(defaults.weight_decay))
        self.grad_clip = tk.StringVar(value=str(defaults.grad_clip))
        self.precision = tk.StringVar(value=defaults.precision)
        self.scheduler = tk.StringVar(value=defaults.scheduler)
        self.checkpoint_every = tk.StringVar(value=str(defaults.checkpoint_every))
        self.device = tk.StringVar(value="auto")
        fields = [
            ("模型规模", self.preset), ("训练步数", self.steps), ("批大小", self.batch),
            ("序列长度", self.seq_len), ("梯度累积", self.grad_accum), ("学习率", self.lr),
            ("Warmup 步数", self.warmup), ("权重衰减", self.weight_decay),
            ("梯度裁剪", self.grad_clip), ("精度", self.precision),
            ("学习率计划", self.scheduler), ("保存间隔", self.checkpoint_every),
            ("设备", self.device),
        ]
        for row, (label, var) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=4)
            if label == "模型规模":
                widget = ttk.Combobox(form, textvariable=var, values=["300m", "1b", "3b", "7b", "14b", "38b"], state="readonly", width=24)
                widget.bind("<<ComboboxSelected>>", lambda _: self._update_estimate())
            elif label in {"精度", "学习率计划", "设备"}:
                values = {"精度": ["auto", "fp32", "fp16", "bf16"], "学习率计划": ["cosine", "constant"], "设备": ["auto", "mps", "cuda", "cpu"]}[label]
                widget = ttk.Combobox(form, textvariable=var, values=values, state="readonly", width=24)
            else:
                widget = ttk.Entry(form, textvariable=var, width=26)
            widget.grid(row=row, column=1, sticky="w", padx=12, pady=4)
        self.estimate = ttk.Label(form, style="Hint.TLabel")
        self.estimate.grid(row=0, column=2, rowspan=6, sticky="nw", padx=20)
        self._update_estimate()
        ttk.Label(self.train_tab, text="训练文本（用户自己的数据）").pack(anchor="w", pady=(16, 4))
        self.corpus = tk.Text(self.train_tab, height=7, wrap="word")
        self.corpus.pack(fill="x")
        ttk.Button(self.train_tab, text="从文件加载训练文本", command=self._load_corpus).pack(anchor="w", pady=(6, 0))
        bottom = ttk.Frame(self.train_tab)
        bottom.pack(fill="x", pady=(10, 0))
        self.start_button = ttk.Button(bottom, text="在本机开始训练", command=self._start_training)
        self.start_button.pack(side="left")
        ttk.Button(bottom, text="导出远程任务包", command=self._export_training).pack(side="left", padx=8)
        ttk.Button(bottom, text="停止", command=self.stop_event.set).pack(side="left")
        self.progress = ttk.Progressbar(bottom, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=8)
        self.log = tk.Text(self.train_tab, height=7, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, pady=(10, 0))

    def _update_estimate(self):
        cfg = OrbitConfig.for_preset(self.preset.get())
        params = cfg.estimate_parameters() / 1e6
        memory = cfg.estimated_training_memory_gb()
        check = cfg.memory_check()
        status = "本机可尝试" if check["can_train"] else "本机内存不足"
        self.estimate.config(text=f"约 {params:.0f}M 参数\n训练内存估算：{memory:.1f}GB\n本机内存：{check['system_gb']:.1f}GB\n状态：{status}")

    def _append(self, widget, text):
        widget.configure(state="normal")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def _load_corpus(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*")])
        if not path:
            return
        try:
            self.corpus.delete("1.0", "end")
            self.corpus.insert("1.0", Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc))

    def _start_training(self):
        try:
            train_cfg = TrainingConfig(
                steps=int(self.steps.get()), batch_size=int(self.batch.get()), seq_len=int(self.seq_len.get()),
                grad_accum=int(self.grad_accum.get()), learning_rate=float(self.lr.get()),
                warmup_steps=int(self.warmup.get()), weight_decay=float(self.weight_decay.get()),
                grad_clip=float(self.grad_clip.get()), precision=self.precision.get(),
                scheduler=self.scheduler.get(), checkpoint_every=int(self.checkpoint_every.get()),
            )
            train_cfg.validate()
        except ValueError:
            messagebox.showerror("参数错误", "请检查训练参数格式。")
            return
        cfg = OrbitConfig.for_preset(self.preset.get())
        memory = cfg.memory_check()
        if self.device.get() in {"auto", "mps", "cpu"} and not memory["can_train"]:
            messagebox.showerror("内存不足", f"{self.preset.get()} 预计需要约 {memory['required_gb']:.1f}GB，当前机器约 {memory['system_gb']:.1f}GB。请降低模型规模或改用 CUDA GPU。")
            return
        text = self.corpus.get("1.0", "end").strip()
        path = filedialog.asksaveasfilename(title="保存 Orbit checkpoint", defaultextension=".pt", initialfile=f"orbit-{self.preset.get()}.pt")
        if not path:
            return
        self.stop_event.clear()
        self.start_button.configure(state="disabled")
        self.progress.configure(value=0, maximum=train_cfg.steps)
        def callback(step, loss):
            self.root.after(0, lambda: (self.progress.configure(value=step), self._append(self.log, f"step {step}/{train_cfg.steps}  loss={loss:.4f}\n")))
        def worker():
            try:
                result = run_training(
                    device_name=self.device.get(), checkpoint=Path(path), text=text,
                    preset=self.preset.get(), callback=callback, stop_event=self.stop_event,
                    training_config=train_cfg,
                )
                self.root.after(0, lambda: self._append(self.log, f"训练完成，已保存：{result}\n"))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("训练失败", str(exc)))
            finally:
                self.root.after(0, lambda: self.start_button.configure(state="normal"))
        threading.Thread(target=worker, daemon=True).start()

    def _export_training(self):
        try:
            train_cfg = TrainingConfig(
                steps=int(self.steps.get()), batch_size=int(self.batch.get()), seq_len=int(self.seq_len.get()),
                grad_accum=int(self.grad_accum.get()), learning_rate=float(self.lr.get()),
                warmup_steps=int(self.warmup.get()), weight_decay=float(self.weight_decay.get()),
                grad_clip=float(self.grad_clip.get()), precision=self.precision.get(),
                scheduler=self.scheduler.get(), checkpoint_every=int(self.checkpoint_every.get()),
            )
            train_cfg.validate()
        except ValueError:
            messagebox.showerror("参数错误", "请检查训练参数格式。")
            return
        output = filedialog.askdirectory(title="选择任务包保存目录")
        if not output:
            return
        text = self.corpus.get("1.0", "end").strip() or "Orbit training sample. " * 100
        try:
            path = create_job_bundle(Path(output), self.preset.get(), train_cfg.steps, train_cfg.batch_size, train_cfg.seq_len, train_cfg.learning_rate, text, training_config=train_cfg)
            self._append(self.log, f"已生成远程任务包：{path}\n")
            messagebox.showinfo("任务包已生成", f"已保存到：\n{path}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def _load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Orbit checkpoint", "*.pt"), ("All files", "*")])
        if not path:
            return
        try:
            checkpoint = torch.load(path, map_location="cpu")
            cfg = OrbitConfig(**checkpoint["config"])
            self.model = OrbitForCausalLM(cfg)
            self.model.load_state_dict(checkpoint["model"])
            self.model.eval()
            self._append(self.chat_output, f"已加载模型：{path}\n\n")
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc))

    def _send(self):
        prompt = self.prompt.get().strip()
        if not prompt:
            return
        self.prompt.delete(0, "end")
        self._append(self.chat_output, f"你：{prompt}\n")
        if self.model is None:
            self._append(self.chat_output, "Orbit：请先加载训练好的 checkpoint。\n\n")
            return
        ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long)
        result = self.model.generate(ids, max_new_tokens=64, temperature=0.8)
        answer = bytes(result[0].tolist()).decode("utf-8", errors="replace")
        self._append(self.chat_output, f"Orbit：{answer[len(prompt):]}\n\n")


def main():
    root = tk.Tk()
    OrbitApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
