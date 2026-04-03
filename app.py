"""
Mosaic Noise Removal Desktop App
=================================
AI-powered image denoiser that removes mosaic/block noise while preserving
the original resolution.  Uses Real-ESRGAN with ``outscale=1`` so that the
output dimensions match the input exactly.

Usage
-----
    python app.py

Requirements
------------
    pip install customtkinter tkinterdnd2 Pillow numpy torch torchvision \
                realesrgan basicsr gfpgan opencv-python
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Real-ESRGAN helper
# ---------------------------------------------------------------------------

_upsampler: Optional[object] = None
_upsampler_lock = threading.Lock()


def _get_upsampler():
    """Lazily initialise the RealESRGANer upsampler (singleton)."""
    global _upsampler
    if _upsampler is not None:
        return _upsampler

    with _upsampler_lock:
        if _upsampler is not None:          # double-check
            return _upsampler

        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        # Prefer CUDA, fall back to CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        half_precision = device.type == "cuda"

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )

        # Resolve the model weight path shipped with the realesrgan package
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "weights",
            "RealESRGAN_x4plus.pth",
        )

        # Fall back: let realesrgan download the weight automatically
        if not os.path.isfile(model_path):
            from basicsr.utils.download_util import load_file_from_url

            model_url = (
                "https://github.com/xinntao/Real-ESRGAN/releases/download/"
                "v0.1.0/RealESRGAN_x4plus.pth"
            )
            os.makedirs(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"),
                exist_ok=True,
            )
            model_path = load_file_from_url(
                url=model_url,
                model_dir=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "weights"
                ),
                progress=True,
                file_name="RealESRGAN_x4plus.pth",
            )

        _upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=half_precision,
            device=device,
        )
        return _upsampler


def denoise_image(img_bgr: np.ndarray, denoise_strength: float) -> np.ndarray:
    """Remove mosaic / block-noise while keeping the original resolution.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR uint8 format (as read by ``cv2.imread``).
    denoise_strength : float
        Value between 0.0 (no change) and 1.0 (full AI restoration).
        Intermediate values blend the original with the AI output.

    Returns
    -------
    np.ndarray
        Denoised image in BGR uint8, same shape as *img_bgr*.
    """
    upsampler = _get_upsampler()

    # outscale=1 keeps the resolution unchanged
    output, _ = upsampler.enhance(img_bgr, outscale=1)

    # Blend original ↔ restored according to denoise_strength
    if denoise_strength < 1.0:
        output = cv2.addWeighted(
            img_bgr, 1.0 - denoise_strength, output, denoise_strength, 0
        )

    return output


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------

class App(ctk.CTk):
    """Main application window."""

    TITLE = "AI Mosaic Noise Remover"
    MIN_WIDTH = 960
    MIN_HEIGHT = 640

    def __init__(self) -> None:
        super().__init__()

        self.title(self.TITLE)
        self.geometry(f"{self.MIN_WIDTH}x{self.MIN_HEIGHT}")
        self.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # State ---------------------------------------------------------------
        self._original_bgr: Optional[np.ndarray] = None
        self._result_bgr: Optional[np.ndarray] = None
        self._show_original = True  # toggle flag
        self._processing = False

        # Layout --------------------------------------------------------------
        self._build_sidebar()
        self._build_main_area()

        # Drag & Drop ---------------------------------------------------------
        self._setup_dnd()

    # ----- sidebar -----------------------------------------------------------

    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # Title
        ctk.CTkLabel(
            sidebar, text=self.TITLE, font=ctk.CTkFont(size=16, weight="bold")
        ).pack(padx=20, pady=(20, 10))

        # Open button
        ctk.CTkButton(sidebar, text="Open Image…", command=self._open_image).pack(
            padx=20, pady=10, fill="x"
        )

        # Denoise strength slider
        ctk.CTkLabel(sidebar, text="Denoise Strength").pack(padx=20, pady=(20, 0))
        self._strength_var = ctk.DoubleVar(value=1.0)
        self._strength_slider = ctk.CTkSlider(
            sidebar,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            variable=self._strength_var,
            command=self._on_strength_changed,
        )
        self._strength_slider.pack(padx=20, pady=5, fill="x")
        self._strength_label = ctk.CTkLabel(sidebar, text="1.00")
        self._strength_label.pack(padx=20)

        # Process button
        self._process_btn = ctk.CTkButton(
            sidebar, text="▶  Process", command=self._start_processing
        )
        self._process_btn.pack(padx=20, pady=20, fill="x")

        # Toggle button
        self._toggle_btn = ctk.CTkButton(
            sidebar,
            text="Toggle Original / Result",
            command=self._toggle_view,
            state="disabled",
        )
        self._toggle_btn.pack(padx=20, pady=5, fill="x")

        # Save button
        self._save_btn = ctk.CTkButton(
            sidebar, text="Save Result…", command=self._save_result, state="disabled"
        )
        self._save_btn.pack(padx=20, pady=10, fill="x")

        # Status label
        self._status_label = ctk.CTkLabel(
            sidebar, text="Ready", text_color="gray"
        )
        self._status_label.pack(padx=20, pady=(30, 10))

        # Progress bar
        self._progress = ctk.CTkProgressBar(sidebar, mode="indeterminate")
        self._progress.pack(padx=20, fill="x")
        self._progress.set(0)

    # ----- main area ---------------------------------------------------------

    def _build_main_area(self) -> None:
        self._canvas_frame = ctk.CTkFrame(self)
        self._canvas_frame.pack(side="right", fill="both", expand=True)

        self._canvas = tk.Canvas(
            self._canvas_frame, bg="#1a1a1a", highlightthickness=0
        )
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        self._photo_image: Optional[ImageTk.PhotoImage] = None

    # ----- drag & drop -------------------------------------------------------

    def _setup_dnd(self) -> None:
        """Try to register drag-and-drop via TkinterDnD2."""
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD  # noqa: F401

            # Patch the root to support DnD
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            # TkinterDnD2 not available – silent fallback
            pass

    def _on_drop(self, event) -> None:
        path = event.data.strip().strip("{}")
        if os.path.isfile(path):
            self._load_image(path)

    # ----- callbacks ---------------------------------------------------------

    def _on_strength_changed(self, value: float) -> None:
        self._strength_label.configure(text=f"{value:.2f}")

    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str) -> None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", f"Cannot open image:\n{path}")
            return
        self._original_bgr = img
        self._result_bgr = None
        self._show_original = True
        self._toggle_btn.configure(state="disabled")
        self._save_btn.configure(state="disabled")
        self._status_label.configure(text=f"Loaded {Path(path).name}")
        self._display_image(img)

    def _start_processing(self) -> None:
        if self._original_bgr is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        if self._processing:
            return

        self._processing = True
        self._process_btn.configure(state="disabled")
        self._status_label.configure(text="Processing…")
        self._progress.start()

        strength = self._strength_var.get()
        thread = threading.Thread(
            target=self._process_worker, args=(strength,), daemon=True
        )
        thread.start()

    def _process_worker(self, strength: float) -> None:
        """Run denoising in a background thread."""
        try:
            result = denoise_image(self._original_bgr, strength)
            self.after(0, self._on_process_done, result, None)
        except Exception as exc:
            self.after(0, self._on_process_done, None, exc)

    def _on_process_done(
        self, result: Optional[np.ndarray], error: Optional[Exception]
    ) -> None:
        self._progress.stop()
        self._progress.set(0)
        self._processing = False
        self._process_btn.configure(state="normal")

        if error is not None:
            self._status_label.configure(text="Error")
            messagebox.showerror("Processing Error", str(error))
            return

        self._result_bgr = result
        self._show_original = False
        self._toggle_btn.configure(state="normal")
        self._save_btn.configure(state="normal")
        self._status_label.configure(text="Done – showing result")
        self._display_image(result)

    def _toggle_view(self) -> None:
        if self._result_bgr is None:
            return
        self._show_original = not self._show_original
        if self._show_original:
            self._status_label.configure(text="Showing: Original")
            self._display_image(self._original_bgr)
        else:
            self._status_label.configure(text="Showing: Result")
            self._display_image(self._result_bgr)

    def _save_result(self) -> None:
        if self._result_bgr is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("BMP", "*.bmp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            cv2.imwrite(path, self._result_bgr)
            self._status_label.configure(text=f"Saved → {Path(path).name}")

    # ----- display helpers ---------------------------------------------------

    def _display_image(self, img_bgr: np.ndarray) -> None:
        """Resize *img_bgr* to fit the canvas and display it."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        pil_img.thumbnail((cw, ch), Image.LANCZOS)
        self._photo_image = ImageTk.PhotoImage(pil_img)
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, image=self._photo_image)

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self._show_original and self._original_bgr is not None:
            self._display_image(self._original_bgr)
        elif not self._show_original and self._result_bgr is not None:
            self._display_image(self._result_bgr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = App()
    app.mainloop()
