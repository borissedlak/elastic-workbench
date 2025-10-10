import os
import math
import glob
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw


class LeverGUI(tk.Tk):
    """
    GUI with a big slider on top and a 3x3 image grid below.

    Improvements:
    - images are resized to *fill* each cell (cover behavior) and cropped if necessary
    - canvases redraw on resize so images always fill the available cell space
    - simple caching of opened PIL images to avoid repeated disk reads
    """

    def __init__(self, images_folder: str = "images"):
        super().__init__()
        self.title("Levers & Image Grid — example GUI")
        self.geometry("900x700")
        self.minsize(640, 480)
        self._is_fullscreen = False

        # image data
        self.images_folder = images_folder
        self.image_paths = []  # list of file paths
        self.photo_refs = {}   # keep PhotoImage references alive per cell {(r,c): PhotoImage}
        self.pil_cache = {}    # cache opened PIL.Image objects by path (or index for placeholders)
        self.page_count = 1
        self.current_page = 0

        self._create_widgets()
        self._layout_widgets()

        # load images (synchronously) and show first page
        self._load_images()
        self._show_page(0)

    def _create_widgets(self):
        # Top bar (maximize/reset)
        self.top_bar = ttk.Frame(self, padding=(6, 4))
        self.max_btn = ttk.Button(self.top_bar, text="🔳 Maximize", width=12, command=self._toggle_maximize)
        self.reset_btn = ttk.Button(self.top_bar, text="Reset", command=self._reset)

        # Big slider on top (we will use Int scale to select page index)
        self.slider_var = tk.IntVar(value=0)
        self.page_slider = ttk.Scale(self, from_=0, to=0, orient="horizontal",
                                     command=lambda e: self._on_slider_change(), variable=self.slider_var)
        self.page_label = ttk.Label(self, text="Page 1 / 1")

        # Grid container frame (3x3)
        self.grid_frame = ttk.Frame(self, padding=8)
        self.cell_frames = []
        self.cell_canvases = []
        for r in range(3):
            row_frames = []
            row_canvases = []
            for c in range(3):
                f = ttk.Frame(self.grid_frame, relief="solid")
                # allow frame to expand; canvas will control drawing
                cv = tk.Canvas(f, bd=0, highlightthickness=0)
                cv.pack(expand=True, fill="both")
                # bind resize to redraw the image for this cell
                cv.bind("<Configure>", lambda e, rr=r, cc=c: self._on_canvas_configure(rr, cc))
                row_frames.append(f)
                row_canvases.append(cv)
            self.cell_frames.append(row_frames)
            self.cell_canvases.append(row_canvases)

    def _layout_widgets(self):
        # top bar
        self.top_bar.pack(fill="x")
        self.max_btn.pack(side="left", padx=(2, 6))
        self.reset_btn.pack(side="left")

        # slider and page label
        self.page_slider.pack(fill="x", padx=12, pady=(10, 2))
        self.page_label.pack(padx=12, pady=(0, 6))

        # grid
        self.grid_frame.pack(expand=True, fill="both", padx=12, pady=8)

        # configure grid_frame rows/cols
        for i in range(3):
            self.grid_frame.columnconfigure(i, weight=1)
            self.grid_frame.rowconfigure(i, weight=1)
            for j in range(3):
                # place frame and let it expand
                self.cell_frames[i][j].grid(row=i, column=j, padx=6, pady=6, sticky="nsew")

    def _load_images(self):
        """Load image file paths from the supplied folder.
        If none found, create placeholder entries (None) so the UI is usable.
        """
        supported = ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")
        imgs = []
        if os.path.isdir(self.images_folder):
            for pat in supported:
                imgs.extend(sorted(glob.glob(os.path.join(self.images_folder, pat))))
        imgs = sorted(list(dict.fromkeys(imgs)))
        if not imgs:
            imgs = [None] * 9
        self.image_paths = imgs
        self.page_count = max(1, math.ceil(len(self.image_paths) / 9))

        # reconfigure slider range
        self.page_slider.configure(from_=0, to=max(0, self.page_count - 1), length=800)
        self.slider_var.set(0)
        self.page_label.config(text=f"Page 1 / {self.page_count}")

        # clear caches
        self.pil_cache.clear()
        self.photo_refs.clear()

    def _open_pil(self, path, placeholder_index=0):
        """Return a PIL.Image for the given path, caching results. If path is None produce a placeholder image."""
        key = path if path is not None else f"_placeholder_{placeholder_index}"
        if key in self.pil_cache:
            return self.pil_cache[key]

        if path is None:
            # create neutral placeholder with index text
            img = Image.new("RGB", (800, 600), (220, 220, 220))
            draw = ImageDraw.Draw(img)
            txt = f"Placeholder {placeholder_index + 1}"
            draw.text((10, 10), txt, fill=(60, 60, 60))
        else:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (800, 600), (200, 200, 200))
        self.pil_cache[key] = img
        return img

    def _resize_cover(self, pil_img, target_size):
        """Resize PIL image to fill target_size (cover), center-cropped when necessary."""
        tw, th = target_size
        iw, ih = pil_img.size
        if iw == 0 or ih == 0 or tw == 0 or th == 0:
            return Image.new("RGB", target_size, (240, 240, 240))

        # compute scale factor to cover target
        scale = max(tw / iw, th / ih)
        nw = int(math.ceil(iw * scale))
        nh = int(math.ceil(ih * scale))
        resized = pil_img.resize((nw, nh), Image.LANCZOS)
        # crop center to target
        left = (nw - tw) // 2
        top = (nh - th) // 2
        right = left + tw
        bottom = top + th
        cropped = resized.crop((left, top, right, bottom))
        return cropped

    def _on_slider_change(self):
        val = int(round(self.slider_var.get()))
        val = max(0, min(self.page_count - 1, val))
        if val != self.current_page:
            self._show_page(val)

    def _show_page(self, page_idx: int):
        """Display images for the given page index (0-based)."""
        self.current_page = page_idx
        start = page_idx * 9
        end = start + 9
        page_paths = self.image_paths[start:end]
        page_paths = list(page_paths) + [None] * (9 - len(page_paths))

        # For each cell, draw (using current canvas size)
        for i in range(9):
            r = i // 3
            c = i % 3
            cv = self.cell_canvases[r][c]
            path = page_paths[i]
            self._draw_cell(r, c, path, placeholder_index=start + i)

        # update slider label and snap
        self.page_label.config(text=f"Page {page_idx + 1} / {self.page_count}")
        self.slider_var.set(page_idx)

    def _draw_cell(self, r, c, path, placeholder_index=0):
        """Draw the image for cell (r,c) filling the canvas area."""
        cv = self.cell_canvases[r][c]
        cv.delete("all")
        w = max(1, cv.winfo_width())
        h = max(1, cv.winfo_height())
        pil = self._open_pil(path, placeholder_index=placeholder_index)
        fitted = self._resize_cover(pil, (w, h))
        photo = ImageTk.PhotoImage(fitted)
        # store reference keyed by cell to prevent GC
        self.photo_refs[(r, c)] = photo
        # draw at top-left corner (fills canvas)
        cv.create_image(0, 0, image=photo, anchor="nw")

    def _on_canvas_configure(self, r, c):
        """Called when a specific canvas changes size; redraw that cell with cover resizing."""
        # if current page is set, determine which image belongs to this cell
        start = self.current_page * 9
        idx = r * 3 + c
        overall_idx = start + idx
        path = None
        if overall_idx < len(self.image_paths):
            path = self.image_paths[overall_idx]
        self._draw_cell(r, c, path, placeholder_index=overall_idx)

    def _reset(self):
        self._load_images()
        self._show_page(0)

    def _toggle_maximize(self):
        if not self._is_fullscreen:
            try:
                self.attributes("-fullscreen", True)
            except Exception:
                try:
                    self.state("zoomed")
                except Exception:
                    pass
            self.max_btn.config(text="🗗 Restore")
            self._is_fullscreen = True
        else:
            try:
                self.attributes("-fullscreen", False)
            except Exception:
                try:
                    self.state("normal")
                except Exception:
                    pass
            self.max_btn.config(text="🔳 Maximize")
            self._is_fullscreen = False


if __name__ == "__main__":
    app = LeverGUI(images_folder="./rask_plots")
    app.mainloop()
