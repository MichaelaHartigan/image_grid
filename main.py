import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from sklearn.cluster import KMeans


class ImageGridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Grid Color Mapper")
        self.root.geometry("1200x800")
        self.root.minsize(900, 650)

        self.image = None
        self.result = None
        self.preview_photo = None

        self.rows_var = tk.IntVar(value=20)
        self.cols_var = tk.IntVar(value=20)
        self.colors_var = tk.IntVar(value=8)
        self.line_width_var = tk.IntVar(value=2)
        self.labels_var = tk.BooleanVar(value=True)
        self.fit_mode_var = tk.StringVar(value="Crop to grid")

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        controls = ttk.Frame(self.root, padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        preview_frame = ttk.Frame(self.root, padding=(0, 10, 10, 10))
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(
            controls, text="Image Grid Color Mapper",
            font=("TkDefaultFont", 16, "bold")
        ).pack(anchor="w", pady=(0, 15))

        ttk.Button(
            controls, text="Open Image...", command=self.open_image
        ).pack(fill=tk.X, pady=4)

        self.file_label = ttk.Label(controls, text="No image selected", wraplength=230)
        self.file_label.pack(anchor="w", pady=(2, 15))

        self._add_spinbox(controls, "Rows", self.rows_var, 1, 300)
        self._add_spinbox(controls, "Columns", self.cols_var, 1, 300)
        self._add_spinbox(controls, "Number of colors", self.colors_var, 1, 64)
        self._add_spinbox(controls, "Grid line width", self.line_width_var, 0, 20)

        ttk.Label(controls, text="Image fitting").pack(anchor="w", pady=(15, 3))
        fit = ttk.Combobox(
            controls,
            textvariable=self.fit_mode_var,
            values=["Crop to grid", "Stretch to grid"],
            state="readonly",
        )
        fit.pack(fill=tk.X)

        ttk.Checkbutton(
            controls, text="Show row/column labels",
            variable=self.labels_var
        ).pack(anchor="w", pady=12)

        ttk.Button(
            controls, text="Generate Preview", command=self.generate
        ).pack(fill=tk.X, pady=4)

        ttk.Button(
            controls, text="Save Result...", command=self.save_result
        ).pack(fill=tk.X, pady=4)

        ttk.Separator(controls).pack(fill=tk.X, pady=18)

        info = (
            "How it works:\n\n"
            "1. The image is divided into the requested grid.\n"
            "2. Each cell gets its average RGB color.\n"
            "3. K-means reduces those cell colors to the requested number.\n"
            "4. The simplified colors are rendered as a grid."
        )
        ttk.Label(controls, text=info, wraplength=230, justify="left").pack(anchor="w")

        self.status_var = tk.StringVar(value="Open an image to begin.")
        ttk.Label(
            controls, textvariable=self.status_var,
            wraplength=230, justify="left"
        ).pack(anchor="w", side=tk.BOTTOM, pady=(20, 0))

        self.canvas = tk.Canvas(preview_frame, background="#dddddd", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._redraw_preview)

    def _add_spinbox(self, parent, label, variable, minimum, maximum):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 3))
        box = ttk.Spinbox(
            parent,
            from_=minimum,
            to=maximum,
            textvariable=variable,
            width=12
        )
        box.pack(fill=tk.X)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            self.image = Image.open(path).convert("RGB")
            self.file_label.config(
                text=f"{Path(path).name}\n{self.image.width} × {self.image.height}px"
            )
            self.status_var.set("Image loaded. Set the grid/color values and generate.")
            self.generate()
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))

    def _validate_settings(self):
        try:
            rows = int(self.rows_var.get())
            cols = int(self.cols_var.get())
            colors = int(self.colors_var.get())
            line_width = int(self.line_width_var.get())
        except (ValueError, tk.TclError):
            raise ValueError("Rows, columns, colors, and line width must be integers.")

        if rows < 1 or cols < 1:
            raise ValueError("Rows and columns must be at least 1.")
        if colors < 1:
            raise ValueError("Number of colors must be at least 1.")
        if colors > rows * cols:
            raise ValueError(
                "Number of colors cannot exceed the number of grid cells."
            )
        if line_width < 0:
            raise ValueError("Grid line width cannot be negative.")

        return rows, cols, colors, line_width

    def _prepare_image(self, image, rows, cols):
        target_ratio = cols / rows
        image_ratio = image.width / image.height

        if self.fit_mode_var.get() == "Stretch to grid":
            return image.resize((cols, rows), Image.Resampling.LANCZOS)

        # Crop to the requested aspect ratio, preserving the subject scale.
        if image_ratio > target_ratio:
            new_width = int(image.height * target_ratio)
            left = (image.width - new_width) // 2
            image = image.crop((left, 0, left + new_width, image.height))
        else:
            new_height = int(image.width / target_ratio)
            top = (image.height - new_height) // 2
            image = image.crop((0, top, image.width, top + new_height))

        return image.resize((cols, rows), Image.Resampling.BOX)

    def _cell_colors(self, image, rows, cols):
        # Resize using BOX filtering: each output pixel approximates
        # the average color of the corresponding source region.
        tiny = self._prepare_image(image, rows, cols)
        return np.asarray(tiny, dtype=np.float32).reshape(-1, 3)

    def _quantize(self, cell_colors, number_of_colors):
        unique = np.unique(cell_colors.astype(np.uint8), axis=0)
        actual_colors = min(number_of_colors, len(unique))

        if actual_colors == 1:
            center = np.mean(cell_colors, axis=0).round().astype(np.uint8)
            return np.repeat(center[None, :], len(cell_colors), axis=0), center[None, :]

        # A deterministic seed makes repeated generations stable.
        model = KMeans(
            n_clusters=actual_colors,
            random_state=42,
            n_init=10,
        )
        labels = model.fit_predict(cell_colors)
        centers = np.clip(np.round(model.cluster_centers_), 0, 255).astype(np.uint8)
        mapped = centers[labels]
        return mapped, centers

    def _make_result(self, rows, cols, colors, line_width):
        cell_colors = self._cell_colors(self.image, rows, cols)
        mapped, palette = self._quantize(cell_colors, colors)

        # Build a clean, high-resolution image: one cell is 40 px minimum,
        # then add room for labels around the grid.
        cell_size = max(20, min(60, 900 // max(rows, cols)))
        grid_w = cols * cell_size
        grid_h = rows * cell_size

        label_size = 45 if self.labels_var.get() else 0
        left = label_size
        top = label_size
        right = 20
        bottom = label_size

        output = Image.new(
            "RGB",
            (left + grid_w + right, top + grid_h + bottom),
            "white",
        )
        draw = ImageDraw.Draw(output)

        # Grid cells
        for r in range(rows):
            for c in range(cols):
                rgb = tuple(int(x) for x in mapped[r * cols + c])
                x0 = left + c * cell_size
                y0 = top + r * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                draw.rectangle((x0, y0, x1, y1), fill=rgb)

        # Grid lines
        if line_width > 0:
            line_color = (35, 35, 35)
            for c in range(cols + 1):
                x = left + c * cell_size
                draw.line((x, top, x, top + grid_h), fill=line_color, width=line_width)
            for r in range(rows + 1):
                y = top + r * cell_size
                draw.line((left, y, left + grid_w, y), fill=line_color, width=line_width)

        # Axis labels: 1-based row/column numbering.
        if self.labels_var.get():
            font = self._get_font(13)
            small_font = self._get_font(11)

            # Column numbers
            for c in range(cols):
                text = str(c + 1)
                bbox = draw.textbbox((0, 0), text, font=small_font)
                tw = bbox[2] - bbox[0]
                x = left + c * cell_size + (cell_size - tw) / 2
                draw.text((x, 15), text, fill="black", font=small_font)

            # Row numbers
            for r in range(rows):
                text = str(r + 1)
                bbox = draw.textbbox((0, 0), text, font=small_font)
                th = bbox[3] - bbox[1]
                y = top + r * cell_size + (cell_size - th) / 2
                draw.text((12, y), text, fill="black", font=small_font)

            # Axis titles
            draw.text((left, 2), "Columns", fill="black", font=font)
            # Rotate "Rows" so it sits naturally on the left.
            row_label = Image.new("RGBA", (50, 100), (255, 255, 255, 0))
            rd = ImageDraw.Draw(row_label)
            rd.text((2, 0), "Rows", fill="black", font=font)
            row_label = row_label.rotate(90, expand=True)
            output.paste(row_label, (0, top + grid_h // 2 - row_label.height // 2), row_label)

        # Add a small palette legend below the grid.
        if len(palette) > 0:
            legend_y = top + grid_h + 8
            swatch = 16
            x = left
            for i, color in enumerate(palette):
                draw.rectangle(
                    (x, legend_y, x + swatch, legend_y + swatch),
                    fill=tuple(int(v) for v in color),
                    outline=(30, 30, 30),
                )
                draw.text(
                    (x + swatch + 4, legend_y + 1),
                    str(i + 1),
                    fill="black",
                    font=self._get_font(10),
                )
                x += 35

        return output, len(palette)

    def _get_font(self, size):
        # Use a common font if available; Pillow's default is the fallback.
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
        for path in candidates:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    def generate(self):
        if self.image is None:
            messagebox.showinfo("No image", "Open an image first.")
            return

        try:
            rows, cols, colors, line_width = self._validate_settings()
            self.status_var.set("Processing...")
            self.root.update_idletasks()

            self.result, actual_colors = self._make_result(
                rows, cols, colors, line_width
            )
            self._redraw_preview()

            self.status_var.set(
                f"Generated {rows} × {cols} grid using {actual_colors} colors."
            )
        except Exception as exc:
            messagebox.showerror("Could not generate result", str(exc))
            self.status_var.set("Generation failed.")

    def _redraw_preview(self, event=None):
        if self.result is None:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="Open an image and generate a preview",
                fill="#666666",
                font=("TkDefaultFont", 14),
            )
            return

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())

        margin = 20
        scale = min(
            (canvas_w - margin * 2) / self.result.width,
            (canvas_h - margin * 2) / self.result.height,
        )
        scale = max(0.05, scale)

        new_size = (
            max(1, int(self.result.width * scale)),
            max(1, int(self.result.height * scale)),
        )

        preview = self.result.resize(new_size, Image.Resampling.NEAREST)
        self.preview_photo = ImageTk.PhotoImage(preview)

        self.canvas.delete("all")
        x = (canvas_w - new_size[0]) // 2
        y = (canvas_h - new_size[1]) // 2
        self.canvas.create_image(x, y, image=self.preview_photo, anchor="nw")

    def save_result(self):
        if self.result is None:
            messagebox.showinfo("Nothing to save", "Generate a result first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save grid image",
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("JPEG image", "*.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            # PNG preserves the exact cell colors and grid lines.
            self.result.save(path)
            self.status_var.set(f"Saved: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Could not save image", str(exc))


def main():
    root = tk.Tk()
    ImageGridApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
