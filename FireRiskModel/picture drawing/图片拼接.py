"""
图片拼接组合
功能简介:
1. 从用户指定文件夹读取所有图片文件。
2. 根据用户输入的行、列数将图片拼接成一张大图。
3. 可选为每张图添加 a, b, c, d... 或 A, B, C, D... 编号。
4. 输出到 B:\\WUI\\Picture，保存 DPI 不低于 3000。
"""

import os
import sys
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
Image.MAX_IMAGE_PIXELS = None  # 本地可信图片允许读取超大尺寸栅格


def _read_folder() -> str:
    folder = input("请输入图片文件夹路径: ").strip().strip('"').strip("'")
    if not folder:
        raise ValueError("图片文件夹路径不能为空。")
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise ValueError(f"文件夹不存在: {folder}")
    return folder


def _read_grid() -> tuple[int, int]:
    raw = input("请输入图片排布的行,列(例如 2,3): ").strip()
    if "," not in raw:
        raise ValueError("行,列格式应为例如 2,3")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("行,列格式应为例如 2,3")
    rows, cols = (int(parts[0]), int(parts[1]))
    if rows <= 0 or cols <= 0:
        raise ValueError("行列数必须为正整数。")
    return rows, cols


def _read_label_choice() -> bool:
    raw = input("是否添加编号(a/A,b/B,c/C,d/D...)? (y/n): ").strip().lower()
    return raw in {"y", "yes", "是", "1", "true", "t"}


def _read_label_case() -> str:
    raw = input("编号使用小写还是大写? (l/u): ").strip().lower()
    if raw in {"u", "upper", "uppercase", "大写", "2"}:
        return "upper"
    return "lower"


def _list_images(folder: str) -> list[str]:
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTS:
            files.append(path)
    if not files:
        raise ValueError("未在文件夹中找到图片文件。")
    return files


def _load_font(size: int) -> ImageFont.ImageFont:
    # Prefer Arial Bold for panel labels; fall back to regular Arial or SimHei.
    for font_name in (
        "arialbd.ttf",
        "Arial Bold.ttf",
        "ARIALBD.TTF",
        "arial.ttf",
        "Arial.ttf",
        "simhei.ttf",
        "SimHei.ttf",
    ):
        try:
            return ImageFont.truetype(font_name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _label_for_index(index: int, label_case: str) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if label_case == "upper" else "abcdefghijklmnopqrstuvwxyz"
    if index < 26:
        return alphabet[index]
    # 26 -> aa, 27 -> ab ...
    first = alphabet[(index // 26) - 1]
    second = alphabet[index % 26]
    return f"{first}{second}"


def _collect_image_sizes(paths: list[str]) -> tuple[list[tuple[str, int, int]], int, int]:
    image_sizes = []
    max_w = 0
    max_h = 0
    for path in paths:
        with Image.open(path) as img:
            image_sizes.append((path, img.width, img.height))
            max_w = max(max_w, img.width)
            max_h = max(max_h, img.height)
    return image_sizes, max_w, max_h


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def _paste_image(canvas: Image.Image, path: str, offset_x: int, offset_y: int) -> None:
    with Image.open(path) as img:
        if img.mode in ("RGBA", "LA"):
            alpha = img.getchannel("A")
            canvas.paste(img, (offset_x, offset_y), mask=alpha)
            return
        if img.mode == "P" and "transparency" in img.info:
            rgba = img.convert("RGBA")
            canvas.paste(rgba, (offset_x, offset_y), mask=rgba.getchannel("A"))
            return
        if img.mode != "RGB":
            img = img.convert("RGB")
        canvas.paste(img, (offset_x, offset_y))


def main() -> int:
    try:
        folder = _read_folder()
        images = _list_images(folder)
        print(f"已找到图片数量: {len(images)}")
        rows, cols = _read_grid()
        add_labels = _read_label_choice()
        label_case = _read_label_case() if add_labels else "lower"
    except Exception as exc:
        print(f"输入错误: {exc}")
        return 1

    total_slots = rows * cols
    if len(images) > total_slots:
        print(f"图片数量({len(images)})超过行列格子数({total_slots})，请调整行列数。")
        return 1

    try:
        image_sizes, max_w, max_h = _collect_image_sizes(images)
    except Exception as exc:
        print(f"读取图片失败: {exc}")
        return 1

    if max_w == 0 or max_h == 0:
        print("图片尺寸无效。")
        return 1

    canvas_w = cols * max_w
    canvas_h = rows * max_h
    canvas_pixels = canvas_w * canvas_h
    estimated_ram = canvas_pixels * 3
    print(
        f"输出画布尺寸: {canvas_w} x {canvas_h} "
        f"({canvas_pixels:,} 像素, 画布内存约 {_format_gib(estimated_ram)})"
    )

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(size=max(12, int(min(max_w, max_h) * 0.06)))

    for i, (path, img_w, img_h) in enumerate(image_sizes):
        row = i // cols
        col = i % cols
        x0 = col * max_w
        y0 = row * max_h
        # Center image inside its cell
        offset_x = x0 + (max_w - img_w) // 2
        offset_y = y0 + (max_h - img_h) // 2
        _paste_image(canvas, path, offset_x, offset_y)

        if add_labels:
            label = _label_for_index(i, label_case)
            margin = max(5, int(min(max_w, max_h) * 0.02))
            draw.text((x0 + margin, y0 + margin), label, fill=(0, 0, 0), font=font)

    # Fill remaining slots with white background (already white)
    out_dir = folder
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"拼接结果_{timestamp}.png")
    canvas.save(out_path, dpi=(4000, 4000))

    print(f"已输出到: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
