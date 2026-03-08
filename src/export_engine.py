import io
import os
import tempfile
from typing import Dict, List, Tuple

from fpdf import FPDF
from PIL import Image


def generate_leaderboard_pdf(
    top_items: List[Tuple[str, float]],
    output_path: str,
    image_paths: Dict[str, str] | None = None,
    title: str = "My Aesthetic Preferences Report",
):
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)

    pdf.add_page()
    pdf.set_font("Helvetica", size=28)
    pdf.cell(0, 20, title, ln=True, align="C")

    if not top_items:
        pdf.set_font("Helvetica", size=14)
        pdf.cell(0, 12, "No leaderboard data available.", ln=True, align="C")
        pdf.output(output_path)
        return output_path

    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    page_height = pdf.h - pdf.t_margin - pdf.b_margin
    x_start = pdf.l_margin
    y_start = pdf.t_margin + 16
    max_h = 58
    gutter = 4
    caption_h = 10
    dpi = 300
    px_per_mm = dpi / 25.4

    temp_files = []
    try:
        pdf.add_page()
        x = x_start
        y = y_start
        row_max_h = 0
        for idx, (filename, score) in enumerate(top_items):
            path = image_paths.get(filename) if image_paths else filename
            if not path or not os.path.exists(path):
                continue
            with Image.open(path) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size
                if img_h == 0 or img_w == 0:
                    continue
                scale = max_h / img_h
                new_w_mm = img_w * scale
                new_h_mm = max_h
                max_w_mm = page_width * 0.95
                if new_w_mm > max_w_mm:
                    scale = max_w_mm / img_w
                    new_w_mm = max_w_mm
                    new_h_mm = img_h * scale
                img_px_w = max(1, int(new_w_mm * px_per_mm))
                img_px_h = max(1, int(new_h_mm * px_per_mm))
                img = img.resize((img_px_w, img_px_h))
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=92, optimize=True)
                buffer.seek(0)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tmp.write(buffer.read())
                tmp.close()
                temp_files.append(tmp.name)
                if x + new_w_mm > x_start + page_width:
                    x = x_start
                    y = y + row_max_h + caption_h + gutter
                    row_max_h = 0
                if y + new_h_mm + caption_h > y_start + page_height:
                    pdf.add_page()
                    x = x_start
                    y = y_start
                    row_max_h = 0
                pdf.image(tmp.name, x=x, y=y, w=new_w_mm, h=new_h_mm)
                pdf.set_xy(x, y + new_h_mm + 2)
                pdf.set_font("Helvetica", size=11)
                pdf.cell(new_w_mm, 6, f"#{idx + 1}  Elo {score:.1f}", ln=True)
                x += new_w_mm + gutter
                row_max_h = max(row_max_h, new_h_mm)
        pdf.output(output_path)
        return output_path
    finally:
        for temp_path in temp_files:
            try:
                os.remove(temp_path)
            except Exception:
                pass
