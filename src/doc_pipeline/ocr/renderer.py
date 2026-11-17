"""Page image renderer (pypdfium2 + Pillow) and Tesseract OCR fallback engine."""

from __future__ import annotations

import base64
import io
import logging
import shutil

import pypdfium2 as pdfium
import pytesseract
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


def detect_mime_type(file_name: str, raw_bytes: bytes) -> str:
    if raw_bytes.startswith(b"%PDF"):
        return "application/pdf"
    if raw_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if raw_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    lower = file_name.lower()
    if lower.endswith(".pdf"):
        return "application/pdf"
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith((".tif", ".tiff")):
        return "image/tiff"
    return "application/octet-stream"


def render_document_to_png_pages(
    raw_bytes: bytes, mime_type: str, dpi: int = 200
) -> tuple[list[bytes], str]:
    """Render any PDF or image upload into normalized RGB PNG page byte arrays.

    Also extracts any embedded native PDF text layer so safety scanners inspect
    embedded text even when vision mode is used.
    """
    pages_png: list[bytes] = []
    embedded_text_parts: list[str] = []

    if mime_type == "application/pdf" or raw_bytes.startswith(b"%PDF"):
        pdf = pdfium.PdfDocument(raw_bytes)
        scale = dpi / 72.0
        for idx in range(len(pdf)):
            page = pdf[idx]
            textpage = page.get_textpage()
            txt = textpage.get_text_range()
            if txt:
                embedded_text_parts.append(f"--- PDF_TEXT_PAGE {idx + 1} ---\n{txt.strip()}")
            textpage.close()

            bitmap = page.render(scale=scale)
            pil_img = bitmap.to_pil().convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            pages_png.append(buf.getvalue())
            page.close()
        pdf.close()
    else:
        img = Image.open(io.BytesIO(raw_bytes))
        n_frames = getattr(img, "n_frames", 1)
        for frame_idx in range(n_frames):
            if n_frames > 1:
                img.seek(frame_idx)
            frame = ImageOps.exif_transpose(img).convert("RGB")
            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            pages_png.append(buf.getvalue())

    return pages_png, "\n\n".join(embedded_text_parts)


def preprocess_for_ocr(png_bytes: bytes) -> Image.Image:
    """Enhance page contrast and reduce scan speckle before running Tesseract OCR."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    gray = ImageOps.grayscale(img)
    # Upscale compact thermal receipts for sharper glyph recognition
    if gray.width < 900:
        scale = 2
        gray = gray.resize((gray.width * scale, gray.height * scale), Image.Resampling.BICUBIC)
    contrasted = ImageOps.autocontrast(gray, cutoff=1)
    denoised = contrasted.filter(ImageFilter.MedianFilter(size=3))
    return denoised


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def extract_ocr_text_from_pages(
    pages_png: list[bytes], fallback_embedded_text: str = ""
) -> str:
    """Run local Tesseract OCR across rendered page PNGs for text-only model endpoints."""
    if not tesseract_available():
        if fallback_embedded_text.strip():
            logger.warning(
                "Tesseract binary not on PATH; falling back to embedded PDF text layer."
            )
            return fallback_embedded_text
        raise RuntimeError(
            "Tesseract OCR binary is required for non-vision image extraction "
            "(installed automatically in the Docker container)."
        )

    available_langs = set(pytesseract.get_languages(config=""))
    lang_spec = "eng+deu" if {"eng", "deu"}.issubset(available_langs) else "eng"

    chunks: list[str] = []
    for idx, png_bytes in enumerate(pages_png, start=1):
        prepped = preprocess_for_ocr(png_bytes)
        page_text = pytesseract.image_to_string(
            prepped,
            lang=lang_spec,
            config="--oem 3 --psm 6",
        )
        chunks.append(f"--- OCR_PAGE {idx} ---\n{page_text.strip()}")
    return "\n\n".join(chunks)


def extract_embedded_pdf_text(raw_bytes: bytes, mime_type: str) -> str:
    """Fast extraction of native PDF text layer without rasterizing bitmaps."""
    if not (mime_type == "application/pdf" or raw_bytes.startswith(b"%PDF")):
        return ""
    parts: list[str] = []
    pdf = pdfium.PdfDocument(raw_bytes)
    for idx in range(len(pdf)):
        page = pdf[idx]
        textpage = page.get_textpage()
        txt = textpage.get_text_range()
        if txt:
            parts.append(f"--- PDF_TEXT_PAGE {idx + 1} ---\n{txt.strip()}")
        textpage.close()
        page.close()
    pdf.close()
    return "\n\n".join(parts)


def render_first_page_png(raw_bytes: bytes, dpi: int = 96) -> bytes | None:
    """Render only page 1 quickly for visual fingerprinting."""
    if raw_bytes.startswith(b"%PDF"):
        pdf = pdfium.PdfDocument(raw_bytes)
        if len(pdf) == 0:
            pdf.close()
            return None
        page = pdf[0]
        bitmap = page.render(scale=dpi / 72.0)
        pil_img = bitmap.to_pil().convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG", compress_level=1)
        page.close()
        pdf.close()
        return buf.getvalue()
    img = Image.open(io.BytesIO(raw_bytes))
    frame = ImageOps.exif_transpose(img).convert("RGB")
    buf = io.BytesIO()
    frame.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def png_bytes_to_data_uri(png_bytes: bytes, max_edge: int = 1600) -> str:
    """Resize oversized scans if needed and encode as an OpenAI base64 image_url data URI."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    if max(img.width, img.height) > max_edge:
        img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=2)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"
