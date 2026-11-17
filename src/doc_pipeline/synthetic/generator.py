"""Deterministic 200-document synthetic AP dataset generator.

Generates invoices, receipts, and purchase orders in English and German across
five distinct visual layouts, multi-page PDFs, skewed/noisy flatbed scans,
perspective-warped smartphone photos, handwritten pen notes, and controlled
business/security anomalies—paired 1-to-1 with ground-truth JSON files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.pdfgen import canvas

from doc_pipeline.schemas import ExtractedDocument, LineItem
from doc_pipeline.synthetic.catalog import (
    BENIGN_HANDWRITTEN_NOTES_DE,
    BENIGN_HANDWRITTEN_NOTES_EN,
    CATALOG_ITEMS_DE,
    CATALOG_ITEMS_EN,
    MASTER_SUPPLIERS,
    PROMPT_INJECTION_PAYLOADS,
    UNREGISTERED_SUPPLIERS,
)

REFERENCE_TODAY = date(2026, 11, 15)

LAYOUT_STYLES = [
    "modern_grid",
    "classic_boxed",
    "sidebar_accent",
    "minimal_euro",
    "thermal_receipt",
]

LABELS: dict[str, dict[str, str]] = {
    "en": {
        "invoice": "INVOICE",
        "receipt": "PAYMENT RECEIPT",
        "purchase_order": "PURCHASE ORDER",
        "doc_no": "Document No",
        "issue_date": "Issue Date",
        "due_date": "Due Date",
        "tax_id": "Tax ID",
        "bill_to": "Bill To: Acme Global Manufacturing Corp., AP Dept, 500 Howard St, San Francisco, CA",
        "sku": "SKU",
        "desc": "Description",
        "qty": "Qty",
        "unit_price": "Unit Price",
        "line_total": "Amount",
        "subtotal": "Subtotal",
        "tax": "Tax",
        "total": "Total Due",
        "continued": "Continued on next page...",
        "page": "Page",
    },
    "de": {
        "invoice": "RECHNUNG",
        "receipt": "QUITTUNG / KASSENBELEG",
        "purchase_order": "BESTELLUNG",
        "doc_no": "Belegnummer",
        "issue_date": "Belegdatum",
        "due_date": "Faelligkeitsdatum",
        "tax_id": "USt-IdNr.",
        "bill_to": "Rechnungsempfaenger: Acme Industriewerke GmbH, Kreditorenbuchhaltung, Frankfurt",
        "sku": "Art.-Nr.",
        "desc": "Bezeichnung",
        "qty": "Menge",
        "unit_price": "Einzelpreis",
        "line_total": "Betrag",
        "subtotal": "Zwischensumme",
        "tax": "MwSt.",
        "total": "Gesamtbetrag",
        "continued": "Fortsetzung auf naechster Seite...",
        "page": "Seite",
    },
}


def _fmt_money(val: float, currency: str) -> str:
    symbol = {"USD": "$", "EUR": "EUR ", "GBP": "GBP "}[currency]
    return f"{symbol}{val:,.2f}"


def _choose_supplier(rng: random.Random, lang: str, unregistered: bool) -> dict[str, Any]:
    if unregistered:
        pool = [s for s in UNREGISTERED_SUPPLIERS if (s["country"] == "DE") == (lang == "de")]
        if not pool:
            pool = UNREGISTERED_SUPPLIERS
        return rng.choice(pool)
    pool = [s for s in MASTER_SUPPLIERS if (s["country"] == "DE") == (lang == "de")]
    return rng.choice(pool)


def _build_line_items(
    rng: random.Random, lang: str, count: int
) -> list[LineItem]:
    catalog = CATALOG_ITEMS_DE if lang == "de" else CATALOG_ITEMS_EN
    chosen = [rng.choice(catalog) for _ in range(count)]
    items: list[LineItem] = []
    for idx, (sku, desc, base_price) in enumerate(chosen, start=1):
        qty = float(rng.choice([1, 2, 3, 4, 5, 8, 10, 12, 15, 20]))
        price_factor = rng.uniform(0.92, 1.12)
        unit_price = round(base_price * price_factor, 2)
        line_total = round(qty * unit_price, 2)
        items.append(
            LineItem(
                sku=f"{sku}-{idx:02d}" if rng.random() > 0.15 else None,
                description=desc,
                quantity=qty,
                unit_price=unit_price,
                line_total=line_total,
            )
        )
    return items


def _render_vector_pdf(
    doc: ExtractedDocument,
    layout: str,
    injection_banner: str | None = None,
) -> tuple[bytes, int]:
    """Render clean single- or multi-page vector PDF via ReportLab."""
    buf = io.BytesIO()
    pagesize = A4 if doc.language == "de" else LETTER
    if layout == "thermal_receipt" and len(doc.line_items) <= 6:
        pagesize = (310, 680)

    c = canvas.Canvas(buf, pagesize=pagesize)
    width, height = pagesize
    lbl = LABELS[doc.language]

    # Paginate line items: max 6 items on page 1 if multi-page, else up to 7 per page
    items_per_page = 6 if layout != "thermal_receipt" else 8
    chunks = [
        doc.line_items[i : i + items_per_page]
        for i in range(0, len(doc.line_items), items_per_page)
    ]
    total_pages = len(chunks)

    accent_map = {
        "modern_grid": colors.HexColor("#1E3A8A"),
        "classic_boxed": colors.HexColor("#1F2937"),
        "sidebar_accent": colors.HexColor("#0F766E"),
        "minimal_euro": colors.HexColor("#374151"),
        "thermal_receipt": colors.HexColor("#111827"),
    }
    accent = accent_map.get(layout, colors.HexColor("#1E3A8A"))

    for page_idx, page_items in enumerate(chunks, start=1):
        margin_x = 22 if layout == "thermal_receipt" else 44
        y = height - 42

        if layout == "sidebar_accent":
            c.setFillColor(accent)
            c.rect(0, 0, 14, height, stroke=0, fill=1)
        elif layout == "modern_grid":
            c.setFillColor(accent)
            c.rect(0, height - 12, width, 12, stroke=0, fill=1)

        # Header block
        c.setFillColor(accent)
        title_size = 13 if layout == "thermal_receipt" else 18
        c.setFont("Helvetica-Bold", title_size)
        c.drawString(margin_x, y, lbl[doc.document_type])

        c.setFillColor(colors.HexColor("#4B5563"))
        c.setFont("Helvetica", 8 if layout == "thermal_receipt" else 9)
        c.drawRightString(width - margin_x, y + 2, f"{lbl['page']} {page_idx}/{total_pages}")

        y -= 20
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 10 if layout == "thermal_receipt" else 12)
        c.drawString(margin_x, y, doc.supplier_name)
        y -= 13

        c.setFont("Helvetica", 8 if layout == "thermal_receipt" else 9)
        if doc.supplier_address:
            c.drawString(margin_x, y, doc.supplier_address[:72])
            y -= 12
        if doc.supplier_tax_id:
            c.drawString(margin_x, y, f"{lbl['tax_id']}: {doc.supplier_tax_id}")
            y -= 14

        # Metadata box / key-value row
        if layout == "classic_boxed":
            c.setStrokeColor(colors.HexColor("#9CA3AF"))
            c.rect(margin_x, y - 34, width - 2 * margin_x, 38, stroke=1, fill=0)

        c.setFont("Helvetica-Bold", 8.5 if layout == "thermal_receipt" else 9.5)
        c.drawString(margin_x + 4, y - 10, f"{lbl['doc_no']}: {doc.document_number}")
        c.drawString(margin_x + 4, y - 24, f"{lbl['issue_date']}: {doc.issue_date.isoformat()}")
        if doc.due_date:
            if layout == "thermal_receipt":
                y -= 14
                c.drawString(margin_x + 4, y - 24, f"{lbl['due_date']}: {doc.due_date.isoformat()}")
            else:
                c.drawString(
                    width / 2,
                    y - 24,
                    f"{lbl['due_date']}: {doc.due_date.isoformat()}",
                )
        if layout != "thermal_receipt":
            c.setFont("Helvetica", 8.5)
            c.drawString(width / 2, y - 10, f"Currency: {doc.currency}")

        y -= 46
        if layout != "thermal_receipt":
            c.setFont("Helvetica-Oblique", 8.5)
            c.setFillColor(colors.HexColor("#374151"))
            c.drawString(margin_x, y, lbl["bill_to"])
            c.setFillColor(colors.black)
            y -= 18

        # Table header
        c.setFillColor(colors.HexColor("#F3F4F6"))
        c.rect(margin_x, y - 4, width - 2 * margin_x, 16, stroke=0, fill=1)
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8 if layout == "thermal_receipt" else 9)

        if layout == "thermal_receipt":
            c.drawString(margin_x + 2, y, lbl["desc"])
            c.drawRightString(width - margin_x - 2, y, lbl["line_total"])
        else:
            c.drawString(margin_x + 4, y, lbl["sku"])
            c.drawString(margin_x + 82, y, lbl["desc"])
            c.drawRightString(width - margin_x - 145, y, lbl["qty"])
            c.drawRightString(width - margin_x - 72, y, lbl["unit_price"])
            c.drawRightString(width - margin_x - 4, y, lbl["line_total"])

        y -= 18
        c.setFont("Helvetica", 8 if layout == "thermal_receipt" else 9)

        for item in page_items:
            if layout == "thermal_receipt":
                c.drawString(margin_x + 2, y, item.description[:34])
                y -= 11
                c.drawString(
                    margin_x + 8,
                    y,
                    f"{item.quantity:g} x {_fmt_money(item.unit_price, doc.currency)}",
                )
                c.drawRightString(
                    width - margin_x - 2, y, _fmt_money(item.line_total, doc.currency)
                )
                y -= 15
            else:
                c.drawString(margin_x + 4, y, item.sku or "-")
                c.drawString(margin_x + 82, y, item.description[:46])
                c.drawRightString(width - margin_x - 145, y, f"{item.quantity:g}")
                c.drawRightString(
                    width - margin_x - 72, y, _fmt_money(item.unit_price, doc.currency)
                )
                c.drawRightString(
                    width - margin_x - 4, y, _fmt_money(item.line_total, doc.currency)
                )
                c.setStrokeColor(colors.HexColor("#E5E7EB"))
                c.line(margin_x, y - 4, width - margin_x, y - 4)
                y -= 18

        # Totals on the last page; continuation notice on intermediate pages
        if page_idx < total_pages:
            y -= 12
            c.setFont("Helvetica-Oblique", 9)
            c.setFillColor(colors.HexColor("#4B5563"))
            c.drawRightString(width - margin_x, y, lbl["continued"])
        else:
            y -= 14
            label_x = margin_x + 10 if layout == "thermal_receipt" else width - margin_x - 180
            val_x = width - margin_x - 4

            c.setFont("Helvetica", 9)
            c.drawString(label_x, y, f"{lbl['subtotal']}:")
            c.drawRightString(val_x, y, _fmt_money(doc.subtotal, doc.currency))
            y -= 15

            tax_pct = f" ({doc.tax_rate * 100:.1f}%)" if doc.tax_rate is not None else ""
            c.drawString(label_x, y, f"{lbl['tax']}{tax_pct}:")
            c.drawRightString(val_x, y, _fmt_money(doc.tax_amount, doc.currency))
            y -= 17

            c.setStrokeColor(accent)
            c.line(label_x, y + 11, val_x, y + 11)
            c.setFont("Helvetica-Bold", 10.5)
            c.drawString(label_x, y - 2, f"{lbl['total']} ({doc.currency}):")
            c.drawRightString(val_x, y - 2, _fmt_money(doc.total_amount, doc.currency))
            y -= 28

            if injection_banner:
                c.setStrokeColor(colors.HexColor("#DC2626"))
                c.setFillColor(colors.HexColor("#FEF2F2"))
                c.rect(margin_x, max(28, y - 28), width - 2 * margin_x, 34, stroke=1, fill=1)
                c.setFillColor(colors.HexColor("#7F1D1D"))
                c.setFont("Courier-Bold", 7.5)
                c.drawString(margin_x + 6, max(46, y - 10), injection_banner[:78])
                if len(injection_banner) > 78:
                    c.drawString(margin_x + 6, max(34, y - 22), injection_banner[78:156])

        # Embed machine-readable footer reference code (realistic ERP barcode/reference line)
        c.setFillColor(colors.HexColor("#6B7280"))
        c.setFont("Courier", 7.5)
        c.drawString(
            margin_x,
            16,
            f"AP-REF::{doc.document_number}::{doc.currency}::{doc.total_amount:.2f}",
        )

        c.showPage()

    c.save()
    return buf.getvalue(), total_pages


def _pdf_to_pil_pages(pdf_bytes: bytes, dpi: int = 200) -> list[Image.Image]:
    pdf = pdfium.PdfDocument(pdf_bytes)
    scale = dpi / 72.0
    pages: list[Image.Image] = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=scale)
        pages.append(bitmap.to_pil().convert("RGB"))
    pdf.close()
    return pages


def _draw_handwritten_note(img: Image.Image, note: str, rng: random.Random) -> Image.Image:
    """Overlay a realistic slanted ink annotation simulating handwritten AP clerk notes."""
    out = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", (540, 110), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    ink_colors = [
        (22, 58, 156, 215),   # ballpoint blue
        (168, 28, 38, 210),   # red review pen
        (35, 38, 45, 210),    # dark rollerball
    ]
    ink = rng.choice(ink_colors)
    font = ImageFont.load_default(size=19)

    # Draw character by character with slight vertical jitter to mimic natural handwriting
    cursor_x = 16
    base_y = 42
    for ch in note[:58]:
        jitter_y = base_y + rng.randint(-2, 2)
        draw.text((cursor_x, jitter_y), ch, fill=ink, font=font)
        bbox = draw.textbbox((cursor_x, jitter_y), ch, font=font)
        cursor_x += max(7, (bbox[2] - bbox[0])) + rng.randint(0, 1)

    # Underline scribble
    draw.line(
        [(14, base_y + 27), (min(520, cursor_x), base_y + 24 + rng.randint(-3, 3))],
        fill=ink,
        width=2,
    )

    angle = rng.uniform(-7.5, 6.5)
    rotated = overlay.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    max_x = max(20, out.width - rotated.width - 30)
    pos_x = rng.randint(20, max_x)
    pos_y = rng.randint(int(out.height * 0.62), max(int(out.height * 0.63), out.height - rotated.height - 45))
    out.alpha_composite(rotated, dest=(pos_x, pos_y))
    return out.convert("RGB")


def _degrade_noisy_scan(img: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate a skewed, dusty, slightly blurred office scanner page."""
    # Off-white paper tint
    arr = np.asarray(img, dtype=np.float32)
    tint = np.array([rng.uniform(0.95, 0.99), rng.uniform(0.94, 0.98), rng.uniform(0.91, 0.96)])
    arr = arr * tint

    # Additive Gaussian sensor noise + occasional dark toner speckles
    sigma = rng.uniform(5.0, 11.5)
    np_rng = np.random.default_rng(rng.randint(1, 10_000_000))
    noise = np_rng.normal(0.0, sigma, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255)

    speckle_mask = np_rng.random(arr.shape[:2]) < 0.0012
    arr[speckle_mask] = rng.randint(25, 85)

    noisy = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    angle = rng.choice([-1, 1]) * rng.uniform(1.4, 3.6)
    rotated = noisy.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(238, 236, 230),
    )
    return rotated.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.55)))


def _degrade_phone_photo(img: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate a handheld smartphone photo on an office desk with uneven lighting."""
    w, h = img.size
    desk = Image.new("RGB", (w + 120, h + 120), (194, 184, 168))
    np_rng = np.random.default_rng(rng.randint(1, 10_000_000))
    desk_arr = np.asarray(desk, dtype=np.float32)
    desk_arr += np_rng.normal(0, 7.0, size=desk_arr.shape)
    desk = Image.fromarray(np.clip(desk_arr, 0, 255).astype(np.uint8))

    # Slight rotation + border placement on desk
    tilted = ImageOps.expand(img, border=3, fill=(220, 218, 212)).rotate(
        rng.uniform(-3.2, 3.2),
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(194, 184, 168),
    )
    ox = max(10, (desk.width - tilted.width) // 2)
    oy = max(10, (desk.height - tilted.height) // 2)
    desk.paste(tilted, (ox, oy))

    # Directional lighting / shadow gradient
    arr = np.asarray(desk, dtype=np.float32)
    gy = np.linspace(rng.uniform(0.84, 0.94), rng.uniform(0.98, 1.04), arr.shape[0])[:, None, None]
    gx = np.linspace(rng.uniform(0.88, 1.02), rng.uniform(0.92, 1.03), arr.shape[1])[None, :, None]
    arr = np.clip(arr * gy * gx, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def generate_dataset(output_dir: str | Path, count: int = 200, seed: int = 42) -> dict[str, Any]:
    """Generate `count` synthetic AP documents + 1:1 ground-truth JSONs deterministically."""
    root = Path(output_dir)
    docs_dir = root / "documents"
    gt_dir = root / "ground_truth"
    docs_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    manifest_items: list[dict[str, Any]] = []
    generated_for_duplicates: list[ExtractedDocument] = []

    # Stratified anomaly plan across 200 documents
    anomaly_schedule: dict[int, str] = {}
    if count >= 40:
        for idx in [12, 38, 64, 89, 114, 139, 158, 174, 186, 195]:
            if idx <= count:
                anomaly_schedule[idx] = "duplicate_invoice"
        for idx in [7, 31, 57, 83, 108, 133, 163, 191]:
            if idx <= count:
                anomaly_schedule[idx] = "math_mismatch"
        for idx in [15, 43, 71, 98, 126, 151, 177, 197]:
            if idx <= count:
                anomaly_schedule[idx] = "unknown_supplier"
        for idx in [19, 68, 119, 167, 199]:
            if idx <= count:
                anomaly_schedule[idx] = "bad_dates"
        for idx in [22, 76, 144, 188]:
            if idx <= count:
                anomaly_schedule[idx] = "prompt_injection"
    else:
        if count >= 6:
            anomaly_schedule[2] = "math_mismatch"
            anomaly_schedule[4] = "unknown_supplier"
            anomaly_schedule[5] = "prompt_injection"
            anomaly_schedule[6] = "duplicate_invoice"

    doc_types = ["invoice"] * 9 + ["receipt"] * 5 + ["purchase_order"] * 6

    for i in range(1, count + 1):
        doc_id = f"doc_{i:04d}"
        anomaly = anomaly_schedule.get(i, "none")

        lang = "de" if (i % 3 == 0) else "en"
        doc_type = doc_types[(i - 1) % len(doc_types)]
        if anomaly == "duplicate_invoice" and generated_for_duplicates:
            # Clone an earlier valid invoice's commercial data onto a fresh scan/layout
            source_doc = rng.choice(generated_for_duplicates)
            extracted = source_doc.model_copy(deep=True)
            lang = extracted.language
            doc_type = extracted.document_type
        else:
            unregistered = anomaly == "unknown_supplier"
            sup = _choose_supplier(rng, lang=lang, unregistered=unregistered)

            # Decide condition first so multipage gets 9-14 line items
            is_multipage = (i % 6 == 0) and doc_type in ("invoice", "purchase_order")
            n_items = rng.randint(9, 14) if is_multipage else rng.randint(2, 5)
            items = _build_line_items(rng, lang=lang, count=n_items)

            subtotal = round(sum(it.line_total for it in items), 2)
            tax_rate = rng.choice([0.0, 0.07, 0.0825, 0.10, 0.19, 0.20])
            if lang == "de":
                tax_rate = rng.choice([0.07, 0.19])
            tax_amount = round(subtotal * tax_rate, 2)
            total_amount = round(subtotal + tax_amount, 2)

            if anomaly == "math_mismatch":
                # Vendor printed an arithmetic error on total_amount (+$47.50 overcharge)
                total_amount = round(total_amount + 47.50, 2)

            days_ago = rng.randint(3, 90)
            issue_dt = REFERENCE_TODAY - timedelta(days=days_ago)
            due_dt: date | None = (
                issue_dt + timedelta(days=rng.choice([14, 30, 45]))
                if doc_type != "receipt"
                else None
            )

            if anomaly == "bad_dates":
                if i % 2 == 0:
                    # Impossible due date earlier than issue date
                    due_dt = issue_dt - timedelta(days=15)
                else:
                    # Future-dated invoice (+40 days ahead)
                    issue_dt = REFERENCE_TODAY + timedelta(days=40)
                    due_dt = issue_dt + timedelta(days=30)

            prefix = {"invoice": "INV", "receipt": "RCP", "purchase_order": "PO"}[doc_type]
            doc_number = f"{prefix}-2026-{1000 + i:04d}"

            has_note = (i % 5 in (0, 2)) or (anomaly == "prompt_injection")
            note_text: str | None = None
            if anomaly == "prompt_injection":
                note_text = PROMPT_INJECTION_PAYLOADS[i % len(PROMPT_INJECTION_PAYLOADS)]
            elif has_note:
                notes_pool = (
                    BENIGN_HANDWRITTEN_NOTES_DE if lang == "de" else BENIGN_HANDWRITTEN_NOTES_EN
                )
                note_text = rng.choice(notes_pool)

            extracted = ExtractedDocument(
                document_type=doc_type,  # type: ignore[arg-type]
                language=lang,  # type: ignore[arg-type]
                document_number=doc_number,
                supplier_name=sup["name"],
                supplier_tax_id=sup["tax_id"],
                supplier_address=sup["address"],
                issue_date=issue_dt,
                due_date=due_dt,
                currency=sup["default_currency"],
                subtotal=subtotal,
                tax_rate=tax_rate,
                tax_amount=tax_amount,
                total_amount=total_amount,
                line_items=items,
                handwritten_notes=note_text,
            )

            if anomaly == "none" and doc_type == "invoice":
                generated_for_duplicates.append(extracted)

        # Assign layout & physical condition
        if doc_type == "receipt" and len(extracted.line_items) <= 5 and i % 2 == 1:
            layout = "thermal_receipt"
        else:
            layout = LAYOUT_STYLES[(i - 1) % 4]

        if len(extracted.line_items) > 6:
            condition = "multipage_pdf"
        elif anomaly == "prompt_injection":
            condition = "noisy_scan" if i % 2 == 0 else "clean_pdf"
        elif i % 4 == 1:
            condition = "clean_pdf"
        elif i % 4 == 2:
            condition = "noisy_scan"
        elif i % 4 == 3:
            condition = "phone_photo"
        else:
            condition = "clean_pdf"

        injection_banner = extracted.handwritten_notes if anomaly == "prompt_injection" else None
        vector_pdf_bytes, page_count = _render_vector_pdf(
            extracted, layout=layout, injection_banner=injection_banner
        )

        # Render physical artifact based on condition
        if condition in ("clean_pdf", "multipage_pdf") and not (
            extracted.handwritten_notes and anomaly != "prompt_injection"
        ):
            file_ext = ".pdf"
            raw_bytes = vector_pdf_bytes
        else:
            pil_pages = _pdf_to_pil_pages(vector_pdf_bytes, dpi=200)
            processed_pages: list[Image.Image] = []
            for p_idx, p_img in enumerate(pil_pages):
                cur = p_img
                if (
                    extracted.handwritten_notes
                    and p_idx == len(pil_pages) - 1
                    and anomaly != "prompt_injection"
                ):
                    cur = _draw_handwritten_note(cur, extracted.handwritten_notes, rng)
                if condition == "noisy_scan":
                    cur = _degrade_noisy_scan(cur, rng)
                elif condition == "phone_photo":
                    cur = _degrade_phone_photo(cur, rng)
                processed_pages.append(cur)

            if len(processed_pages) > 1:
                file_ext = ".pdf"
                out_buf = io.BytesIO()
                processed_pages[0].save(
                    out_buf,
                    format="PDF",
                    save_all=True,
                    append_images=processed_pages[1:],
                    resolution=200.0,
                )
                raw_bytes = out_buf.getvalue()
            elif condition == "phone_photo":
                file_ext = ".jpg"
                out_buf = io.BytesIO()
                processed_pages[0].save(out_buf, format="JPEG", quality=84)
                raw_bytes = out_buf.getvalue()
            else:
                file_ext = ".png"
                out_buf = io.BytesIO()
                processed_pages[0].save(out_buf, format="PNG")
                raw_bytes = out_buf.getvalue()

        file_name = f"{doc_id}{file_ext}"
        doc_path = docs_dir / file_name
        doc_path.write_bytes(raw_bytes)
        sha256_hex = hashlib.sha256(raw_bytes).hexdigest()

        subsets = ["all"]
        if i <= 48 or anomaly != "none":
            if len([m for m in manifest_items if "benchmark" in m["subsets"]]) < 48:
                subsets.append("benchmark")
        if i <= 24:
            subsets.append("ci")

        gt_payload = {
            "doc_id": doc_id,
            "file_name": file_name,
            "sha256": sha256_hex,
            "layout": layout,
            "condition": condition,
            "page_count": page_count,
            "has_handwritten_note": extracted.handwritten_notes is not None,
            "anomaly_type": anomaly,
            "expected_needs_review": anomaly != "none",
            "subsets": subsets,
            "ground_truth": extracted.model_dump(mode="json"),
        }
        (gt_dir / f"{doc_id}.json").write_text(
            json.dumps(gt_payload, indent=2), encoding="utf-8"
        )
        manifest_items.append(
            {
                "doc_id": doc_id,
                "file_name": file_name,
                "sha256": sha256_hex,
                "document_type": extracted.document_type,
                "document_number": extracted.document_number,
                "language": extracted.language,
                "condition": condition,
                "page_count": page_count,
                "anomaly_type": anomaly,
                "subsets": subsets,
            }
        )

    summary = {
        "seed": seed,
        "reference_today": REFERENCE_TODAY.isoformat(),
        "total_documents": len(manifest_items),
        "by_type": {
            t: sum(1 for m in manifest_items if m["document_type"] == t)
            for t in ("invoice", "receipt", "purchase_order")
        },
        "by_language": {
            lang: sum(1 for m in manifest_items if m["language"] == lang)
            for lang in ("en", "de")
        },
        "by_condition": {
            cond: sum(1 for m in manifest_items if m["condition"] == cond)
            for cond in ("clean_pdf", "multipage_pdf", "noisy_scan", "phone_photo")
        },
        "by_anomaly": {
            anom: sum(1 for m in manifest_items if m["anomaly_type"] == anom)
            for anom in (
                "none",
                "duplicate_invoice",
                "math_mismatch",
                "unknown_supplier",
                "bad_dates",
                "prompt_injection",
            )
        },
        "documents": manifest_items,
    }
    (root / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic AP document dataset")
    parser.add_argument("--output-dir", default="./data/synthetic", help="Target synthetic folder")
    parser.add_argument("--count", type=int, default=200, help="Number of documents to generate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip regeneration if manifest with matching count already exists",
    )
    args = parser.parse_args()

    manifest_path = Path(args.output_dir) / "manifest.json"
    if args.skip_if_exists and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("total_documents") == args.count and existing.get("seed") == args.seed:
            print(
                f"[dataset] Reusing existing {args.count}-document synthetic corpus at {args.output_dir}"
            )
            return

    summary = generate_dataset(args.output_dir, count=args.count, seed=args.seed)
    print(
        f"[dataset] Generated {summary['total_documents']} synthetic documents -> "
        f"types={summary['by_type']}, languages={summary['by_language']}, "
        f"conditions={summary['by_condition']}"
    )


if __name__ == "__main__":
    main()
