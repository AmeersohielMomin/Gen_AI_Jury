"""Build the complete, data-backed GenAI Jury Professional Report (.docx).

Replaces the early proposal draft with the full Topic 146 ISIC 2019 study,
verified 3-arm classifier results, cGAN domain-shift audit, interactive FastAPI
& Grad-CAM Studio documentation, and Future Healthcare Imaging Roadmap.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
CACHE_PATH = ARTIFACT_DIR / "jury_eval_cache.json"
OUT = PROJECT_DIR / "GenAI_Jury_Professional_Report.docx"

TEAM_MEMBERS = [
    ("Ameersohiel N Momin", "Group Member"),
    ("Avneet Bhatt", "Group Member"),
    ("Aditi Gaikwad", "Group Member"),
    ("Dewang Walke", "Group Member"),
]

REPO_URL = "https://github.com/AmeersohielMomin/Gen_AI_Jury.git"
REPORT_DATE = "25 November 2026"

# Design palette: Clinical Navy + Surgical Teal + Amber Target Accent
NAVY = "0B2545"
TEAL = "0F766E"
GOLD = "B45309"
INK = "172B4D"
MUTED = "5B677A"
PALE_BLUE = "EAF2F8"
PALE_TEAL = "E8F5F3"
PALE_GOLD = "FEF3C7"
PALE_GRAY = "F4F6F8"
PALE_RED = "FDECEC"
RED = "9B1C1C"


def set_font(run, name="Aptos", size=None, color=None, bold=None, italic=None):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rpr.rFonts.set(qn("w:ascii"), name)
    rpr.rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:fill"), fill)
    shd.set(qn("w:val"), "clear")


def set_cell_margins(cell, top=75, start=110, bottom=75, end=110):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in("w:tcMar")
    if tcMar is None:
        tcMar = OxmlElement("w:tcMar")
        tcPr.append(tcMar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tcMar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tcMar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_cell_border(cell, color="D7DEE8", sz="6", val="single"):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), val)
        element.set(qn("w:sz"), sz)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_table_geometry(table, widths, indent=80):
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tbl = table._tbl
    tblPr = tbl.tblPr
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW")
        tblPr.append(tblW)
    tblW.set(qn("w:w"), str(sum(widths)))
    tblW.set(qn("w:type"), "dxa")
    tblInd = tblPr.find(qn("w:tblInd"))
    if tblInd is None:
        tblInd = OxmlElement("w:tblInd")
        tblPr.append(tblInd)
    tblInd.set(qn("w:w"), str(indent))
    tblInd.set(qn("w:type"), "dxa")
    grid = tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            cell.width = Inches(widths[idx] / 1440)
            tcPr = cell._tc.get_or_add_tcPr()
            tcW = tcPr.find(qn("w:tcW"))
            if tcW is None:
                tcW = OxmlElement("w:tcW")
                tcPr.append(tcW)
            tcW.set(qn("w:w"), str(widths[idx]))
            tcW.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)


def set_keep_with_next(paragraph, value=True):
    pPr = paragraph._p.get_or_add_pPr()
    keep = pPr.find(qn("w:keepNext"))
    if keep is None:
        keep = OxmlElement("w:keepNext")
        pPr.append(keep)
    keep.set(qn("w:val"), "1" if value else "0")


def add_page_number(paragraph):
    run = paragraph.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = " PAGE "
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    set_font(run, size=9, color=MUTED)


def create_numbering(doc):
    numbering = doc.part.numbering_part.element
    abstract_ids = [
        int(x.get(qn("w:abstractNumId")))
        for x in numbering.findall(qn("w:abstractNum"))
        if x.get(qn("w:abstractNumId"))
    ]
    num_ids = [
        int(x.get(qn("w:numId")))
        for x in numbering.findall(qn("w:num"))
        if x.get(qn("w:numId"))
    ]
    abstract_id = max(abstract_ids or [0]) + 1
    num_id = max(num_ids or [0]) + 1

    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "singleLevel")
    abstract.append(multi)
    lvl = OxmlElement("w:lvl")
    lvl.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    num_fmt = OxmlElement("w:numFmt")
    num_fmt.set(qn("w:val"), "decimal")
    lvl_text = OxmlElement("w:lvlText")
    lvl_text.set(qn("w:val"), "%1.")
    lvl_jc = OxmlElement("w:lvlJc")
    lvl_jc.set(qn("w:val"), "left")
    ppr = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), "540")
    tabs.append(tab)
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "540")
    ind.set(qn("w:hanging"), "280")
    ppr.append(tabs)
    ppr.append(ind)
    for child in (start, num_fmt, lvl_text, lvl_jc, ppr):
        lvl.append(child)
    abstract.append(lvl)
    numbering.append(abstract)

    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    abs_ref = OxmlElement("w:abstractNumId")
    abs_ref.set(qn("w:val"), str(abstract_id))
    num.append(abs_ref)
    numbering.append(num)
    return num_id


def add_rule(paragraph, color=GOLD, size="16"):
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = pPr.find(qn("w:pBdr"))
    if pBdr is None:
        pBdr = OxmlElement("w:pBdr")
        pPr.append(pBdr)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), "6")
    bottom.set(qn("w:color"), color)
    pBdr.append(bottom)


def add_body(doc, text, style="Normal", bold_lead=None):
    p = doc.add_paragraph(style=style)
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        set_font(r, size=10.5, bold=True, color=INK)
        r2 = p.add_run(text[len(bold_lead):])
        set_font(r2, size=10.5, color=INK)
    else:
        r = p.add_run(text)
        set_font(r, size=10.5, color=INK)
    return p


def add_bullet(doc, text, bold_lead=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.first_line_indent = Inches(-0.18)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.2
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        set_font(r, size=10.2, bold=True, color=INK)
        r2 = p.add_run(text[len(bold_lead):])
        set_font(r2, size=10.2, color=INK)
    else:
        r = p.add_run(text)
        set_font(r, size=10.2, color=INK)
    return p


def add_number(doc, text, num_id=None, bold_lead=None):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.first_line_indent = Inches(-0.18)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.2
    if num_id is not None:
        pPr = p._p.get_or_add_pPr()
        numPr = OxmlElement("w:numPr")
        ilvl = OxmlElement("w:ilvl")
        ilvl.set(qn("w:val"), "0")
        num = OxmlElement("w:numId")
        num.set(qn("w:val"), str(num_id))
        numPr.append(ilvl)
        numPr.append(num)
        pPr.append(numPr)
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        set_font(r, size=10.2, bold=True, color=INK)
        r2 = p.add_run(text[len(bold_lead):])
        set_font(r2, size=10.2, color=INK)
    else:
        r = p.add_run(text)
        set_font(r, size=10.2, color=INK)
    return p


def add_callout(doc, label, text, fill=PALE_TEAL, accent=TEAL):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9400], indent=80)
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    set_cell_border(cell, color=accent, sz="10")
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(label.upper() + "  ")
    set_font(r, size=9, color=accent, bold=True)
    r2 = p.add_run(text)
    set_font(r2, size=10, color=INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_section_title(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    r = p.add_run(text)
    set_font(r, size={1: 15.5, 2: 12.5, 3: 11.2}[level], color=TEAL if level == 1 else NAVY, bold=True)
    set_keep_with_next(p)
    return p


def add_table(doc, headers, rows, widths, header_fill=PALE_BLUE, font_size=9.1):
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths, indent=80)
    header = table.rows[0]
    repeat_header(header)
    for i, text in enumerate(headers):
        cell = header.cells[i]
        set_cell_shading(cell, header_fill)
        set_cell_border(cell, color="B8C9D9", sz="8")
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(text)
        set_font(r, size=9.0, color=NAVY, bold=True)
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, text in enumerate(row):
            set_cell_border(cells[i], color="D7DEE8", sz="6")
            if ridx % 2 == 1:
                set_cell_shading(cells[i], PALE_GRAY)
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.08
            r = p.add_run(str(text))
            set_font(r, size=font_size, color=INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_figure(doc, image_path: Path, caption: str, width_inches: float = 6.35):
    if not image_path.exists():
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(4)
    p_img.paragraph_format.space_after = Pt(3)
    set_keep_with_next(p_img)
    run = p_img.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(8)
    rc = p_cap.add_run(caption)
    set_font(rc, size=8.8, color=MUTED, italic=True)


def setup_document():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(0.75)
    sec.bottom_margin = Inches(0.72)
    sec.left_margin = Inches(0.95)
    sec.right_margin = Inches(0.95)
    sec.header_distance = Inches(0.35)
    sec.footer_distance = Inches(0.35)
    sec.different_first_page_header_footer = True

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.28
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    for name, size, color, before, after in [
        ("Heading 1", 15.5, TEAL, 14, 7),
        ("Heading 2", 12.5, NAVY, 10, 5),
        ("Heading 3", 11.2, NAVY, 7, 3),
    ]:
        st = styles[name]
        st.font.name = "Aptos Display"
        st._element.rPr.rFonts.set(qn("w:ascii"), "Aptos Display")
        st._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos Display")
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor.from_string(color)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True

    header = sec.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    hp.paragraph_format.space_after = Pt(0)
    r = hp.add_run("GENAI JURY · TOPIC 146")
    set_font(r, size=8.5, color=TEAL, bold=True)
    r = hp.add_run("   |   CONTROLLED CGAN SKIN-LESION AUGMENTATION & ROADMAP REPORT")
    set_font(r, size=8.5, color=MUTED)

    footer = sec.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    fp.paragraph_format.space_before = Pt(0)
    r = fp.add_run("A. N. Momin, A. Bhatt, A. Gaikwad, D. Walke   |   Page ")
    set_font(r, size=8.5, color=MUTED)
    add_page_number(fp)
    return doc


def load_verified_cache() -> dict:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Expected {CACHE_PATH} to exist. Run `python3 -m scripts.build_eval_cache` first."
        )
    cache = json.loads(CACHE_PATH.read_text())
    ref_csv = ARTIFACT_DIR / "classifier_comparison.csv"
    if ref_csv.exists():
        ref = pd.read_csv(ref_csv).set_index("arm")
        for arm in cache["arm_order"]:
            if arm in ref.index:
                for col in ref.columns:
                    if col in cache["arms"][arm]["metrics"]:
                        cache["arms"][arm]["metrics"][col] = float(ref.loc[arm, col])
    return cache


def generate_report_figures(cache: dict, fig_dir: Path) -> dict[str, Path]:
    """Render publication-grade Matplotlib figures directly from verified cache metrics."""
    figs: dict[str, Path] = {}

    # 1. ISIC 2019 Class Imbalance Chart
    counts = cache["splits"]["train"]["counts"]
    ordered = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in ordered]
    vals = [v for _, v in ordered]
    colors = ["#B45309" if n == "SCC" else "#0F766E" for n in names]

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    bars = ax.bar(names, vals, color=colors, width=0.62, edgecolor="#0B2545", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 180,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold" if v == 429 else "normal",
            color="#172B4D",
        )
    ax.set_title("ISIC 2019 Training Partition Class Imbalance (18,623 Real Images, 22:1 NV:SCC Ratio)", fontsize=11, fontweight="bold", color="#0B2545")
    ax.set_ylabel("Training Image Count", fontsize=9.5, color="#172B4D")
    ax.set_ylim(0, max(vals) * 1.14)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p1 = fig_dir / "fig1_class_imbalance.png"
    plt.savefig(p1, dpi=220)
    plt.close(fig)
    figs["imbalance"] = p1

    # 2. Real_only Confusion Matrix + Per-Class F1 Comparison
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.15, 1.0]})
    cls_names = cache["class_names"]
    cm = np.array(cache["arms"]["real_only"]["confusion_matrix"], dtype=float)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    im = axes[0].imshow(cm_norm, cmap="Teal" if "Teal" in plt.colormaps() else "YlGnBu", vmin=0, vmax=0.9)
    axes[0].set_xticks(range(len(cls_names)), cls_names, rotation=35, fontsize=8.5)
    axes[0].set_yticks(range(len(cls_names)), cls_names, fontsize=8.5)
    axes[0].set_xlabel("Predicted Diagnosis", fontsize=9)
    axes[0].set_ylabel("True Diagnosis", fontsize=9)
    axes[0].set_title("Baseline (real_only) Confusion Matrix (N=3,664)", fontsize=10, fontweight="bold", color="#0B2545")
    for r in range(len(cls_names)):
        for c in range(len(cls_names)):
            val = int(cm[r, c])
            txt_col = "white" if cm_norm[r, c] > 0.48 else "#172B4D"
            axes[0].text(c, r, str(val), ha="center", va="center", fontsize=7.5, color=txt_col)

    # Right: Per-class F1 across real_only vs cgan_scc
    x = np.arange(len(cls_names))
    w = 0.36
    f1_real = [r["f1"] for r in cache["arms"]["real_only"]["per_class"]]
    f1_cgan = [r["f1"] for r in cache["arms"]["cgan_scc"]["per_class"]]
    axes[1].bar(x - w / 2, f1_real, width=w, label="real_only (Baseline)", color="#0F766E")
    axes[1].bar(x + w / 2, f1_cgan, width=w, label="cgan_scc (+600 Fakes)", color="#D97706")
    axes[1].set_xticks(x, cls_names, rotation=35, fontsize=8.5)
    axes[1].set_ylabel("Held-Out Test F1-Score", fontsize=9)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Per-Class F1: Real Only vs. cGAN SCC", fontsize=10, fontweight="bold", color="#0B2545")
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    p2 = fig_dir / "fig2_cm_and_per_class.png"
    plt.savefig(p2, dpi=220)
    plt.close(fig)
    figs["cm_per_class"] = p2

    # 3. cGAN Audit Composite: 80-Epoch Loss + Real vs Synthetic Visual Comparison
    loss_rows = pd.DataFrame(cache["cgan_audit"]["loss_history"])
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), gridspec_kw={"width_ratios": [1.55, 0.85, 0.85]})
    axes[0].plot(loss_rows["epoch"], loss_rows["discriminator_loss"], color="#9B1C1C", linewidth=2.0, label="Discriminator Hinge Loss (~2.000)")
    axes[0].plot(loss_rows["epoch"], loss_rows["generator_loss"], color="#0F766E", linewidth=2.0, label="Generator Loss")
    axes[0].set_title("80-Epoch cGAN Hinge Loss Trajectory", fontsize=10, fontweight="bold", color="#0B2545")
    axes[0].set_xlabel("Epoch", fontsize=9)
    axes[0].set_ylabel("Loss", fontsize=9)
    axes[0].set_ylim(0, 2.25)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    real_panel = ARTIFACT_DIR / "realism_survey" / "panel_01.png"
    fake_panel = ARTIFACT_DIR / "realism_survey" / "panel_02.png"
    if real_panel.exists():
        with Image.open(real_panel) as im_real:
            axes[1].imshow(im_real.convert("RGB").resize((256, 256)))
        axes[1].set_title("Real ISIC SCC (panel_01)", fontsize=9.2, fontweight="bold", color="#0F766E")
    axes[1].axis("off")

    if fake_panel.exists():
        with Image.open(fake_panel) as im_fake:
            axes[2].imshow(im_fake.convert("RGB").resize((256, 256), Image.Resampling.NEAREST))
        axes[2].set_title("cGAN SCC (panel_02, Grid)", fontsize=9.2, fontweight="bold", color="#9B1C1C")
    axes[2].axis("off")

    plt.tight_layout()
    p3 = fig_dir / "fig3_cgan_audit.png"
    plt.savefig(p3, dpi=220)
    plt.close(fig)
    figs["cgan_audit"] = p3

    return figs


def main():
    cache = load_verified_cache()
    if OUT.exists():
        OUT.unlink()

    doc = setup_document()
    step_num_id = create_numbering(doc)
    impl_num_id = create_numbering(doc)

    with tempfile.TemporaryDirectory() as tmp_dir:
        figs = generate_report_figures(cache, Path(tmp_dir))

        # ---------------- COVER PAGE ----------------
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(34)
        p.paragraph_format.space_after = Pt(8)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("GENAI JURY  ·  TOPIC 146 FINAL PROJECT REPORT")
        set_font(r, size=11, color=GOLD, bold=True)

        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run("Controlled cGAN Augmentation & Explainable Skin-Lesion Classification")
        set_font(r, name="Aptos Display", size=24, color=NAVY, bold=True)

        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(16)
        r = p.add_run(
            "Empirical Audit on ISIC 2019, Live Grad-CAM Interactive Jury Studio, and Future Multi-Modal Healthcare Roadmap"
        )
        set_font(r, size=12.5, color=TEAL)

        rule = doc.add_paragraph()
        rule.paragraph_format.space_after = Pt(16)
        add_rule(rule, color=GOLD, size="20")

        add_section_title(doc, "Project Team", level=2)
        add_table(
            doc,
            ["MEMBER NAME", "PROJECT ROLE"],
            TEAM_MEMBERS,
            [5200, 4200],
            header_fill=PALE_TEAL,
            font_size=9.5,
        )

        metadata_rows = [
            ("Project Focus", "Controlled cGAN minority-class (SCC) augmentation, explainability UI & multi-modal roadmap"),
            ("Dataset & Split Policy", "ISIC 2019 (25,331 real dermoscopic images; 8 active diagnoses; zero lesion-group leakage)"),
            ("Evaluated Hardware", "Apple Silicon MPS (5.33 GB working set, 68% cap) with PyTorch 2.10 & EfficientNet-B0"),
            ("Current Status", "All models trained & verified; FastAPI + Grad-CAM Jury Studio UI built; 5/5 pytest passed"),
            ("Source Repository", REPO_URL),
            ("Completion Date", REPORT_DATE),
        ]
        add_table(doc, ["AREA", "VERIFIED DETAIL"], metadata_rows, [2200, 7200], header_fill=PALE_GOLD, font_size=9.2)

        add_callout(
            doc,
            "Core Scientific Contribution",
            "Rather than assuming synthetic data always improves performance, this study isolates generative diversity from class-frequency repetition under strict lesion-group splits, proves why naive 128px DCGAN artifacts degraded minority-class F1 (0.308 vs. 0.354 baseline), and packages the complete audit into an interactive Grad-CAM Jury application with a concrete GenAI v2 roadmap.",
            fill=PALE_TEAL,
            accent=TEAL,
        )

        exec_heading = add_section_title(doc, "Executive Summary")
        exec_heading.paragraph_format.page_break_before = True
        add_body(
            doc,
            "Squamous cell carcinoma (SCC) is an underrepresented malignancy in dermoscopic screening benchmarks. In the ISIC 2019 dataset, melanocytic nevi (NV) outnumber SCC by roughly 22:1 in the training partition (9,454 vs. 429 images). This project investigates whether a conditional generative adversarial network (cGAN) can synthesize sufficiently diverse SCC training examples to improve real-world SCC detection without introducing clinical shortcuts.",
        )
        add_body(
            doc,
            "To answer this rigorously, we engineered a leakage-safe 3-arm comparison using identical EfficientNet-B0 architectures, class-balanced sampling weights, augmentations, and validation selection rules across 3,664 held-out real test lesions. We further built a production-style FastAPI web studio featuring live multi-arm inference, spatial Grad-CAM overlays, an interactive 24-panel blinded realism quiz, a live conditional sampler, and a structured roadmap for extending this methodology to MRI, CT, X-ray, and digital pathology.",
        )

        base_m = cache["arms"]["real_only"]["metrics"]
        rep_m = cache["arms"]["repeated_real_scc"]["metrics"]
        cgan_m = cache["arms"]["cgan_scc"]["metrics"]
        audit = cache["cgan_audit"]

        add_callout(
            doc,
            "Key Empirical Verdict",
            f"The Real-Only baseline achieved the highest held-out SCC F1 ({base_m['target_f1']:.3f}, 95% CI [{base_m['target_f1_ci_low']:.3f}, {base_m['target_f1_ci_high']:.3f}]) and PR-AUC ({base_m['target_pr_auc']:.3f}), outperforming both Repeated-Real SCC ({rep_m['target_f1']:.3f} F1) and cGAN-SCC ({cgan_m['target_f1']:.3f} F1). Post-hoc auditing traced this gap to 80-epoch discriminator saturation (~2.000 hinge loss), 4x4 ConvTranspose2d checkerboard artifacts, and a {audit['synthetic_detector_auc']:.3f} real-vs-synthetic detector AUC.",
            fill=PALE_GOLD,
            accent=GOLD,
        )

        add_section_title(doc, "Project At a Glance", level=2)
        add_table(
            doc,
            ["DIMENSION", "IMPLEMENTATION & VERIFIED EVIDENCE"],
            [
                ("Problem Solved", "Severe 22:1 dermoscopic class imbalance (429 real SCC vs. 9,454 NV) and opaque black-box augmentation claims."),
                ("Experimental Design", "3 controlled EfficientNet-B0 arms: Real Only (0 extra), Repeated Real (+600 resampled SCC), cGAN SCC (+600 screened fakes)."),
                ("Leakage Safeguards", "StratifiedGroupKFold grouped by lesion_id (seed 146); 0 shared lesions and 0 synthetic images in val/test."),
                ("Quality Screening", "1,200 cGAN candidates screened via perceptual hashing (963 accepted at pHash > 4) + 12-feature domain-shift AUC audit."),
                ("Interactive UI", "FastAPI + zero-CDN HTML/CSS/JS studio with live Grad-CAM overlays, 24-panel quiz, Epoch 1-80 scrubber, and report exporter."),
                ("Future Vision", "cGAN v2 (Upsample+Conv + DiffAugment + R1 + Diffusion) and MONAI/DICOM expansion across MRI, CT, X-ray, and WSI."),
            ],
            [2150, 7250],
            header_fill=PALE_BLUE,
            font_size=9.2,
        )

        # ---------------- SECTION 1 ----------------
        add_section_title(doc, "1. Problem Statement & Leakage-Safe Data Engineering")
        add_body(
            doc,
            "Standard accuracy is unreliable on multi-class medical datasets dominated by benign nevi: a classifier can exceed 70% overall accuracy while missing the majority of high-risk squamous cell carcinoma cases. Furthermore, multi-image dermoscopy collections frequently include multiple photographs of the same physical lesion. Random image-level splitting allows near-identical photographs of one lesion to appear in both training and test splits, artificially inflating scores.",
        )
        add_bullet(doc, "Enforced StratifiedGroupKFold grouped by lesion_id (falling back to image ID only when lesion_id is missing), yielding zero lesion overlap across train, validation, and test partitions.", bold_lead="Group-Safe Partitioning: ")
        add_bullet(doc, "Retained all 8 active ISIC 2019 diagnostic categories (MEL, NV, BCC, AK, BKL, DF, VASC, SCC; UNK had 0 training instances) totaling 25,331 real images across 13,931 unique lesion groups.", bold_lead="Diagnostic Coverage: ")
        add_bullet(doc, "Trained the conditional GAN strictly on the 18,623-image training partition; validation (3,044) and test (3,664) partitions remained 100% real lesions.", bold_lead="Strict Partition Isolation: ")

        split_rows = []
        for diag in cache["class_names"]:
            tr_c = cache["splits"]["train"]["counts"][diag]
            va_c = cache["splits"]["val"]["counts"][diag]
            te_c = cache["splits"]["test"]["counts"][diag]
            split_rows.append((
                diag,
                cache["diagnosis_full_names"][diag].replace(" (Target)", ""),
                f"{tr_c:,}",
                f"{va_c:,}",
                f"{te_c:,}",
                f"{tr_c + va_c + te_c:,}",
            ))
        split_rows.append((
            "TOTAL",
            "All Active Diagnoses (Lesion Groups: 10,238 / 1,704 / 1,989)",
            f"{cache['splits']['train']['images']:,}",
            f"{cache['splits']['val']['images']:,}",
            f"{cache['splits']['test']['images']:,}",
            "25,331",
        ))
        add_table(
            doc,
            ["CODE", "CLINICAL DIAGNOSIS", "TRAIN", "VAL", "TEST", "TOTAL"],
            split_rows,
            [900, 4100, 1100, 1050, 1050, 1200],
            header_fill=PALE_BLUE,
            font_size=8.9,
        )
        add_figure(doc, figs["imbalance"], "Figure 1: Training-set class distribution across the 18,623 real ISIC 2019 images, highlighting the 22:1 ratio between NV (9,454) and target SCC (429).", width_inches=6.2)

        # ---------------- SECTION 2 ----------------
        add_section_title(doc, "2. Methodology: cGAN Architecture & Controlled 3-Arm Design")
        add_body(
            doc,
            "To separate genuine generative image diversity from mere oversampling, all three training arms share the exact same WeightedRandomSampler class-balancing scheme, ImageNet-initialized EfficientNet-B0 backbone (224x224 input), AdamW optimizer (lr=2e-4, weight_decay=1e-4), and 12-epoch training schedule selected by validation Macro-F1.",
        )
        add_number(doc, "Conditional DCGAN Generator (4,123,360 parameters): Concatenates a 128-dim Gaussian latent vector with a 32-dim class embedding and upsamples through 5 ConvTranspose2d layers to 128x128 RGB.", step_num_id, bold_lead="Conditional DCGAN Generator (4,123,360 parameters): ")
        add_number(doc, "Projection Discriminator (2,792,929 parameters): Uses 5 spectral-normalized 4x4 strided convolutions, global average pooling, and inner-product class projection trained with hinge loss for 80 epochs (batch size 8, lr=2e-4, betas=(0.0, 0.9)).", step_num_id, bold_lead="Projection Discriminator (2,792,929 parameters): ")
        add_number(doc, "Perceptual-Hash Deduplication: Generated 1,200 SCC candidates and rejected 237 samples within Hamming distance <= 4 of any real training SCC or previously accepted synthetic image, yielding 963 accepted candidates (first 600 used in Arm 3).", step_num_id, bold_lead="Perceptual-Hash Deduplication: ")
        add_number(doc, "3-Arm Controlled Training: Arm 1 (real_only, 18,623 records), Arm 2 (repeated_real_scc, +600 resampled real SCC = 19,223 records), and Arm 3 (cgan_scc, +600 screened synthetic SCC = 19,223 records).", step_num_id, bold_lead="3-Arm Controlled Training: ")

        # ---------------- SECTION 3 ----------------
        add_section_title(doc, "3. Quantitative Evaluation on Held-Out Real Lesions")
        add_body(
            doc,
            "All three saved EfficientNet-B0 checkpoints were evaluated uniformly on the 3,664 held-out real test images (103 SCC positive, 1,989 independent lesion groups) with 1,000-round lesion-group bootstrap confidence intervals.",
        )

        comp_rows = []
        for arm in cache["arm_order"]:
            a = cache["arms"][arm]
            m = a["metrics"]
            cal = a["calibration"]
            comp_rows.append((
                a["label"],
                f"{m['target_recall']:.3f}",
                f"{m['target_precision']:.3f}",
                f"{m['target_f1']:.3f} [{m['target_f1_ci_low']:.3f}, {m['target_f1_ci_high']:.3f}]",
                f"{m['target_pr_auc']:.3f}",
                f"{m['macro_f1']:.3f}",
                f"{m['accuracy']:.3f}",
                f"{cal['mean_prob_true_scc']:.3f} / {cal['mean_prob_non_scc']:.3f}",
            ))
        add_table(
            doc,
            ["TRAINING ARM", "SCC REC", "SCC PREC", "SCC F1 [95% CI]", "PR-AUC", "MACRO-F1", "ACC", "MEAN P(SCC) +/-"],
            comp_rows,
            [1800, 820, 840, 1780, 780, 920, 700, 1760],
            header_fill=PALE_TEAL,
            font_size=8.3,
        )

        add_figure(
            doc,
            ARTIFACT_DIR / "jury_classifier_comparison.png",
            "Figure 2: Held-out test Precision-Recall curves (left) and target SCC F1-scores with 95% group-bootstrap confidence intervals (right) across the three controlled arms.",
            width_inches=6.4,
        )

        add_section_title(doc, "Per-Class Breakdown & Error Analysis", level=2)
        add_body(
            doc,
            "Across the 8 active ISIC categories, baseline real_only achieves strong performance on well-represented classes (NV F1=0.846, BCC F1=0.701, MEL F1=0.565, BKL F1=0.573) while struggling most on rare keratosis/carcinoma boundaries (AK F1=0.397, SCC F1=0.354, DF F1=0.230). Adding 600 cGAN fakes lowered SCC recall from 0.311 to 0.272 and lowered mean predicted SCC probability on true SCC cases from 0.304 to 0.263.",
        )

        per_rows = []
        real_pc = {r["diagnosis"]: r for r in cache["arms"]["real_only"]["per_class"]}
        rep_pc = {r["diagnosis"]: r for r in cache["arms"]["repeated_real_scc"]["per_class"]}
        cgan_pc = {r["diagnosis"]: r for r in cache["arms"]["cgan_scc"]["per_class"]}
        for diag in cache["class_names"]:
            r0, r1, r2 = real_pc[diag], rep_pc[diag], cgan_pc[diag]
            per_rows.append((
                diag,
                r0["full_name"].replace(" (Target)", ""),
                str(r0["support"]),
                f"{r0['precision']:.3f} / {r0['recall']:.3f} / {r0['f1']:.3f}",
                f"{r1['f1']:.3f}",
                f"{r2['f1']:.3f}",
                f"{r2['f1'] - r0['f1']:+.3f}",
            ))
        add_table(
            doc,
            ["CODE", "DIAGNOSIS", "SUPPORT", "REAL ONLY (P / R / F1)", "REPEATED F1", "CGAN F1", "DELTA (CGAN-REAL)"],
            per_rows,
            [760, 2240, 850, 2150, 1080, 980, 1340],
            header_fill=PALE_BLUE,
            font_size=8.4,
        )
        add_figure(
            doc,
            figs["cm_per_class"],
            "Figure 3: Row-normalized 8x8 confusion matrix for baseline real_only (left) and per-class F1 comparison between real_only and cgan_scc (right).",
            width_inches=6.4,
        )

        # ---------------- SECTION 4 ----------------
        add_section_title(doc, "4. Generative Model Audit: Why cGAN v1 Did Not Improve SCC")
        add_body(
            doc,
            "Because generative visual grids can appear plausible at a glance while introducing systematic frequency artifacts, we conducted four quantitative and visual audits on the 80-epoch cGAN checkpoint (artifacts/cgan/cgan_latest.pt):",
        )
        add_bullet(doc, "In artifacts/cgan/loss_history.csv, discriminator hinge loss jumped to 2.000 at Epoch 3 and stayed pinned near 2.0001 through Epoch 80, indicating weak discriminator gradient signal under small batch size (8) and spectral-normalized global average pooling.", bold_lead="Discriminator Saturation: ")
        add_bullet(doc, "Visual inspection of samples_epoch_080.png and synthetic candidates exposes prominent 4x4 grid checkerboards caused by ConvTranspose2d stride overlap, alongside black corner artifacts.", bold_lead="ConvTranspose2d Checkerboards: ")
        add_bullet(doc, f"Perceptual hashing rejected 237/1,200 candidates (mean real pHash distance {audit['mean_real_phash']:.2f}, mean synthetic distance {audit['mean_synthetic_phash']:.2f}), proving candidates are not pixel copies — yet pHash is blind to synthetic grid texture.", bold_lead="pHash Limitations: ")
        add_bullet(doc, f"Fitting a simple 12-feature color/edge LogisticRegression detector between 429 real SCC training images and 600 accepted synthetic SCC images yielded {audit['synthetic_detector_auc']:.3f} ROC-AUC, flagging strong synthetic domain shift. Because Arm 3 mixed 600 synthetic SCC images with 429 real SCC images under equal class-sampling, ~58% of SCC training draws carried grid artifacts.", bold_lead="0.928 Domain-Shift Detector AUC: ")

        add_figure(
            doc,
            figs["cgan_audit"],
            "Figure 4: 80-epoch cGAN hinge-loss trajectory showing D-loss saturation at 2.000 (left), real ISIC SCC lesion panel_01 (middle), and synthetic cGAN sample panel_02 exhibiting 4x4 checkerboard texture (right).",
            width_inches=6.4,
        )
        add_figure(
            doc,
            ARTIFACT_DIR / "cgan" / "samples_epoch_080.png",
            "Figure 5: Fixed-noise conditional cGAN output grid at Epoch 80 across ISIC diagnosis embeddings.",
            width_inches=5.4,
        )

        # ---------------- SECTION 5 ----------------
        add_section_title(doc, "5. Interactive Jury Studio UI & Explainability Platform")
        add_body(
            doc,
            "To make the project immediately demonstrable to technical and clinical juries, we built a FastAPI + zero-CDN modern web application (app/main.py and app/static/) backed by a precomputed evaluation cache (artifacts/jury_eval_cache.json) for sub-50ms metric rendering and live PyTorch MPS inference.",
        )
        add_table(
            doc,
            ["UI WORKSPACE", "KEY CAPABILITIES & JURY VALUE"],
            [
                ("1. Overview", "KPI dashboard (25,331 lesions, 0.354 vs. 0.308 SCC F1, 0.928 detector AUC), ISIC class-imbalance SVG chart, and 3-arm verdict."),
                ("2. Live Analyzer & Grad-CAM", "Drag-and-drop PNG/JPEG upload or 30-case curated test/synthetic gallery; runs all 3 EfficientNet-B0 arms simultaneously with 7x7->224x224 Grad-CAM overlays, layer switcher, and HTML/Markdown Preliminary Report export."),
                ("3. Interactive Jury Lab", "Precision-Recall curves, 95% bootstrap CI plots, 8x8 confusion matrices, per-class responsive table, Epoch 1-80 scrubber, live cGAN conditional sampler, and server-scored 24-panel Blinded Realism Quiz."),
                ("4. Future Roadmap Hub", "Interactive cGAN v1->v2 engineering diff, multi-modal MRI/CT/X-ray/Pathology workflow explorer, hardware planner, and Responsible AI checklist."),
            ],
            [2300, 7100],
            header_fill=PALE_TEAL,
            font_size=8.9,
        )

        ui_shot = ARTIFACT_DIR / "report_ui_showcase.png"
        if ui_shot.exists():
            add_figure(
                doc,
                ui_shot,
                "Figure 6: Live Analyzer & Grad-CAM Studio running all three trained EfficientNet-B0 arms simultaneously on held-out SCC test case ISIC_0055350.",
                width_inches=6.2,
            )

        # ---------------- SECTION 6 ----------------
        add_section_title(doc, "6. Technology Stack, Hardware Strategy & Verification")
        add_table(
            doc,
            ["LAYER", "TOOLS & LIBRARIES", "ROLE IN SYSTEM"],
            [
                ("Core ML & Vision", "Python 3.13, PyTorch 2.10 (MPS), torchvision EfficientNet-B0", "Group-safe training, 3-arm checkpoint inference, ConditionalGenerator, and Grad-CAM."),
                ("Evaluation & Audit", "scikit-learn, imagehash, pandas, numpy, matplotlib", "Bootstrap CIs, PR-AUC, confusion matrices, pHash screening, and 0.928 detector AUC."),
                ("Web Backend & API", "FastAPI, Uvicorn, Pydantic, python-multipart", "REST endpoints, path-safe image streaming, server-side quiz scoring, XSS-safe report export."),
                ("Frontend UI", "Semantic HTML5, Modern CSS (light-dark), Vanilla JS + SVG", "Zero-CDN accessible dark/light interface with native <dialog> and container-query tables."),
                ("QA & Verification", "pytest, FastAPI TestClient, Headless Browser Audit", "Exact parity verification (diff=0.00000), 5/5 backend/ML tests, and 50/50 HTTP 200 UI checks."),
            ],
            [1850, 3250, 4300],
            header_fill=PALE_BLUE,
            font_size=8.8,
        )

        add_section_title(doc, "Hardware Scaling & Software Verification Evidence", level=2)
        add_table(
            doc,
            ["ENVIRONMENT / CHECK", "SPECIFICATION / OBSERVED RESULT"],
            [
                ("Current Local Compute", "Apple Silicon MPS (5.33 GB recommended working set, 68% memory cap); batch size 8; live Grad-CAM inference < 0.3s/image."),
                ("Next-Stage Training Hardware", "NVIDIA RTX (16-24 GB VRAM CUDA) or Cloud A100/L4 for 256px DiffAugment cGAN v2, Diffusion LoRA, and >=3-seed sweeps."),
                ("Institutional Deployment", "On-premise HIPAA/GDPR-compliant cluster with Docker, PACS/DICOM gateway, and federated learning for multi-site validation."),
                ("Cache Parity Check", "python3 -m scripts.build_eval_cache completed in 244.7s; matched classifier_comparison.csv across all 3 arms with 0.00000 F1 discrepancy."),
                ("Automated Integration Suite", "python3 -m pytest tests/test_app.py -v: 5 passed in 16.65s (cache parity, endpoints, PRIVATE_answer_key.csv 404 isolation, Grad-CAM, cGAN sampler)."),
                ("Live Browser Verification", "Verified all 4 views at 1440x900 and 879x804 viewports with 50/50 HTTP 200 responses, zero console errors, and working theme/quiz/report dialogs."),
            ],
            [2400, 7000],
            header_fill=PALE_GRAY,
            font_size=8.8,
        )

        # ---------------- SECTION 7 ----------------
        add_section_title(doc, "7. Future Roadmap: GenAI v2 & Multi-Modal Healthcare Imaging")
        add_body(
            doc,
            "Drawing directly from the Topic 146 experimental audit and our broader GenAI Jury methodology, future development proceeds across two concrete tracks: resolving the generative domain shift in cGAN v2 and expanding the explainable pipeline to volumetric and radiographic modalities.",
        )

        add_section_title(doc, "Track A: cGAN v1 -> GenAI v2 Architecture Upgrades", level=2)
        add_table(
            doc,
            ["DIMENSION", "V1 (CURRENT TRAINED)", "V2 (PROPOSED UPGRADE)", "EXPECTED BENEFIT"],
            [
                ("Upsampling", "ConvTranspose2d (4x4, stride 2)", "Bilinear Upsample + 3x3 Conv2d + GroupNorm", "Eliminates 4x4 checkerboard grid artifacts."),
                ("Discriminator", "Global Avg Pool + spectral_norm", "Multi-scale PatchGAN / ResNet D + DiffAugment + R1", "Breaks D-loss saturation under small N=429."),
                ("Resolution & Model", "128px DCGAN", "256px StyleGAN2-ADA or Latent Diffusion + EMA", "Preserves fine dermoscopic vascular/pigment cues."),
                ("Validation Gate", "Single seed (146), pHash > 4", ">=3 seeds + FID/KID + detector AUC <= 0.65 gate", "Prevents harmful synthetic data from entering training."),
            ],
            [1400, 2400, 2900, 2700],
            header_fill=PALE_TEAL,
            font_size=8.5,
        )

        add_section_title(doc, "Track B: Multi-Modal Healthcare Expansion & 9-Phase Methodology", level=2)
        add_table(
            doc,
            ["MODALITY", "FORMATS & PREPROCESSING", "TARGET MODEL & EXPLAINABILITY"],
            [
                ("Dermoscopy (Active)", "JPEG/PNG (ISIC), 224px crop, lesion_id splits", "EfficientNet-B0 + Grad-CAM + Preliminary Technical Report."),
                ("Chest / MSK X-Ray", "DICOM (.dcm) / 16-bit PNG, pydicom PHI scrub, CLAHE", "DenseNet-121 / ConvNeXt multi-label + Grad-CAM++ ROI + ECE."),
                ("Brain & Body MRI", "NIfTI (.nii.gz) / DICOM, MONAI/SimpleITK bias-field norm", "3D Swin-UNETR / SegResNet + volumetric Dice & uncertainty maps."),
                ("Thoracic / Abd. CT", "DICOM CT volume, Hounsfield multi-windowing, 3D tiling", "3D nnU-Net nodule detector + axial/coronal/sagittal ROI cards."),
                ("Digital Pathology", "Whole-Slide (SVS/TIFF), Macenko stain normalization", "Multiple Instance Learning (MIL) ViT + federated learning pilot."),
            ],
            [1900, 3600, 3900],
            header_fill=PALE_BLUE,
            font_size=8.6,
        )

        add_number(doc, "Discover & Specify: Define clinical/research objective, minority target KPIs, and group-safe splitting rules.", impl_num_id, bold_lead="1-2. Discover & Specify: ")
        add_number(doc, "Prepare & Design: De-identify images, prevent lesion leakage, and design controlled baseline/repetition/synthetic arms.", impl_num_id, bold_lead="3-4. Prepare & Design: ")
        add_number(doc, "Train & Validate: Train under reproducible seeds and evaluate strictly on held-out real cohorts with bootstrap CIs & domain-shift detectors.", impl_num_id, bold_lead="5-6. Train & Validate: ")
        add_number(doc, "Explain, Implement & Improve: Deliver spatial Grad-CAM studio + preliminary reports, then iterate to GenAI v2 and external multi-site clinical trials.", impl_num_id, bold_lead="7-9. Explain, Implement & Improve: ")

        # ---------------- SECTION 8 ----------------
        add_section_title(doc, "8. Responsible AI, Privacy, Limitations & Conclusion")
        add_table(
            doc,
            ["GOVERNANCE PILLAR", "STATUS", "CONTROL & TECHNICAL SAFEGUARD"],
            [
                ("Zero Data Leakage", "Verified Active", "StratifiedGroupKFold by lesion_id keeps all photographs of a lesion in one partition."),
                ("Clean Evaluation Sets", "Verified Active", "Validation (3,044) and Test (3,664) splits contain strictly 100% real clinical images."),
                ("Shortcut & Artifact Audit", "Verified Active", "Simple-stats detector (0.928 AUC) & pHash flagged cGAN v1 domain shift rather than hiding it."),
                ("Explainability & Uncertainty", "Active in UI", "Grad-CAM heatmaps + multi-arm disagreement + probability calibration on every inference."),
                ("Survey & Secret Isolation", "Verified Active", "PRIVATE_answer_key.csv blocked from static serving (HTTP 404) and scored server-side only."),
                ("Demographic & Site Fairness", "Planned v2", "Subgroup error auditing across age_approx, sex, anatom_site_general, and external hospitals."),
            ],
            [2250, 1450, 5700],
            header_fill=PALE_GOLD,
            font_size=8.8,
        )

        add_callout(
            doc,
            "Mandatory Clinical Limitation",
            "GenAI Jury Studio is a research prototype, NOT a cleared medical device. Neither visual realism surveys nor perceptual hashes establish clinical validity. Outputs must never drive independent diagnosis or treatment decisions without external multi-center validation, regulatory clearance, and qualified medical professional review.",
            fill=PALE_RED,
            accent=RED,
        )

        add_section_title(doc, "Conclusion")
        add_body(
            doc,
            "This project demonstrates a complete, disciplined lifecycle for generative AI in healthcare imaging: from rigorous group-safe splitting and controlled experimentation on 25,331 real ISIC 2019 lesions, to honest negative-control auditing of cGAN domain shift, to a responsive FastAPI + Grad-CAM presentation platform and actionable GenAI v2 roadmap. By pairing technical transparency with explainable visual tooling and strict safety guardrails, the platform provides a reproducible foundation for future multi-modal medical-imaging research.",
        )

    props = doc.core_properties
    props.title = "GenAI Jury — Topic 146 Final Project Report"
    props.subject = "Controlled cGAN Skin-Lesion Augmentation, Grad-CAM Studio & Healthcare Roadmap"
    props.author = "Ameersohiel N Momin, Avneet Bhatt, Aditi Gaikwad, Dewang Walke"
    props.keywords = "GenAI, cGAN, ISIC 2019, SCC, Grad-CAM, EfficientNet-B0, Medical Imaging"
    doc.save(OUT)
    print(f"Successfully wrote complete report: {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
