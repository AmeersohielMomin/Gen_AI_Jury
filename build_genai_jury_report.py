from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_LINE_SPACING


OUT = "GenAI_Jury_Professional_Report.docx"

# Resolved narrative_proposal design tokens with a restrained healthcare/technology override.
NAVY = "0B2545"
TEAL = "0F766E"
GOLD = "C58B2A"
INK = "172B4D"
MUTED = "5B677A"
PALE_BLUE = "EAF2F8"
PALE_TEAL = "E8F5F3"
PALE_GOLD = "FBF4E6"
PALE_GRAY = "F4F6F8"
WHITE = "FFFFFF"
RED = "9B1C1C"


def set_font(run, name="Aptos", size=None, color=None, bold=None, italic=None):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
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


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
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


def set_table_geometry(table, widths, indent=120):
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
    abstract_ids = [int(x.get(qn("w:abstractNumId"))) for x in numbering.findall(qn("w:abstractNum")) if x.get(qn("w:abstractNumId"))]
    num_ids = [int(x.get(qn("w:numId"))) for x in numbering.findall(qn("w:num")) if x.get(qn("w:numId"))]
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
        set_font(r, bold=True, color=INK)
        r2 = p.add_run(text[len(bold_lead):])
        set_font(r2, color=INK)
    else:
        r = p.add_run(text)
        set_font(r, color=INK)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.375)
    p.paragraph_format.first_line_indent = Inches(-0.194)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.208
    r = p.add_run(text)
    set_font(r, size=10.5, color=INK)
    return p


def add_number(doc, text, num_id=None):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.left_indent = Inches(0.375)
    p.paragraph_format.first_line_indent = Inches(-0.194)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.208
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
    r = p.add_run(text)
    set_font(r, size=10.5, color=INK)
    return p


def add_callout(doc, label, text, fill=PALE_TEAL, accent=TEAL):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360], indent=120)
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    set_cell_border(cell, color=accent, sz="10")
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(label.upper() + "  ")
    set_font(r, size=9, color=accent, bold=True)
    r2 = p.add_run(text)
    set_font(r2, size=10.5, color=INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_section_title(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    r = p.add_run(text)
    set_font(r, size={1:16, 2:13, 3:12}[level], color=TEAL if level == 1 else NAVY, bold=True)
    set_keep_with_next(p)
    return p


def add_table(doc, headers, rows, widths, header_fill=PALE_BLUE, font_size=9.4):
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths, indent=120)
    header = table.rows[0]
    repeat_header(header)
    for i, text in enumerate(headers):
        cell = header.cells[i]
        set_cell_shading(cell, header_fill)
        set_cell_border(cell, color="B8C9D9", sz="8")
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(text)
        set_font(r, size=9.2, color=NAVY, bold=True)
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, text in enumerate(row):
            set_cell_border(cells[i], color="D7DEE8", sz="6")
            if ridx % 2 == 1:
                set_cell_shading(cells[i], PALE_GRAY)
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            r = p.add_run(text)
            set_font(r, size=font_size, color=INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def setup_document():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(0.82)
    sec.bottom_margin = Inches(0.78)
    sec.left_margin = Inches(1.0)
    sec.right_margin = Inches(1.0)
    sec.header_distance = Inches(0.35)
    sec.footer_distance = Inches(0.35)
    sec.different_first_page_header_footer = True

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.333
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    for name, size, color, before, after in [
        ("Heading 1", 16, TEAL, 18, 10),
        ("Heading 2", 13, NAVY, 12, 6),
        ("Heading 3", 12, NAVY, 8, 4),
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

    # Header/footer on subsequent pages.
    header = sec.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    hp.paragraph_format.space_after = Pt(0)
    r = hp.add_run("GENAI JURY")
    set_font(r, size=8.5, color=TEAL, bold=True)
    r = hp.add_run("   |   PROFESSIONAL PROJECT REPORT")
    set_font(r, size=8.5, color=MUTED)
    footer = sec.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    fp.paragraph_format.space_before = Pt(0)
    r = fp.add_run("CONFIDENTIAL • RESEARCH AND DEVELOPMENT   |   ")
    set_font(r, size=8.5, color=MUTED)
    add_page_number(fp)
    return doc


def main():
    doc = setup_document()
    solution_num_id = create_numbering(doc)
    implementation_num_id = create_numbering(doc)

    # Cover page: editorial cover pattern with restrained technology palette.
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(55)
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run("GENAI JURY")
    set_font(r, size=12, color=GOLD, bold=True)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("From Idea to Intelligent Solution")
    set_font(r, name="Aptos Display", size=29, color=NAVY, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(22)
    r = p.add_run("A professional methodology for developing responsible GenAI and medical-imaging solutions")
    set_font(r, size=14, color=TEAL)

    rule = doc.add_paragraph()
    rule.paragraph_format.space_after = Pt(22)
    add_rule(rule, color=GOLD, size="22")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(9)
    r = p.add_run("PROJECT REPORT")
    set_font(r, size=10, color=MUTED, bold=True)

    metadata = [
        ("Project focus", "Idea evaluation, methodology, and implementation planning"),
        ("Future application", "MRI, CT, X-ray, and other medical-imaging workflows"),
        ("Current status", "Research, prototyping, and model training in progress"),
        ("Prepared for", "GenAI Jury evaluation and technical presentation"),
        ("Date", "2 September 2026"),
    ]
    add_table(doc, ["AREA", "DETAIL"], metadata, [1900, 7460], header_fill=PALE_GOLD, font_size=9.5)
    add_callout(doc, "Core proposition", "GenAI Jury explains the complete journey from a problem statement to a tested, implementable AI solution - including data, methodology, tools, hardware, output, validation, and responsible deployment.", fill=PALE_TEAL, accent=TEAL)

    doc.add_page_break()

    add_section_title(doc, "Executive Summary")
    add_body(doc, "GenAI Jury is an advanced idea-to-solution framework designed to evaluate an idea, clarify the problem it addresses, recommend an appropriate artificial-intelligence approach, and describe how the solution can be developed and implemented. The platform is intended to go beyond idea generation by presenting a complete and credible project journey for review by a jury, technical team, or organizational decision-maker.")
    add_body(doc, "The long-term vision is to apply this methodology to healthcare imaging, including MRI scans, CT scans, X-rays, and related medical-image workflows. In this context, GenAI Jury may support image classification, abnormality detection, segmentation, visual explanation, and preliminary report generation. The system will be designed as a decision-support tool and will require clinical, regulatory, privacy, and safety validation before any real-world use.")
    add_callout(doc, "Current position", "Training and experimentation are in progress. Early model results should be treated as research findings, not as clinical evidence or a final diagnostic capability.", fill=PALE_GOLD, accent=GOLD)

    add_section_title(doc, "Project At a Glance", level=2)
    add_table(doc, ["COMPONENT", "PROPOSED DIRECTION"], [
        ("Problem", "Ideas are often presented without a complete route to implementation, validation, and measurable outcomes."),
        ("Solution", "A structured GenAI workflow that transforms an idea into a documented, testable, and implementable solution."),
        ("Primary users", "Students, researchers, innovators, technical teams, healthcare stakeholders, and project evaluators."),
        ("Initial domain", "General idea evaluation and project planning."),
        ("Future domain", "AI-assisted analysis of MRI, CT, X-ray, and other medical images."),
        ("Success criteria", "Technical feasibility, useful outputs, explainability, privacy, reliability, and responsible implementation."),
    ], [2100, 7260], header_fill=PALE_BLUE, font_size=9.4)

    add_section_title(doc, "1. Problem Being Solved")
    add_body(doc, "Innovative ideas frequently remain at the concept stage because the problem, data requirements, technical pathway, expected output, and implementation plan are not connected into one coherent story. This makes it difficult for a jury or stakeholder to understand whether the idea is practical, how it will be built, and how success will be measured.")
    add_bullet(doc, "Lack of a consistent method for moving from an idea to an executable project.")
    add_bullet(doc, "Difficulty selecting suitable datasets, AI models, tools, and hardware.")
    add_bullet(doc, "Unclear connection between input data, model processing, output, and user action.")
    add_bullet(doc, "Limited visibility into testing, risks, privacy, explainability, and deployment requirements.")
    add_bullet(doc, "In medical imaging, manual review can be time-intensive and may benefit from carefully validated decision support.")

    add_section_title(doc, "2. Proposed GenAI Solution")
    add_body(doc, "GenAI Jury will accept a problem statement, project idea, dataset description, or set of user requirements. It will organize this information into a complete solution blueprint. The output will explain what is being solved, why the solution is needed, how the AI system will work, what methodology will be followed, and how the final system can be implemented.")
    add_number(doc, "Understand the idea and identify the underlying problem.", solution_num_id)
    add_number(doc, "Define the target users, data, constraints, and desired result.", solution_num_id)
    add_number(doc, "Recommend an AI approach, technical stack, and development environment.", solution_num_id)
    add_number(doc, "Describe data preparation, training, testing, and evaluation steps.", solution_num_id)
    add_number(doc, "Generate an implementation plan with risks, safeguards, and future improvements.", solution_num_id)

    add_section_title(doc, "3. End-to-End Methodology")
    add_body(doc, "The project will follow a staged methodology so that each phase produces a clear output for the next phase. This creates traceability from the original idea to the final result.")
    add_table(doc, ["PHASE", "ACTIVITY", "KEY OUTPUT"], [
        ("1. Discover", "Define the problem, users, context, and expected benefit.", "Approved problem statement and project objective."),
        ("2. Specify", "Document requirements, constraints, input formats, and success criteria.", "Functional and technical requirements."),
        ("3. Prepare", "Collect, de-identify, clean, label, normalize, and split the data.", "Training-ready dataset and data documentation."),
        ("4. Design", "Select model architecture, preprocessing pipeline, and system components.", "Model and solution architecture."),
        ("5. Train", "Train the model, monitor learning, tune parameters, and save checkpoints.", "Trained prototype model."),
        ("6. Validate", "Test unseen data and assess performance, reliability, and limitations.", "Evaluation report and error analysis."),
        ("7. Explain", "Provide visual or textual explanations and confidence information.", "Interpretable result presentation."),
        ("8. Implement", "Package the model into an application, API, or local workflow.", "Demonstrable solution prototype."),
        ("9. Improve", "Use feedback and evidence to refine data, model, usability, and safeguards.", "Next-release roadmap."),
    ], [1350, 4070, 3940], header_fill=PALE_TEAL, font_size=9.0)

    add_section_title(doc, "4. System Inputs")
    add_body(doc, "The quality of the final result depends on the quality and clarity of the input. GenAI Jury will accept both project-level inputs and, for medical-imaging use cases, image-level inputs.")
    add_table(doc, ["INPUT CATEGORY", "EXAMPLES"], [
        ("Project input", "Problem statement, idea description, target users, constraints, and desired outcome."),
        ("Image input", "MRI, CT, or X-ray images in approved formats such as DICOM, PNG, or JPEG."),
        ("Dataset input", "Images, labels, class definitions, metadata, and dataset documentation."),
        ("Operational input", "Required processing time, deployment environment, user workflow, and output format."),
        ("Safety input", "Privacy rules, access controls, human-review requirements, and known limitations."),
    ], [2200, 7160], header_fill=PALE_BLUE, font_size=9.3)

    add_section_title(doc, "5. Expected Outputs")
    add_body(doc, "The system will produce a structured result that is understandable to both technical and non-technical reviewers. The exact output will depend on the selected use case.")
    add_bullet(doc, "A clearly written problem definition and solution objective.")
    add_bullet(doc, "Recommended methodology, tools, technologies, and hardware environment.")
    add_bullet(doc, "Model prediction, classification, segmentation, or detection result.")
    add_bullet(doc, "Confidence information and, where appropriate, a visual heatmap or region of interest.")
    add_bullet(doc, "Performance metrics, validation findings, limitations, and error analysis.")
    add_bullet(doc, "Implementation plan, responsible-AI safeguards, and future development roadmap.")
    add_callout(doc, "Example medical-imaging result", "The system may identify a possible region of interest in an X-ray, provide a confidence score, and generate a preliminary technical summary for review. The result must be verified by a qualified medical professional before clinical use.", fill=PALE_GOLD, accent=GOLD)

    add_section_title(doc, "6. Tools and Technologies")
    add_body(doc, "The technology stack will be selected according to the task, dataset size, privacy requirements, and available hardware. The following tools are suitable candidates for the prototype and future expansion.")
    add_table(doc, ["AREA", "TOOLS / TECHNOLOGIES", "PURPOSE"], [
        ("Programming", "Python", "Model development, data processing, experimentation, and automation."),
        ("Deep learning", "PyTorch or TensorFlow", "Training, evaluation, and deployment of AI models."),
        ("Medical imaging", "MONAI, pydicom, SimpleITK, NiBabel", "Medical-image loading, preprocessing, transformation, and analysis."),
        ("Computer vision", "OpenCV, NumPy, Pandas", "Image operations, numerical processing, and dataset management."),
        ("Visualization", "Matplotlib, Seaborn, heatmaps", "Performance analysis and explanation of model behaviour."),
        ("Development", "Jupyter, Git, GitHub", "Experiment tracking, version control, collaboration, and reproducibility."),
        ("Application", "FastAPI, Flask, Streamlit, or React", "API, dashboard, or user-facing prototype."),
        ("Acceleration", "NVIDIA CUDA and cuDNN", "GPU-accelerated model training and inference where supported."),
        ("Deployment", "Docker and secure local/cloud infrastructure", "Repeatable packaging and controlled deployment."),
    ], [1650, 3300, 4410], header_fill=PALE_BLUE, font_size=8.9)

    add_section_title(doc, "7. Hardware and Training Strategy")
    add_body(doc, "Training requirements will increase as the project moves from a small prototype to high-resolution, multi-modal medical imaging. Hardware decisions will therefore be made progressively rather than all at once.")
    add_table(doc, ["OPTION", "BEST USE", "ADVANTAGES", "LIMITATIONS"], [
        ("Current laptop", "Coding, data preparation, small experiments, and lightweight inference.", "No additional setup; useful for rapid iteration.", "CPU training may be slow; memory and storage may limit dataset size."),
        ("Laptop/workstation with NVIDIA GPU", "Regular model training and local experiments.", "CUDA cores can accelerate training and inference.", "Requires compatible GPU memory, drivers, CUDA, and thermal capacity."),
        ("Google Colab or approved cloud GPU", "Larger training runs and experiments beyond local capacity.", "Flexible access to GPU resources and easier scaling.", "Requires stable connectivity; sensitive patient data requires strict authorization and security controls."),
        ("Institutional or hospital infrastructure", "Validated research and controlled deployment.", "Better governance, security, and integration potential.", "Requires approvals, IT support, compliance, and operational planning."),
    ], [1750, 2550, 2500, 2560], header_fill=PALE_TEAL, font_size=8.65)
    add_callout(doc, "Recommended approach", "Use the current laptop for development and preprocessing, use CUDA acceleration when a compatible NVIDIA GPU is available, and use Google Colab or an approved GPU environment for larger experiments. Keep the same version-controlled code and document every training run.", fill=PALE_TEAL, accent=TEAL)

    add_section_title(doc, "8. Medical-Imaging Application Roadmap")
    add_table(doc, ["STAGE", "FOCUS", "EXPECTED MILESTONE"], [
        ("Stage 1", "Single-task prototype using a curated public dataset.", "Baseline classifier or segmentation model with documented metrics."),
        ("Stage 2", "Improved preprocessing, augmentation, and explainability.", "More stable performance and interpretable visual output."),
        ("Stage 3", "Multiple image types or conditions.", "Expanded evaluation across MRI, CT, or X-ray workflows."),
        ("Stage 4", "User interface and structured preliminary reporting.", "Demonstrable end-to-end prototype for expert review."),
        ("Stage 5", "External validation, governance, and deployment planning.", "Evidence-based readiness assessment; not clinical approval by itself."),
    ], [1350, 4150, 3860], header_fill=PALE_GOLD, font_size=9.0)

    add_section_title(doc, "9. Testing, Evaluation, and Quality Assurance")
    add_body(doc, "Testing will be performed on data that was not used during training. The evaluation will consider both numerical performance and practical usefulness. Medical-imaging models must be assessed carefully because a high overall accuracy can hide errors in important patient groups or uncommon conditions.")
    add_bullet(doc, "Classification: accuracy, precision, recall, F1-score, sensitivity, specificity, and area under the curve.")
    add_bullet(doc, "Segmentation: Dice score, intersection over union, boundary quality, and visual review.")
    add_bullet(doc, "Reliability: confidence calibration, repeatability, image-quality sensitivity, and failure cases.")
    add_bullet(doc, "Generalization: performance across different devices, sites, populations, and acquisition conditions.")
    add_bullet(doc, "Usability: clarity of the output, processing time, workflow fit, and human-review requirements.")

    add_section_title(doc, "10. Responsible AI, Privacy, and Safety")
    add_body(doc, "Because the future application involves medical images, responsible development is a core project requirement. The system must protect patient privacy and make its limitations clear.")
    add_table(doc, ["RISK AREA", "CONTROL MEASURE"], [
        ("Privacy", "Use de-identified data, access control, secure storage, and approved data-handling procedures."),
        ("Bias", "Evaluate performance across relevant patient groups and investigate unequal error rates."),
        ("Explainability", "Provide confidence information and visual or textual evidence where appropriate."),
        ("Human oversight", "Require qualified professional review before clinical interpretation or action."),
        ("Security", "Protect model endpoints, datasets, credentials, logs, and exported reports."),
        ("Transparency", "Document training data, model scope, known failure modes, and intended use."),
        ("Regulatory readiness", "Treat deployment as a separate stage requiring institutional and regulatory assessment."),
    ], [2200, 7160], header_fill=PALE_GOLD, font_size=9.2)
    add_callout(doc, "Important limitation", "A research prototype is not a medical device and should not be used to make independent diagnosis or treatment decisions. Clinical use would require appropriate validation, governance, approvals, and professional oversight.", fill="FDECEC", accent=RED)

    add_section_title(doc, "11. Implementation Plan")
    add_number(doc, "Finalize the use case, target condition, intended users, and success criteria.", implementation_num_id)
    add_number(doc, "Prepare and document a legally and ethically approved dataset.", implementation_num_id)
    add_number(doc, "Build a reproducible preprocessing and training pipeline.", implementation_num_id)
    add_number(doc, "Train a baseline model and record all experiments, datasets, and settings.", implementation_num_id)
    add_number(doc, "Evaluate performance, explain errors, and improve the model iteratively.", implementation_num_id)
    add_number(doc, "Create a secure interface or API for demonstration and expert review.", implementation_num_id)
    add_number(doc, "Conduct usability, safety, privacy, and deployment-readiness reviews.", implementation_num_id)
    add_number(doc, "Define the next phase based on evidence rather than assumptions.", implementation_num_id)

    add_section_title(doc, "12. Future Goals")
    add_bullet(doc, "Support multiple modalities, including MRI, CT, X-ray, ultrasound, and pathology images where appropriate.")
    add_bullet(doc, "Combine image analysis with structured clinical information under approved governance.")
    add_bullet(doc, "Generate clearer, multilingual, and structured preliminary reports for expert review.")
    add_bullet(doc, "Explore privacy-preserving approaches such as federated learning where feasible.")
    add_bullet(doc, "Improve robustness across scanners, hospitals, image qualities, and patient populations.")
    add_bullet(doc, "Develop secure offline or institution-controlled deployment for sensitive environments.")

    add_section_title(doc, "Conclusion")
    add_body(doc, "GenAI Jury provides a complete, professional framework for transforming an idea into a practical AI solution. Its value lies in connecting the problem, methodology, inputs, outputs, technologies, hardware, validation, and implementation into one traceable journey. The healthcare-imaging roadmap gives the project a meaningful long-term direction while preserving a responsible research approach during the current training phase.")
    add_body(doc, "The immediate objective is to complete a reliable prototype, document its results, and establish a strong foundation for future validation. With careful development, appropriate hardware, secure data practices, explainable outputs, and expert review, GenAI Jury can evolve into a powerful platform for responsible innovation in generative AI and medical imaging.")

    # Final metadata cleanup.
    props = doc.core_properties
    props.title = "GenAI Jury: From Idea to Intelligent Solution"
    props.subject = "Professional project report and medical-imaging roadmap"
    props.author = "GenAI Jury Project Team"
    props.keywords = "GenAI, medical imaging, MRI, CT, X-ray, AI methodology"
    doc.save(OUT)


if __name__ == "__main__":
    main()
