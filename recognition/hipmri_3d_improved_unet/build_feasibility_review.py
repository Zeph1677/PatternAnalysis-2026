"""Create the one-page COMP3710 Hard HipMRI 3D feasibility review."""

from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "evidence" / "COMP3710_HipMRI_Hard_Feasibility_Review.docx"


def _shade(cell, color: str) -> None:
    shade = OxmlElement("w:shd")
    shade.set(qn("w:fill"), color)
    cell._tc.get_or_add_tcPr().append(shade)


def _body(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(1.5)
    paragraph.paragraph_format.line_spacing = 1.0
    run = paragraph.add_run(text)
    run.font.size = Pt(9.2)


def _heading(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(1)
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(31, 78, 121)


def main() -> None:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.36)
    section.bottom_margin = Inches(0.36)
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)
    style = doc.styles["Normal"]
    style.font.name = "Aptos"
    style._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    style._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(1)
    run = title.add_run("HipMRI 3D Prostate Segmentation Feasibility Review")
    run.font.size = Pt(15)
    run.bold = True
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(4)
    subrun = subtitle.add_run("COMP3710 Hard difficulty 3D Improved U Net with residual units and deep supervision")
    subrun.italic = True
    subrun.font.size = Pt(9.5)

    table = doc.add_table(rows=2, cols=3)
    table.autofit = False
    cells = [("Dataset", "HipMRI 3D NIfTI volumes"), ("Difficulty", "Hard"), ("Target", "Prostate Dice at least 0.70"), ("Baseline", "Plain 3D U Net"), ("Proposed", "3D Improved U Net"), ("Split", "Patient disjoint; seed 42")]
    for cell, (label, value) in zip((cell for row in table.rows for cell in row.cells), cells):
        _shade(cell, "EAF2F8")
        paragraph = cell.paragraphs[0]
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(f"{label}\n{value}")
        run.font.size = Pt(8.8)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

    _heading(doc, "User need scope and decision requested")
    _body(doc, "A radiotherapy contouring specialist needs a reviewable 3D draft of prostate and organ-at-risk boundaries. The engineering question is whether volumetric context, residual units and deep supervision reduce apex, base, cross-slice discontinuity and bladder or rectum boundary errors enough to justify higher GPU memory and runtime. I request confirmation that this Hard HipMRI 3D scope and resource plan are appropriate before full training.")

    _heading(doc, "Method and acceptance criteria")
    _body(doc, "The comparison uses a plain 3D U-Net baseline and a 3D Improved U-Net with volumetric residual units and decoder deep supervision. Both use one saved patient-level train, validation and test manifest. Acceptance evidence is: zero patient overlap; aligned NIfTI image-mask volumes; held-out prostate Dice at least 0.70; per-class Dice/IoU, peak VRAM and per-patient inference time; and 3-5 real failure cases with false-positive and false-negative overlays.")

    _heading(doc, "Preliminary feasibility evidence")
    _body(doc, "Local synthetic-NIfTI tests currently verify recursive image-mask pairing, missing-pair rejection, 3D shape and affine validation, deterministic patient-disjoint splitting, finite 3D forward/backward passes, residual deep-supervision outputs, validation-only checkpoint selection and held-out metric fields. This establishes the software path only. HipMRI data audit, GPU smoke measurement and final Dice values are not yet claimed.")

    _heading(doc, "Risks budget and fallback")
    _body(doc, "Main risks are 3D volume memory demand, small foreground regions, NIfTI orientation errors, slow I/O and failed cluster jobs. The next experiment is a short Rangpur GPU smoke run on audited volumes to set patch geometry, batch size and mixed-precision settings, followed by matched Slurm runs. If the improved model does not improve real held-out boundary failures at acceptable cost, the plain 3D U-Net remains the fallback recommendation.")

    _heading(doc, "Questions for tutor confirmation")
    _body(doc, "1. What is the HipMRI prostate-label encoding, and is background reported separately?  2. What is the exact calculation protocol for Dice >= 0.70?  3. What patch size, batch size, mixed-precision setting and epoch budget do you recommend?  4. Which logs, checkpoints and base/apex failure panels must be retained after the midpoint?  5. Will this practical be recorded as the completed check-off?")

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_run = footer.add_run("Prepared from the COMP3710 project brief and local implementation checks. Tutor confirmation remains required.")
    footer_run.font.size = Pt(7.2)
    footer_run.font.color.rgb = RGBColor(89, 89, 89)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
