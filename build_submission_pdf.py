"""Build the single BITS submission PDF: GitHub link, Streamlit link, Virtual Lab screenshot, README."""

import argparse
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"
STYLES = getSampleStyleSheet()
BODY = STYLES["BodyText"]


def inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return re.sub(r"(https?://[^\s)]+)", r"<link href='\1' color='blue'><u>\1</u></link>", text)


def find_link(readme_text, label):
    match = re.search(rf"- {label}: (\S+)", readme_text)
    return match.group(1) if match else ""


def scaled_image(path, max_width=17 * cm, max_height=15 * cm):
    image = Image(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
    return image


def table_flowable(rows):
    cells = [[Paragraph(inline(c.strip()), BODY) for c in row.strip("|").split("|")] for row in rows]
    width = 17 * cm
    col_widths = [width * 0.3] + [width * 0.7 / (len(cells[0]) - 1)] * (len(cells[0]) - 1)
    table = Table(cells, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def readme_flowables(readme_text):
    story, table_rows, code_lines, in_code = [], [], [], False
    for line in readme_text.splitlines() + [""]:
        if line.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), STYLES["Code"]))
                code_lines = []
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        if line.startswith("|"):
            if not re.match(r"^\|[\s\-:|]+\|$", line):
                table_rows.append(line)
            continue
        if table_rows:
            story += [table_flowable(table_rows), Spacer(1, 6)]
            table_rows = []
        stripped = line.strip()
        if not stripped:
            continue
        heading = re.match(r"^(#{1,3}) (.*)", stripped)
        image_line = re.match(r"^!\[[^\]]*\]\(([^)]+)\)$", stripped)
        if image_line and (ROOT / image_line.group(1)).exists():
            story.append(scaled_image(ROOT / image_line.group(1)))
        elif heading:
            style = {1: "Heading1", 2: "Heading2", 3: "Heading3"}[len(heading.group(1))]
            story.append(Paragraph(inline(heading.group(2)), STYLES[style]))
        elif re.match(r"^(- |\d+\. )", stripped):
            story.append(Paragraph(inline(stripped), BODY, bulletText=" "))
        else:
            story.append(Paragraph(inline(stripped), BODY))
    return story


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", default="bits_virtual_lab_screenshot.png")
    parser.add_argument("--output", default="2026AJ05411_ML_Assignment2.pdf")
    parser.add_argument("--preview", action="store_true", help="allow placeholders (do not submit)")
    args = parser.parse_args()

    readme_text = README.read_text(encoding="utf-8")
    github = find_link(readme_text, "GitHub repository")
    streamlit_url = find_link(readme_text, "Live Streamlit app")
    screenshot = ROOT / args.screenshot

    problems = []
    if not github.startswith("https://github.com/"):
        problems.append("README.md: set '- GitHub repository: https://github.com/...'")
    if not streamlit_url.startswith("https://"):
        problems.append("README.md: set '- Live Streamlit app: https://....streamlit.app'")
    if not screenshot.exists():
        problems.append(f"missing screenshot file: {screenshot.name}")
    if problems and not args.preview:
        sys.exit("Cannot build final PDF:\n  " + "\n  ".join(problems))

    story = [
        Paragraph("Machine Learning - Assignment 2", STYLES["Title"]),
        Paragraph("Nelli Haritha | BITS ID: 2026AJ05411 | M.Tech (AIML)", STYLES["Heading3"]),
        Spacer(1, 12),
        Paragraph("1. GitHub Repository Link", STYLES["Heading2"]),
        Paragraph(inline(github), BODY),
        Paragraph("2. Live Streamlit App Link", STYLES["Heading2"]),
        Paragraph(inline(streamlit_url), BODY),
        Paragraph("3. Screenshot - Assignment execution on BITS Virtual Lab", STYLES["Heading2"]),
    ]
    if screenshot.exists():
        story.append(scaled_image(screenshot))
    else:
        story.append(Paragraph("[SCREENSHOT MISSING]", BODY))
    story += [PageBreak(), Paragraph("4. GitHub README Content", STYLES["Heading2"])]
    story += readme_flowables(readme_text)

    output = ROOT / args.output
    SimpleDocTemplate(str(output), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=1.8 * cm, bottomMargin=1.8 * cm).build(story)
    print(f"Wrote {output}" + ("  (PREVIEW - placeholders present, do not submit)" if problems else ""))


if __name__ == "__main__":
    main()
