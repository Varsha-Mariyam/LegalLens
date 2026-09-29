"""DOCX text extraction with python-docx (paragraphs and tables, in document order)."""
from __future__ import annotations

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph


def extract_docx(path: str) -> list[dict]:
    document = docx.Document(path)
    blocks: list[str] = []
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            text = Paragraph(child, document).text.strip()
            if text:
                blocks.append(text)
        elif tag == "tbl":
            for row in Table(child, document).rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    blocks.append(" | ".join(dict.fromkeys(cells)))
    return [{"page": 1, "text": "\n\n".join(blocks), "method": "python-docx"}]
