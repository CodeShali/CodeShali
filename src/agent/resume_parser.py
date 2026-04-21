import logging
from docx import Document

logger = logging.getLogger(__name__)


def parse_resume(path: str) -> str:
    doc = Document(path)
    lines = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            lines.append(text)

    # Also pull text from tables (some resumes use table layouts)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                text = cell.text.strip()
                if text and text not in lines:
                    lines.append(text)

    resume_text = "\n".join(lines)
    logger.info("Resume parsed: %d characters", len(resume_text))
    return resume_text
