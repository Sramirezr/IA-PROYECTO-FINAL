from pathlib import Path

import pdfplumber


class CVReadError(ValueError):
    pass


def extract_cv_text(pdf_path: str | Path) -> str:
    text_parts = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
    except Exception as error:
        raise CVReadError(f"No se pudo leer el PDF: {error}") from error

    text = "\n".join(text_parts).strip()
    if len(text) < 100:
        raise CVReadError(
            "El PDF no tiene suficiente texto extraible. Puede ser un CV escaneado o una imagen."
        )

    return text
