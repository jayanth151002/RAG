import PyPDF2
from pathlib import Path
from typing import List


class PdfReader:
    def __init__(self):
        pass

    def load_data(self, file_path: Path) -> List[str]:
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text()
                return full_text

        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []
        except Exception as e:
            print(f"An error occurred while processing the PDF: {e}")
            return []
