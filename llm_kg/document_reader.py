import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from bs4 import BeautifulSoup


class HTMLDocsReader(BaseReader):
    """HTMLDocsReader

    Extract text from text files with HTML tags into Document objects.
    """

    def __init__(
        self,
        *args: Any,
        tags: List[str] = ["h1", "h2"],
        remove_hyperlinks: bool = True,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._tags = tags
        self._remove_hyperlinks = remove_hyperlinks

    def html_to_docs(self, html_text: str, filename: str) -> List[Document]:
        """Convert an HTML text to a list of Document objects."""
        documents: List[Document] = []

        soup = BeautifulSoup(html_text, 'html.parser')
        current_text = ""

        def save_current_text(header_path: str):
            nonlocal current_text
            if current_text.strip():
                documents.append(
                    Document(
                        text=current_text.strip(),
                        metadata={
                            "File Name": filename,
                            "Content Type": "text",
                            "Header Path": header_path,
                        },
                    )
                )
                current_text = ""

        for tag in self._tags:
            for element in soup.find_all(tag):
                header_path = element.get_text().strip()
                sibling = element.find_next_sibling()
                while sibling and sibling.name not in self._tags:
                    if sibling.name:
                        current_text += sibling.get_text() + "\n"
                    sibling = sibling.find_next_sibling()

                save_current_text(header_path)

        return documents

    def remove_hyperlinks(self, content: str) -> str:
        """Remove hyperlinks from HTML content."""
        soup = BeautifulSoup(content, 'html.parser')
        for a in soup.find_all('a'):
            a.unwrap()
        return str(soup)

    def parse_tups(self, filepath: Path) -> List[Document]:
        """Parse file into Document objects."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if self._remove_hyperlinks:
            content = self.remove_hyperlinks(content)
        documents = self.html_to_docs(content, str(filepath))
        return documents

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file into Document objects."""
        documents = self.parse_tups(file)

        # Add additional extra info to metadata
        for doc in documents:
            doc.metadata.update(extra_info or {})

        return documents
