"""
File Loader Service (OOP)
=========================

Loads files into *structured segments* (text + metadata).
- XLSX → one Segment per row (sheet-aware)
- DOCX → one Segment per heading, paragraph, or table row
- Includes safety guards against empty, None, or overly long text
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Type, Optional
import pandas as pd
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table  # type: ignore
from app.models.types import Segment
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Base Loader
# ---------------------------------------------------------------------
class BaseLoader(ABC):
    """Abstract base loader class for all document types."""

    def __init__(self, path: str, base_meta: Optional[Dict] = None):
        self.path = Path(path)
        self.base_meta = base_meta or {}

    def exists(self) -> bool:
        return self.path.exists()

    def validate(self) -> None:
        if not self.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

    def _meta(self, **extra) -> Dict:
        """Combine base metadata with additional contextual data."""
        meta = dict(self.base_meta)
        meta.update({
            "source_path": str(self.path),
            "file_name": self.path.name,
            "file_ext": self.path.suffix.lower(),
        })
        meta.update(extra)
        return meta

    @abstractmethod
    def load(self) -> List[Segment]:
        """Return structured Segment objects."""
        raise NotImplementedError


# ---------------------------------------------------------------------
# XLSX Loader
# ---------------------------------------------------------------------
class XLSXLoader(BaseLoader):
    """Loads Excel such that each row → a Segment with readable metadata."""

    def load(self) -> List[Segment]:
        self.validate()
        sheets = pd.read_excel(self.path, sheet_name=None, dtype=str, keep_default_na=False)

        segments: List[Segment] = []
        for sheet_name, df in sheets.items():
            df.columns = [str(c).strip() for c in df.columns]
            id_col = next((c for c in df.columns if c.lower() in ("id", "row_id", "article_id")), None)

            for row_idx, row in df.iterrows():
                row_map: Dict[str, str] = {}
                parts = []
                for col in df.columns:
                    val = str(row[col]).strip()
                    if val:
                        parts.append(f"{col}: {val}")
                    row_map[col] = val

                if not parts:
                    continue  # skip empty rows

                text = "\n".join(parts)
                meta = self._meta(
                    doc_type="xlsx",
                    sheet=sheet_name,
                    row_index=int(row_idx),
                    row_id=(row[id_col] if id_col else None),
                    columns=list(df.columns),
                    row_nonempty_cols=[c for c in df.columns if row_map.get(c)],
                )
                meta["row"] = row_map
                segments.append(Segment(text=text, meta=meta))

        logger.info(f"✅ Loaded {len(segments)} segments from Excel: {self.path.name}")
        return segments


# ---------------------------------------------------------------------
# DOCX Loader (Hardened)
# ---------------------------------------------------------------------
class DOCXLoader(BaseLoader):
    """
    Parses Word documents into logical Segments:
      - Headings → hierarchical sections
      - Paragraphs → plain text under current section
      - Tables → each row becomes a segment with header mapping
    Adds robust cleaning, truncation, and metadata.
    """

    def __init__(self, path: str, base_meta: Optional[Dict] = None, max_chars: int = 2000):
        super().__init__(path, base_meta)
        self.max_chars = max_chars

    def load(self) -> List[Segment]:
        self.validate()
        doc = Document(self.path)
        segments: List[Segment] = []

        heading_stack: List[str] = []

        def current_section_meta():
            path = [h for h in heading_stack if h]
            return {
                "heading_path": path,
                "section_title": path[-1] if path else None,
                "section_depth": len(path),
            }

        def is_heading(p: Paragraph) -> Optional[int]:
            style = getattr(p.style, "name", "") or ""
            if style.lower().startswith("heading"):
                parts = style.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1])
                return 1
            return None

        def normalize_stack(level: int, title: str):
            while len(heading_stack) >= level:
                heading_stack.pop()
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(title.strip())

        def clean_text(text: str) -> Optional[str]:
            """Normalize text and guard against None/too-long values."""
            if not text:
                return None
            txt = text.strip()
            if not txt:
                return None
            if len(txt) > self.max_chars:
                logger.debug(f"✂️  Truncating long paragraph ({len(txt)} → {self.max_chars})")
                txt = txt[: self.max_chars]
            return txt

        def table_to_segments(table: Table, t_index: int):
            rows = table.rows
            if not rows:
                return
            headers = [c.text.strip() for c in rows[0].cells]
            has_header = all(h for h in headers)
            start_row = 1 if has_header else 0
            if not has_header:
                headers = [f"col_{i}" for i in range(len(rows[0].cells))]

            for r_idx in range(start_row, len(rows)):
                cells = rows[r_idx].cells
                row_map = {headers[i]: cells[i].text.strip() for i in range(len(cells))}
                text = "\n".join(f"{k}: {v}" for k, v in row_map.items() if v)
                text = clean_text(text)
                if not text:
                    continue
                meta = self._meta(
                    doc_type="docx",
                    element_type="table_row",
                    table_index=t_index,
                    table_row_index=r_idx,
                    headers=headers,
                    **current_section_meta(),
                )
                meta["row"] = row_map
                segments.append(Segment(text=text, meta=meta))

        # --- Iterate document body ---
        table_counter = 0
        for block in doc.element.body:
            if block.tag.endswith("}p"):
                p = Paragraph(block, doc)
                text = clean_text(p.text)
                if not text:
                    continue

                lvl = is_heading(p)
                if lvl:
                    normalize_stack(lvl, text)
                    segments.append(
                        Segment(
                            text=text,
                            meta=self._meta(
                                doc_type="docx",
                                element_type="heading",
                                heading_level=lvl,
                                **current_section_meta(),
                            ),
                        )
                    )
                else:
                    segments.append(
                        Segment(
                            text=text,
                            meta=self._meta(
                                doc_type="docx",
                                element_type="paragraph",
                                **current_section_meta(),
                            ),
                        )
                    )
            elif block.tag.endswith("}tbl"):
                table = Table(block, doc)
                table_to_segments(table, table_counter)
                table_counter += 1

        logger.info(f"✅ Loaded {len(segments)} segments from DOCX: {self.path.name}")
        if not segments:
            logger.warning(f"⚠️ No valid text found in DOCX: {self.path.name}")
        return segments


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
class FileLoaderFactory:
    """Factory to select the appropriate loader class based on file extension."""

    _loaders: dict[str, Type[BaseLoader]] = {
        ".xlsx": XLSXLoader,
        ".docx": DOCXLoader,
    }

    @classmethod
    def register_loader(cls, extension: str, loader_cls: Type[BaseLoader]) -> None:
        cls._loaders[extension.lower()] = loader_cls
        logger.debug(f"Registered new loader for {extension}")

    @classmethod
    def create_loader(cls, path: str, **kwargs) -> BaseLoader:
        ext = Path(path).suffix.lower()
        loader_cls = cls._loaders.get(ext)
        if not loader_cls:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader_cls(path, **kwargs)


# ---------------------------------------------------------------------
# Convenience Function
# ---------------------------------------------------------------------
def load_any(path: str, **kwargs) -> List[Segment]:
    """Shortcut to load any supported file (Excel or Word)."""
    loader = FileLoaderFactory.create_loader(path, **kwargs)
    return loader.load()
