from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, List
from app.models.types import Segment


# ---------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------
class BaseChunker(ABC):
    """Chunk Segments while *preserving metadata*."""

    def __init__(self, max_chars: int = 800, overlap: int = 150):
        self.max_chars = max_chars
        self.overlap = overlap

    @abstractmethod
    def chunk(self, segments: Iterable[Segment]) -> List[Segment]:
        raise NotImplementedError

    def __call__(self, segments: Iterable[Segment]) -> List[Segment]:
        return self.chunk(segments)

    # helper: copy meta and add subchunk info
    def _with_submeta(self, seg: Segment, sub_index: int) -> Segment:
        meta = dict(seg.meta)
        meta["subchunk_index"] = sub_index
        meta["max_chars"] = self.max_chars
        meta["overlap"] = self.overlap
        return Segment(text=seg.text, meta=meta)


# ---------------------------------------------------------------------
# Line-aware chunker (good default)
# ---------------------------------------------------------------------
class SimpleTokenChunker(BaseChunker):
    """
    Splits each Segment.text by newlines into chunks <= max_chars.
    Preserves seg metadata and adds subchunk indices.
    """

    def chunk(self, segments: Iterable[Segment]) -> List[Segment]:
        out: List[Segment] = []
        for seg in segments:
            buf = ""
            sub_i = 0
            for line in seg.text.split("\n"):
                if len(buf) + len(line) + 1 > self.max_chars:
                    if buf.strip():
                        out.append(Segment(text=buf.strip(),
                                           meta={**seg.meta, "subchunk_index": sub_i}))
                        sub_i += 1
                    buf = buf[-self.overlap:] if self.overlap else ""
                buf = (buf + "\n" + line) if buf else line
            if buf.strip():
                out.append(Segment(text=buf.strip(),
                                   meta={**seg.meta, "subchunk_index": sub_i}))
        return out


# ---------------------------------------------------------------------
# Paragraph-based chunker
# ---------------------------------------------------------------------
class ParagraphChunker(BaseChunker):
    def chunk(self, segments: Iterable[Segment]) -> List[Segment]:
        out: List[Segment] = []
        for seg in segments:
            buf = ""
            sub_i = 0
            for para in seg.text.split("\n\n"):
                if len(buf) + len(para) + 1 > self.max_chars:
                    if buf.strip():
                        out.append(Segment(text=buf.strip(),
                                           meta={**seg.meta, "subchunk_index": sub_i}))
                        sub_i += 1
                    buf = buf[-self.overlap:] if self.overlap and len(buf) > self.overlap else ""
                buf = (buf + "\n\n" + para) if buf else para
            if buf.strip():
                out.append(Segment(text=buf.strip(),
                                   meta={**seg.meta, "subchunk_index": sub_i}))
        return out


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
class ChunkerFactory:
    _registry = {
        "simple": SimpleTokenChunker,
        "paragraph": ParagraphChunker,
    }

    @classmethod
    def register(cls, name: str, chunker_cls):
        cls._registry[name.lower()] = chunker_cls

    @classmethod
    def create(cls, name: str = "simple", **kwargs) -> BaseChunker:
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown chunker type: {name}")
        return cls._registry[name](**kwargs)
