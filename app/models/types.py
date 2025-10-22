from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

@dataclass
class Segment:
    """
    A unit of content ready for chunking/embedding.
    - text: the content
    - meta: arbitrary metadata (file, sheet, row, section, page, etc.)
    """
    text: str
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "meta": dict(self.meta)}
