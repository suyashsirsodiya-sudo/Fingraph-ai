import re
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator

TierType = Literal["fact", "inference", "hypothesis"]

class Edge(BaseModel):
    id: Optional[str] = None
    src: str
    dst: str
    relation: str
    tier: TierType
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_quote: str
    source_url: str
    source_date: str = "unknown"
    company: Optional[str] = None

    @model_validator(mode="after")
    def validate_rules(self) -> "Edge":
        # Rule 1: Reject any edge missing quote or URL
        if not self.evidence_quote or not self.evidence_quote.strip():
            raise ValueError("evidence_quote must be provided verbatim from source")
        if not self.source_url or not self.source_url.strip():
            raise ValueError("source_url must be provided")

        # Rule 3: Never use the word 'cause' for inference or hypothesis edges
        if self.tier in ("inference", "hypothesis") and re.search(r"\bcause[sd]?\b", self.relation, re.IGNORECASE):
            # Enforce rule: rewrite or raise
            self.relation = re.sub(r"\bcause[sd]?\b", "may affect", self.relation, flags=re.IGNORECASE)

        # Generate id if missing
        if not self.id:
            clean_src = re.sub(r"\W+", "_", self.src.lower()).strip("_")
            clean_dst = re.sub(r"\W+", "_", self.dst.lower()).strip("_")
            self.id = f"{clean_src}->{clean_dst}"

        return self


class GraphData(BaseModel):
    nodes: list[dict] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
