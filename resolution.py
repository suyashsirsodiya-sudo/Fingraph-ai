"""
FinGraph AI - Entity Resolution (Pure Module)
Indexes canonical names, aliases, and tickers.
Rules:
- Whole-word, case-insensitive for canonical names and aliases.
- Exact-case, whole-word for tickers.
- If a mention matches more than one entity, return as 'ambiguous' with candidates.
- Never merge or guess.
"""

import re
from typing import Any, Dict, List, Optional


class EntityResolver:
    """Deterministic, honest entity resolver."""

    def __init__(self, entities: Optional[List[Dict[str, Any]]] = None):
        self.entities: List[Dict[str, Any]] = []
        self._name_matchers: List[Dict[str, Any]] = []
        self._ticker_matchers: List[Dict[str, Any]] = []

        if entities:
            self.index_entities(entities)

    def index_entities(self, entities: List[Dict[str, Any]]) -> None:
        """Index a collection of entity dictionaries."""
        self.entities = list(entities)
        self._name_matchers.clear()
        self._ticker_matchers.clear()

        for ent in entities:
            ent_id = str(ent.get("id"))
            canonical = str(ent.get("canonical_name", "")).strip()
            aliases = ent.get("aliases", []) or []
            ticker = ent.get("ticker")

            # Index canonical name (case-insensitive whole word)
            if canonical:
                pattern = re.compile(rf"\b{re.escape(canonical)}\b", re.IGNORECASE)
                self._name_matchers.append({
                    "entity": ent,
                    "term": canonical,
                    "pattern": pattern,
                    "kind": "canonical",
                })

            # Index aliases (case-insensitive whole word)
            for alias in aliases:
                alias_str = str(alias).strip()
                if alias_str:
                    pattern = re.compile(rf"\b{re.escape(alias_str)}\b", re.IGNORECASE)
                    self._name_matchers.append({
                        "entity": ent,
                        "term": alias_str,
                        "pattern": pattern,
                        "kind": "alias",
                    })

            # Index ticker (exact case whole word)
            if ticker and str(ticker).strip():
                ticker_str = str(ticker).strip()
                # Exact case: NO re.IGNORECASE
                pattern = re.compile(rf"\b{re.escape(ticker_str)}\b")
                self._ticker_matchers.append({
                    "entity": ent,
                    "term": ticker_str,
                    "pattern": pattern,
                    "kind": "ticker",
                })

    def resolve_mention(self, mention: str) -> Dict[str, Any]:
        """
        Resolve a single mention or search term.
        Returns:
          - resolved (single match)
          - ambiguous (multiple candidate matches)
          - unresolved (zero matches)
        """
        cleaned = mention.strip()
        if not cleaned:
            return {
                "status": "unresolved",
                "query": mention,
                "entity": None,
                "candidates": [],
                "detail": "Empty query",
            }

        matched_entity_ids = set()
        matched_entities = []

        # 1. Exact case ticker match check
        for tm in self._ticker_matchers:
            if tm["pattern"].fullmatch(cleaned):
                ent = tm["entity"]
                if ent["id"] not in matched_entity_ids:
                    matched_entity_ids.add(ent["id"])
                    matched_entities.append(ent)

        # 2. Case-insensitive name and alias match check
        for nm in self._name_matchers:
            if nm["pattern"].fullmatch(cleaned):
                ent = nm["entity"]
                if ent["id"] not in matched_entity_ids:
                    matched_entity_ids.add(ent["id"])
                    matched_entities.append(ent)

        # Evaluate match results without guessing or merging
        if len(matched_entities) == 1:
            return {
                "status": "resolved",
                "query": mention,
                "entity": matched_entities[0],
                "candidates": matched_entities,
            }
        elif len(matched_entities) > 1:
            return {
                "status": "ambiguous",
                "query": mention,
                "entity": None,
                "candidates": matched_entities,
                "detail": f"Mention '{mention}' matched multiple candidate entities. Resolution refused to avoid guessing.",
            }
        else:
            return {
                "status": "unresolved",
                "query": mention,
                "entity": None,
                "candidates": [],
                "detail": f"Mention '{mention}' did not match any known entity.",
            }

    def scan_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Scan text for all mentions of known entities.
        Returns matched spans with resolved or ambiguous status.
        """
        findings = []

        # Scan tickers (exact case)
        for tm in self._ticker_matchers:
            for match in tm["pattern"].finditer(text):
                findings.append({
                    "start": match.start(),
                    "end": match.end(),
                    "matched_text": match.group(0),
                    "entity_id": tm["entity"]["id"],
                    "entity": tm["entity"],
                    "match_type": "ticker",
                })

        # Scan names/aliases (case-insensitive)
        for nm in self._name_matchers:
            for match in nm["pattern"].finditer(text):
                findings.append({
                    "start": match.start(),
                    "end": match.end(),
                    "matched_text": match.group(0),
                    "entity_id": nm["entity"]["id"],
                    "entity": nm["entity"],
                    "match_type": nm["kind"],
                })

        return findings

    def resolve_text(self, text: str) -> List[Dict[str, Any]]:
        """Alias for scan_text for extracting entity matches."""
        return self.scan_text(text)
