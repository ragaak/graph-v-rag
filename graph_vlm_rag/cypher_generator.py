"""DSPy-based Cypher generator with strict validation and signature-based generation.

Replaces brittle LLM Cypher generation with a structured DSPy pipeline that
enforces schema-aware output and programmatically sanitizes artifacts.
"""

import re
from typing import Any, Dict, List, Optional

import dspy
from dspy import InputField, OutputField, Signature

from .config import get_settings
from .neo4j_store import Neo4jStore
from .cypher_safety import validate_cypher


# ---------------------------------------------------------------------------
# DSPy Signature: Strict input/output contract
# ---------------------------------------------------------------------------

class CypherGenerator(Signature):
    """
    Generate a clean, read-only Cypher query for the user's question.

    Output MUST be a single line of valid Cypher.
    - No markdown fences
    - No explanatory prose
    - Only patterns from the provided schema
    """

    db_schema: str = InputField(
        desc=(
            "Live database schema containing: entity labels, relationship "
            "types, and example query patterns. Use ONLY these patterns."
        ),
    )
    question: str = InputField(
        desc=(
            "The natural language question to translate to Cypher. "
            "Infer the most direct, read-only query (MATCH, RETURN, WHERE, WITH, OPTIONAL MATCH)."
        ),
    )
    cypher_query: str = OutputField(
        desc=(
            "A single-line, read-only Cypher query string. No markdown, no "
            "backticks, no explanation, no trailing whitespace. Example: "
            "MATCH (m:Model) WHERE m.name = 'DocVLM' RETURN m.name"
        ),
    )


# ---------------------------------------------------------------------------
# Sanitization utility: Strip LLM artifacts programmatically
# ---------------------------------------------------------------------------

# Compile regexes once at module load
_MARKDOWN_FENCE = re.compile(r"^```(?:cypher|graphql|sql)?\s*", re.MULTILINE)
_TRAILING_FENCE = re.compile(r"^```\s*$", re.MULTILINE)
_INLINE_BACKTICK = re.compile(r"`")
_PROSE_PREFACE = re.compile(
    r"^(Here(?:'s| is)|This(?: will)?|Returns?|Query(?: is)?|It will|Note:)\b.*?:?\s*",
    re.IGNORECASE,
)
_SEMICOLON_COMMENT = re.compile(r";.*$", re.MULTILINE)
_MULTI_WHITESPACE = re.compile(r"\s+")
_LEADING_TEXT = re.compile(
    r"^[A-Za-z\s:'\"]+(?:cypher query:|query:|cql:)\s*",
    re.IGNORECASE,
)


def _clean_cypher(raw: str) -> str:
    """
    Sanitize a raw LLM-generated Cypher string.

    Strips:
    - Markdown code fences (```cypher ... ```)
    - Inline backticks
    - Prose prefaces ("Here is the query:")
    - Trailing comments
    - Multi-line whitespace

    Returns:
        A single-line, clean Cypher string.
    """
    if not raw:
        return ""

    cypher = raw.strip()

    # 1. Strip code fences
    cypher = _MARKDOWN_FENCE.sub("", cypher)
    cypher = _TRAILING_FENCE.sub("", cypher)

    # 2. Remove inline backticks
    cypher = _INLINE_BACKTICK.sub("", cypher)

    # 3. Strip prose prefaces like "Here is the query:"
    cypher = _LEADING_TEXT.sub("", cypher)
    cypher = _PROSE_PREFACE.sub("", cypher)

    # 4. Remove inline comments
    cypher = _SEMICOLON_COMMENT.sub("", cypher)

    # 5. Collapse to single line + trim
    cypher = _MULTI_WHITESPACE.sub(" ", cypher).strip()

    # 6. Validate against safety policy
    is_valid, error = validate_cypher(cypher)
    if not is_valid:
        raise ValueError(f"Invalid Cypher after sanitization: {error}")

    return cypher


# ---------------------------------------------------------------------------
# Schema Provider: Live schema from Neo4j
# ---------------------------------------------------------------------------

class SchemaProvider:
    """Loads and formats the live Neo4j schema for the DSPy signature."""

    def __init__(self, schema_path: Optional[str] = None):
        self.schema_path = schema_path
        self._cache: Optional[Dict[str, Any]] = None

    def get_schema(self) -> Dict[str, Any]:
        """Fetch live schema from Neo4j + local domain config."""
        if self._cache is not None:
            return self._cache

        # Local domain schema (yaml)
        local_schema: Dict[str, Any] = {}
        if self.schema_path:
            from pathlib import Path
            import yaml

            path = Path(self.schema_path)
            if path.exists():
                with open(path) as f:
                    local_schema = yaml.safe_load(f) or {}

        # Live database schema
        live: Dict[str, Any] = {
            "entity_labels": [],
            "relationship_types": [],
            "constraints": [],
        }
        try:
            with Neo4jStore() as neo_store:
                db_schema = neo_store.get_schema()
                live["entity_labels"] = db_schema.get("entity_labels", [])
                live["relationship_types"] = db_schema.get("relationship_types", [])
                live["constraints"] = db_schema.get("constraints", [])
        except Exception as exc:  # pragma: no cover
            # Fall back to local schema only if DB is unavailable
            live["entity_labels"] = local_schema.get("entity_labels", [])
            live["relationship_types"] = local_schema.get("relationship_types", [])

        self._cache = live
        return self._cache

    def format_for_prompt(self) -> str:
        """Return a human-readable schema string for the LLM."""
        schema = self.get_schema()
        labels = ", ".join(schema.get("entity_labels", []))
        rels = ", ".join(schema.get("relationship_types", []))
        constraints = "\n".join(f"- {c}" for c in schema.get("constraints", []))

        return (
            f"Entity Labels: {labels}\n"
            f"Relationship Types: {rels}\n"
            f"Available Constraints:\n{constraints}"
        )


# ---------------------------------------------------------------------------
# DSPy Module: The graph brain wrapper
# ---------------------------------------------------------------------------

class GraphBrain(dspy.Module):
    """
    Production-grade Cypher generator using DSPy.

    Usage:
        brain = GraphBrain()
        cypher = brain.generate("What models use OCR Encoder?")
    """

    def __init__(
        self,
        schema_path: Optional[str] = None,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.schema_provider = SchemaProvider(schema_path)
        self.temperature = temperature

        # Configure DSPy with the local Ollama model
        settings = get_settings()
        # DSPy/LiteLLM requires the provider prefix for Ollama models
        # e.g. "ollama/qwen2.5:14b" instead of "qwen2.5:14b"
        model_name = settings.ollama_model
        if "/" not in model_name:
            model_name = f"ollama_chat/{model_name}"

        self._lm = dspy.LM(
            model=model_name,
            api_base=settings.ollama_url,
            max_tokens=500,
            temperature=temperature,
            cache=False,
        )
        dspy.settings.configure(lm=self._lm)

        # DSPy ChainOfThought wrapped around the signature
        self._predictor = dspy.ChainOfThought(CypherGenerator)

    def _format_schema(self) -> str:
        """Format the live schema for prompt injection."""
        return self.schema_provider.format_for_prompt()

    def forward(
        self,
        question: str,
        schema: Optional[str] = None,
    ) -> str:
        """
        Generate a clean Cypher query for the question.

        Args:
            question: User's natural language question
            schema: Optional override for the schema string

        Returns:
            Sanitized, valid Cypher query string
        """
        schema_str = schema or self._format_schema()

        # Use the DSPy predictor to enforce structured output
        prediction = self._predictor(
            db_schema=schema_str,
            question=question,
        )
        raw_cypher = prediction.cypher_query

        return _clean_cypher(raw_cypher)

    def generate(self, question: str) -> str:
        """
        Public API: Generate a Cypher query for a question.

        Args:
            question: User's question

        Returns:
            Clean Cypher string (or empty string on failure)
        """
        try:
            return self.forward(question=question)
        except Exception as exc:
            print(f"⚠️ Cypher generation failed: {exc}")
            return ""


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_graph_brain() -> GraphBrain:
    """
    Factory function to build a configured GraphBrain instance.

    Returns:
        A ready-to-use GraphBrain with live schema.
    """
    settings = get_settings()
    return GraphBrain(schema_path=settings.domain_schema_path)
