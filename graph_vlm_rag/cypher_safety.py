"""Cypher safety validator — read-only whitelist check"""

import re
from typing import Tuple


# Read-only Cypher keywords
READ_ONLY_KEYWORDS = frozenset({
    "MATCH",
    "RETURN",
    "WHERE",
    "WITH",
    "OPTIONAL MATCH",
    "ORDER BY",
    "SKIP",
    "LIMIT",
    "COUNT",
    "SIZE",
    "COLLECT",
    "DISTINCT",
    "AS",
})

# Forbidden keywords (write operations)
FORBIDDEN_KEYWORDS = frozenset({
    "CREATE",
    "MERGE",
    "SET",
    "DELETE",
    "REMOVE",
    "DETACH DELETE",
    "DROP",
    "FOREACH",
    "CALL",
})


def validate_cypher(cypher: str) -> Tuple[bool, str | None]:
    """
    Validate that a Cypher query is read-only.

    Args:
        cypher: Cypher query string

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not cypher or not cypher.strip():
        return False, "Empty Cypher query"

    # Tokenize: split on non-alphanumeric, check each token
    tokens = set(re.findall(r'[A-Za-z_]+', cypher.upper()))

    # Check for forbidden keywords
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in tokens:
            return False, f"Forbidden keyword: {keyword}"

    # Check that MATCH exists (basic validation)
    if "MATCH" not in tokens and "RETURN" not in tokens:
        return False, "Query must contain MATCH or RETURN"

    return True, None


def sanitize_identifier(identifier: str) -> str:
    """
    Sanitize a node/relationship identifier.

    Args:
        identifier: Raw identifier string

    Returns:
        Sanitized identifier (alphanumeric + underscore only)
    """
    # Replace spaces/hyphens/dots/slashes with underscores
    sanitized = re.sub(r"[\s\-./\\]+", "_", identifier)
    # Remove anything that's not alphanumeric or underscore
    sanitized = re.sub(r"[^A-Za-z0-9_]", "", sanitized)
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Strip leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Return default if empty
    if not sanitized:
        return "Entity"

    # Ensure starts with uppercase
    return sanitized[0].upper() + sanitized[1:] if len(sanitized) > 1 else sanitized.upper()