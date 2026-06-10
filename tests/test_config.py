"""Tests for configuration module."""

from graph_vlm_rag.config import get_settings


def test_settings_load():
    """Settings should load with defaults."""
    settings = get_settings()
    assert settings.ollama_url == "http://localhost:11434"
    assert settings.neo4j_password == "password"
