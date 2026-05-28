"""Unit tests for LLM helper utilities."""

import pytest

from services.llm import _strip_markdown_fences


def test_strip_markdown_html_fence():
    raw = "```html\n<!DOCTYPE html><html><body>ok</body></html>\n```"
    assert _strip_markdown_fences(raw).startswith("<!DOCTYPE")


def test_strip_markdown_plain_fence():
    raw = "```\n<div>content</div>\n```"
    assert _strip_markdown_fences(raw).startswith("<div>")


def test_strip_leaves_bare_html_untouched():
    html = "<!DOCTYPE html><html><body>x</body></html>"
    assert _strip_markdown_fences(html) == html
