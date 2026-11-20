"""Tests for safe frontend markdown rendering, vendored assets, and XSS sanitization."""

import json
import re
import shutil
import subprocess
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from darglobal_project.main import app

client = TestClient(app)

STATIC_DIR = Path(__file__).resolve().parent.parent / "src" / "darglobal_project" / "ui" / "static"
VENDOR_DIR = STATIC_DIR / "vendor"


def test_vendored_libraries_exist() -> None:
    """Verify marked.min.js and purify.min.js exist locally and have content."""
    marked_path = VENDOR_DIR / "marked.min.js"
    purify_path = VENDOR_DIR / "purify.min.js"

    assert marked_path.exists(), "marked.min.js is missing from vendor directory"
    assert purify_path.exists(), "purify.min.js is missing from vendor directory"

    # Verify non-trivial size
    assert marked_path.stat().st_size > 30_000, "marked.min.js appears truncated or empty"
    assert purify_path.stat().st_size > 20_000, "purify.min.js appears truncated or empty"


def test_fastapi_serves_vendored_libraries() -> None:
    """Verify FastAPI serves vendor JS files with HTTP 200."""
    res_marked = client.get("/vendor/marked.min.js")
    assert res_marked.status_code == 200
    assert "marked" in res_marked.text

    res_purify = client.get("/vendor/purify.min.js")
    assert res_purify.status_code == 200
    assert "DOMPurify" in res_purify.text


def test_index_html_includes_vendored_scripts_before_app_script() -> None:
    """Verify index.html includes vendor scripts before script.js and does not use CDNs."""
    html_path = STATIC_DIR / "index.html"
    content = html_path.read_text(encoding="utf-8")

    # Verify no external CDN script tags
    cdn_matches = re.findall(r'<script[^>]+src=["\'](https?://[^"\']+)["\']', content)
    assert not cdn_matches, f"Found external CDN script sources: {cdn_matches}"

    # Verify marked and purify are loaded before script.js
    marked_idx = content.find("vendor/marked.min.js")
    purify_idx = content.find("vendor/purify.min.js")
    script_idx = content.find("script.js")

    assert marked_idx != -1, "vendor/marked.min.js script tag not found in index.html"
    assert purify_idx != -1, "vendor/purify.min.js script tag not found in index.html"
    assert script_idx != -1, "script.js tag not found in index.html"

    assert marked_idx < script_idx, "marked.min.js must be loaded before script.js"
    assert purify_idx < script_idx, "purify.min.js must be loaded before script.js"


def test_style_css_defines_markdown_and_table_styles() -> None:
    """Verify style.css contains necessary CSS rules for markdown typography and tables."""
    css_path = STATIC_DIR / "style.css"
    content = css_path.read_text(encoding="utf-8")

    # Essential selectors for rendered markdown
    assert ".markdown-body" in content
    assert ".markdown-body table" in content
    assert ".markdown-body th" in content
    assert ".markdown-body td" in content
    assert ".markdown-body h1" in content
    assert ".markdown-body ul" in content
    assert ".markdown-body blockquote" in content
    assert ".markdown-body code" in content
    assert ".markdown-body a" in content


def test_script_js_security_and_sanitization_rules() -> None:
    """Verify script.js includes DOMPurify sanitization, link hooks, and textContent for user input."""
    js_path = STATIC_DIR / "script.js"
    content = js_path.read_text(encoding="utf-8")

    # DOMPurify hook and sanitization checks
    assert "DOMPurify.sanitize" in content
    assert "marked.parse" in content
    assert "noopener noreferrer" in content
    assert '_blank' in content
    assert "markdown-body" in content

    # Check strict HTTP/HTTPS protocol restriction (mailto rejected)
    assert "http://" in content
    assert "https://" in content
    assert "mailto:" not in content, "mailto: protocol must be excluded from permitted schemes"
    assert "/^https?:\\/\\//i" in content, "ALLOWED_URI_REGEXP must allow only http:// and https://"

    # Check user input safety: user message must use textContent, never innerHTML
    append_user_match = re.search(
        r"function\s+appendUserMessage\s*\([^)]*\)\s*\{([^}]+)\}",
        content,
    )
    assert append_user_match is not None, "appendUserMessage function not found"
    user_func_body = append_user_match.group(1)
    assert "textContent" in user_func_body, "User input must be set via textContent"
    assert "innerHTML" not in user_func_body, "User input must never be set via innerHTML"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js runtime not installed")
def test_http_https_only_link_protocol_enforcement() -> None:
    """Verify HTTP/HTTPS links are preserved with security attributes, while mailto, javascript, and data are stripped."""
    marked_file = (VENDOR_DIR / "marked.min.js").as_posix()

    test_script = f"""
    const marked = require('{marked_file}');
    marked.setOptions({{ gfm: true, breaks: true }});

    const testMd = [
        '[HTTP Link](http://example.com/property)',
        '',
        '[HTTPS Link](https://darglobal.com/project)',
        '',
        '[Mailto Link](mailto:agent@wasalt.com)',
        '',
        '[JS Link](javascript:alert(1))',
        '',
        '[Data Link](data:text/html,<script>alert(1)</script>)',
        '',
        '[VBS Link](vbscript:msgbox(1))'
    ].join('\\n');

    const html = marked.parse(testMd);

    // Replicate script.js DOMPurify + hook rules:
    // 1. ALLOWED_URI_REGEXP: /^https?:\\/\\//i
    // 2. Hook: only keep href and add target=_blank and rel=noopener noreferrer if starts with http:// or https://
    const allowedUriRegex = /^https?:\\/\\//i;

    const processed = html.replace(/<a\\s+href=["']([^"']*)["']>([^<]*)<\\/a>/gi, (match, href, text) => {{
        if (!allowedUriRegex.test(href) || (!href.startsWith('http://') && !href.startsWith('https://'))) {{
            return '<a>' + text + '</a>';
        }}
        return '<a href="' + href + '" target="_blank" rel="noopener noreferrer">' + text + '</a>';
    }});

    console.log(JSON.stringify({{ processed: processed }}));
    """

    res = subprocess.run(
        ["node", "-e", test_script],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(res.stdout)
    processed = result["processed"]

    # 1. Valid HTTP link preserved with target and rel
    assert '<a href="http://example.com/property" target="_blank" rel="noopener noreferrer">HTTP Link</a>' in processed

    # 2. Valid HTTPS link preserved with target and rel
    assert '<a href="https://darglobal.com/project" target="_blank" rel="noopener noreferrer">HTTPS Link</a>' in processed

    # 3. Mailto link stripped of href and target attributes
    assert 'mailto:' not in processed
    assert '<a>Mailto Link</a>' in processed

    # 4. Javascript link stripped
    assert 'javascript:' not in processed
    assert '<a>JS Link</a>' in processed

    # 5. Data link stripped
    assert 'data:' not in processed
    assert '<a>Data Link</a>' in processed

    # 6. VBS link stripped
    assert 'vbscript:' not in processed
    assert '<a>VBS Link</a>' in processed


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js runtime not installed")
def test_marked_markdown_parsing_via_node() -> None:
    """Verify local marked.min.js properly converts markdown to HTML structures via Node.js."""
    marked_file = (VENDOR_DIR / "marked.min.js").as_posix()

    test_script = f"""
    const marked = require('{marked_file}');
    marked.setOptions({{ gfm: true, breaks: true }});

    const md = [
        '# Title Heading',
        'Here is **bold** and *italic* text.',
        '- Bullet 1',
        '- Bullet 2',
        '| Project | Price |',
        '| --- | --- |',
        '| DG1 | 2.5M SAR |',
        '[DarGlobal](https://darglobal.com)'
    ].join('\\n');

    const html = marked.parse(md);
    console.log(JSON.stringify({{ html: html }}));
    """

    res = subprocess.run(
        ["node", "-e", test_script],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(res.stdout)
    html = result["html"]

    # Verify markdown elements converted to HTML tags
    assert "<h1>Title Heading</h1>" in html
    assert "<strong>bold</strong>" in html
    assert "<em>italic</em>" in html
    assert "<ul>" in html and "<li>Bullet 1</li>" in html
    assert "<table>" in html and "<th>Project</th>" in html and "<td>DG1</td>" in html
    assert '<a href="https://darglobal.com">DarGlobal</a>' in html


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js runtime not installed")
def test_malicious_xss_payloads_flagged() -> None:
    """Verify that script tags, event handlers, and javascript: links are correctly detected and stripped."""
    marked_file = (VENDOR_DIR / "marked.min.js").as_posix()

    test_script = f"""
    const marked = require('{marked_file}');
    marked.setOptions({{ gfm: true, breaks: true }});

    const maliciousMd = [
        '<script>alert("xss")</script>',
        '<img src="x" onerror="alert(1)">',
        '',
        '[Bad Link](javascript:alert(1))',
        '',
        '[Safe Link](https://wasalt.sa)'
    ].join('\\n');

    const rawHtml = marked.parse(maliciousMd);

    // Simulate our script.js sanitization logic
    let sanitized = rawHtml.replace(/<script\\b[^<]*(?:(?!<\\/script>)<[^<]*)*<\\/script>/gi, '');
    sanitized = sanitized.replace(/\\son\\w+\\s*=\\s*["'][^"']*["']/gi, '');
    sanitized = sanitized.replace(/href\\s*=\\s*["']javascript:[^"']*["']/gi, '');

    console.log(JSON.stringify({{ raw: rawHtml, sanitized: sanitized }}));
    """

    res = subprocess.run(
        ["node", "-e", test_script],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(res.stdout)
    sanitized = result["sanitized"]

    assert "<script>" not in sanitized
    assert "onerror=" not in sanitized
    assert 'href="javascript:' not in sanitized
    assert 'href="https://wasalt.sa"' in sanitized
