#!/usr/bin/env python3
"""Sync Overleaf project content to/from the local draft/ folder.

This script is meant to be called by Claude Code using the Overleaf MCP server.
It provides a manual sync mechanism — run when you want to pull latest from
Overleaf or push local changes back.

Usage (within Claude Code):
    python scripts/sync_overleaf.py --pull    # Overleaf → local draft/
    python scripts/sync_overleaf.py --push    # local draft/ → Overleaf

Note: Actual sync is done via the Overleaf MCP server tools. This script
just documents the process and provides the project ID reference.
"""

# Overleaf Project ID for the paper draft
OVERLEAF_PROJECT_ID = "6999d011b24a9f1d4e6e53e8"
OVERLEAF_URL = f"https://www.overleaf.com/project/{OVERLEAF_PROJECT_ID}"
LOCAL_DRAFT_DIR = "draft/"

print(f"""
Overleaf Sync Configuration
============================
Project URL: {OVERLEAF_URL}
Project ID:  {OVERLEAF_PROJECT_ID}
Local dir:   {LOCAL_DRAFT_DIR}

To sync, use Claude Code with the Overleaf MCP server:

  Pull (Overleaf → local):
    1. mcp__overleaf__list_files(projectId="{OVERLEAF_PROJECT_ID}")
    2. For each file: mcp__overleaf__read_file(filePath=<path>, projectId="{OVERLEAF_PROJECT_ID}")
    3. Write to draft/<path>

  Push (local → Overleaf):
    1. Read local draft/ files
    2. mcp__overleaf__write_file(filePath=<path>, content=<content>, projectId="{OVERLEAF_PROJECT_ID}")
    3. mcp__overleaf__commit_changes(message="sync from local")
    4. mcp__overleaf__push_changes()
""")
