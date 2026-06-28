"""
tools.py — LangChain tools for the pairing agent.
"""

import json
from langchain_core.tools import tool

_nissaya_db = None

def bind_nissaya_db(nissaya_db):
    global _nissaya_db
    _nissaya_db = nissaya_db

@tool
def get_nissaya_headings(book_id: str) -> str:
    """
    Fetch the top-level structural headings for a given Pali book_id in the Nissaya database.
    Use this tool to verify if the chapters/suttas of a candidate Nissaya book match the Thai headings.
    
    Args:
        book_id: The exact Nissaya book_id (e.g., 'D-i', 'Sv-i', 'Vin-iii').
    """
    if not _nissaya_db:
        return json.dumps({"error": "Database not bound"})
        
    rows = _nissaya_db.get_headings(book_id)
    if not rows:
        return json.dumps({"found": False, "message": f"No headings found for '{book_id}'."})

    out = [{"para_id": r["para_id"], "level": r["level"], "title": r["title"]} for r in rows]
    return json.dumps({"found": True, "count": len(rows), "headings": out})

ALL_TOOLS = [get_nissaya_headings]

def _invoke_tool(tool_name: str, tool_args: dict) -> str:
    tool_map = {t.name: t for t in ALL_TOOLS}
    if tool_name not in tool_map:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = tool_map[tool_name].invoke(tool_args)
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})
