"""
db_tools.py — SQLite tool-call layer for the Pāli RAG app.

Exposes a set of structured tools that can be registered with Gemini's
function-calling API so the model can autonomously explore the corpus
before (or after) vector retrieval.

Tools provided
──────────────
1. list_books            — list books, filterable by nikaya / sub_nikaya / category
2. search_books          — fuzzy search on book titles / descriptions
3. get_table_of_contents — all headings for a book_id
4. get_paragraph         — content + word/line count for one paragraph
5. get_paragraph_range   — fetch N consecutive paragraphs (with optional language)
6. get_nearby_context    — paragraphs immediately before and after a para_id
7. get_book_stats        — paragraph count, line count, first/last para_id
8. find_paragraphs_by_keyword — keyword search across Pali / English / Vietnamese

Usage (in gemini_client.py or app.py)
──────────────────────────────────────
    from db_tools import DB_TOOLS, dispatch_tool

    # Pass DB_TOOLS to Gemini as tools=[...] in generate_content().
    # When the model returns a function_call part, route it through:
    result = dispatch_tool(part.function_call.name,
                           dict(part.function_call.args))
"""

import sqlite3
from contextlib import contextmanager
from typing import Any

from config_tmp import DB_PATH


# ── Connection helper ──────────────────────────────────────────────────────

@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


# ═══════════════════════════════════════════════════════════════════════════
#  TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

def list_books(
    nikaya     : str | None = None,
    sub_nikaya : str | None = None,
    category   : str | None = None,
    limit      : int = 50,
) -> dict:
    """
    List books from the `books` table.

    Parameters
    ----------
    nikaya      : Filter by nikaya (e.g. "Sutta Piṭaka"). Case-insensitive
                  substring match. Omit to return all nikayas.
    sub_nikaya  : Filter by sub-nikaya (e.g. "Dīgha Nikāya"). Same matching.
    category    : Filter by category column. Same matching.
    limit       : Maximum number of books to return (default 50, max 200).

    Returns
    -------
    {"books": [{book_id, book_name, nikaya, sub_nikaya, category}, ...],
     "count": int}
    """
    limit = min(int(limit or 50), 200)
    conditions: list[str] = []
    params: list[Any] = []

    if nikaya:
        conditions.append("LOWER(nikaya) LIKE LOWER(?)")
        params.append(f"%{nikaya}%")
    if sub_nikaya:
        conditions.append("LOWER(sub_nikaya) LIKE LOWER(?)")
        params.append(f"%{sub_nikaya}%")
    if category:
        conditions.append("LOWER(category) LIKE LOWER(?)")
        params.append(f"%{category}%")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"""
        SELECT book_id, book_name, nikaya, sub_nikaya, category
        FROM   books
        {where}
        ORDER  BY nikaya, sub_nikaya, book_name
        LIMIT  ?
    """
    params.append(limit)

    with _conn() as con:
        rows = con.execute(sql, params).fetchall()

    books = [
        {
            "book_id"   : r["book_id"],
            "book_name" : r["book_name"],
            "nikaya"    : r["nikaya"]     or "",
            "sub_nikaya": r["sub_nikaya"] or "",
            "category"  : r["category"]  or "",
        }
        for r in rows
    ]
    return {"books": books, "count": len(books)}


def search_books(query: str, limit: int = 20) -> dict:
    """
    Search for books whose name, nikaya, sub_nikaya, or category contains
    the given query string (case-insensitive).

    Parameters
    ----------
    query : Search term (e.g. "Dhammapada", "vinaya", "jataka").
    limit : Maximum results (default 20).

    Returns
    -------
    {"books": [{book_id, book_name, nikaya, sub_nikaya, category}], "count": int}
    """
    limit = min(int(limit or 20), 100)
    pattern = f"%{query}%"
    sql = """
        SELECT book_id, book_name, nikaya, sub_nikaya, category
        FROM   books
        WHERE  LOWER(book_name)   LIKE LOWER(?)
            OR LOWER(nikaya)      LIKE LOWER(?)
            OR LOWER(sub_nikaya)  LIKE LOWER(?)
            OR LOWER(category)    LIKE LOWER(?)
        ORDER  BY book_name
        LIMIT  ?
    """
    with _conn() as con:
        rows = con.execute(sql, [pattern, pattern, pattern, pattern, limit]).fetchall()

    books = [
        {
            "book_id"   : r["book_id"],
            "book_name" : r["book_name"],
            "nikaya"    : r["nikaya"]     or "",
            "sub_nikaya": r["sub_nikaya"] or "",
            "category"  : r["category"]  or "",
        }
        for r in rows
    ]
    return {"books": books, "count": len(books)}


def get_table_of_contents(book_id: str, max_depth: int = 9) -> dict:
    """
    Return all headings for a given book, forming its table of contents.

    Parameters
    ----------
    book_id   : The book identifier (e.g. "dn1").
    max_depth : Only include headings with level ≤ this value
                (1 = top-level only, 9 = all headings). Default 9.

    Returns
    -------
    {"book_id": str, "headings": [{para_id, level, heading_title}],
     "count": int}
    """
    max_depth = max(1, min(int(max_depth or 9), 20))
    results: list[dict] = []

    with _conn() as con:
        for tbl in ("paragraphs", "headings", "para"):
            try:
                rows = con.execute(
                    f"""
                    SELECT para_id, level, heading_title
                    FROM   {tbl}
                    WHERE  book_id = ?
                      AND  level IS NOT NULL
                      AND  level <= ?
                    ORDER  BY para_id
                    """,
                    (book_id, max_depth),
                ).fetchall()

                results = [
                    {
                        "para_id"       : r["para_id"],
                        "level": r["level"],
                        "heading_title" : r["heading_title"] or "",
                    }
                    for r in rows
                ]
                break   # found the right table
            except Exception:
                continue

    return {"book_id": book_id, "headings": results, "count": len(results)}


def get_paragraph(
    book_id  : str,
    para_id  : int,
    language : str = "english",
) -> dict:
    """
    Fetch all sentences in a paragraph and return the text together with
    basic statistics (line count, word count).

    Parameters
    ----------
    book_id  : Book identifier.
    para_id  : Paragraph number.
    language : Which translation to return: "english", "pali", "vietnamese",
               or "all" (returns all three). Default "english".

    Returns
    -------
    {
      "book_id": str, "para_id": int,
      "pali": str,          # only if language in ("pali", "all")
      "english": str,       # only if language in ("english", "all")
      "vietnamese": str,    # only if language in ("vietnamese", "all")
      "line_count": int,
      "word_count": int,    # word count of the requested language text
      "thaipage": str, "vripage": str, "ptspage": str,
    }
    """
    language = (language or "english").lower()

    with _conn() as con:
        rows = con.execute(
            """
            SELECT line_id, pali_sentence, english_translation,
                   vietnamese_translation, thaipage, vripage, ptspage
            FROM   sentences
            WHERE  book_id = ? AND para_id = ?
            ORDER  BY line_id
            """,
            (book_id, int(para_id)),
        ).fetchall()

    if not rows:
        return {
            "book_id": book_id, "para_id": para_id,
            "error": "Paragraph not found.",
            "line_count": 0, "word_count": 0,
        }

    pali_lines  = [r["pali_sentence"]          or "" for r in rows]
    eng_lines   = [r["english_translation"]    or "" for r in rows]
    viet_lines  = [r["vietnamese_translation"] or "" for r in rows]

    pali_text  = "\n".join(l for l in pali_lines  if l)
    eng_text   = "\n".join(l for l in eng_lines   if l)
    viet_text  = "\n".join(l for l in viet_lines  if l)

    first = rows[0]

    # Word count uses the primary requested language
    primary_text = {
        "pali": pali_text, "english": eng_text, "vietnamese": viet_text,
    }.get(language, eng_text)
    word_count = len(primary_text.split()) if primary_text.strip() else 0

    result: dict = {
        "book_id"   : book_id,
        "para_id"   : para_id,
        "line_count": len(rows),
        "word_count": word_count,
        "thaipage"  : first["thaipage"] or "",
        "vripage"   : first["vripage"]  or "",
        "ptspage"   : first["ptspage"]  or "",
    }

    if language in ("pali", "all"):
        result["pali"] = pali_text
    if language in ("english", "all"):
        result["english"] = eng_text
    if language in ("vietnamese", "all"):
        result["vietnamese"] = viet_text

    # Always include all if "all" was requested
    if language not in ("pali", "english", "vietnamese", "all"):
        result["english"] = eng_text   # safe default

    return result


def get_paragraph_range(
    book_id    : str,
    para_start : int,
    para_end   : int,
    language   : str = "english",
) -> dict:
    """
    Fetch multiple consecutive paragraphs from a book.

    Parameters
    ----------
    book_id    : Book identifier.
    para_start : First paragraph ID (inclusive).
    para_end   : Last paragraph ID (inclusive). Maximum span: 20 paragraphs.
    language   : "english", "pali", "vietnamese", or "all". Default "english".

    Returns
    -------
    {"book_id": str, "paragraphs": [{para_id, text, line_count}], "count": int}
    """
    language  = (language or "english").lower()
    para_start = int(para_start)
    para_end   = min(int(para_end), para_start + 19)   # cap at 20 paras

    with _conn() as con:
        rows = con.execute(
            """
            SELECT para_id, line_id,
                   pali_sentence, english_translation, vietnamese_translation
            FROM   sentences
            WHERE  book_id = ?
              AND  para_id BETWEEN ? AND ?
            ORDER  BY para_id, line_id
            """,
            (book_id, para_start, para_end),
        ).fetchall()

    # Group by para_id
    paras: dict[int, list] = {}
    for r in rows:
        paras.setdefault(r["para_id"], []).append(r)

    paragraphs = []
    for pid in sorted(paras.keys()):
        lines = paras[pid]
        pali  = "\n".join((r["pali_sentence"]          or "").strip() for r in lines if r["pali_sentence"])
        eng   = "\n".join((r["english_translation"]    or "").strip() for r in lines if r["english_translation"])
        viet  = "\n".join((r["vietnamese_translation"] or "").strip() for r in lines if r["vietnamese_translation"])

        entry: dict = {"para_id": pid, "line_count": len(lines)}
        if language in ("pali", "all"):
            entry["pali"] = pali
        if language in ("english", "all"):
            entry["english"] = eng
        if language in ("vietnamese", "all"):
            entry["vietnamese"] = viet
        if language not in ("pali", "english", "vietnamese", "all"):
            entry["english"] = eng

        paragraphs.append(entry)

    return {"book_id": book_id, "paragraphs": paragraphs, "count": len(paragraphs)}


def get_nearby_context(
    book_id  : str,
    para_id  : int,
    before   : int = 2,
    after    : int = 2,
    language : str = "english",
) -> dict:
    """
    Return paragraphs immediately before and after a given paragraph,
    useful for reading context around a RAG-retrieved chunk.

    Parameters
    ----------
    book_id  : Book identifier.
    para_id  : The central paragraph.
    before   : How many paragraphs to include before (default 2, max 5).
    after    : How many paragraphs to include after  (default 2, max 5).
    language : "english", "pali", "vietnamese", or "all". Default "english".

    Returns
    -------
    {
      "book_id": str, "center_para_id": int,
      "paragraphs": [{para_id, text, is_center: bool}],
    }
    """
    before   = max(0, min(int(before or 2), 5))
    after    = max(0, min(int(after  or 2), 5))
    para_id  = int(para_id)
    start    = para_id - before
    end      = para_id + after

    fetched  = get_paragraph_range(book_id, start, end, language)
    paras    = fetched.get("paragraphs", [])

    for p in paras:
        p["is_center"] = (p["para_id"] == para_id)

    return {
        "book_id"       : book_id,
        "center_para_id": para_id,
        "paragraphs"    : paras,
    }


def get_book_stats(book_id: str) -> dict:
    """
    Return aggregate statistics for a book: paragraph count, total line
    count, and first / last para_id.  Useful before deciding how many
    paragraphs to retrieve.

    Parameters
    ----------
    book_id : Book identifier.

    Returns
    -------
    {
      "book_id": str, "book_name": str, "nikaya": str,
      "para_count": int, "line_count": int,
      "first_para_id": int | None, "last_para_id": int | None,
    }
    """
    with _conn() as con:
        agg = con.execute(
            """
            SELECT COUNT(DISTINCT para_id) AS para_count,
                   COUNT(*)               AS line_count,
                   MIN(para_id)           AS first_para,
                   MAX(para_id)           AS last_para
            FROM   sentences
            WHERE  book_id = ?
            """,
            (book_id,),
        ).fetchone()

        book_row = con.execute(
            "SELECT book_name, nikaya, sub_nikaya FROM books WHERE book_id = ? LIMIT 1",
            (book_id,),
        ).fetchone()

    return {
        "book_id"      : book_id,
        "book_name"    : (book_row["book_name"]  if book_row else "") or "",
        "nikaya"       : (book_row["nikaya"]     if book_row else "") or "",
        "sub_nikaya"   : (book_row["sub_nikaya"] if book_row else "") or "",
        "para_count"   : agg["para_count"]  if agg else 0,
        "line_count"   : agg["line_count"]  if agg else 0,
        "first_para_id": agg["first_para"]  if agg else None,
        "last_para_id" : agg["last_para"]   if agg else None,
    }


def find_paragraphs_by_keyword(
    keyword  : str,
    book_id  : str | None = None,
    language : str = "english",
    limit    : int = 20,
) -> dict:
    """
    Search for paragraphs whose text contains a keyword (case-insensitive).
    Useful for exact-match lookups of proper names, verse numbers, or Pali
    terms that vector search might miss.

    Parameters
    ----------
    keyword  : Search term (e.g. "ānāpānasati", "Devadatta", "pātimokkha").
    book_id  : Restrict search to a specific book. Omit to search all books.
    language : Which column to search: "english", "pali", "vietnamese".
               Default "english".
    limit    : Maximum results (default 20, max 100).

    Returns
    -------
    {
      "keyword": str, "language": str,
      "results": [{book_id, para_id, line_id, snippet}],
      "count": int,
    }
    """
    language = (language or "english").lower()
    limit    = min(int(limit or 20), 100)

    col_map = {
        "english"   : "english_translation",
        "pali"      : "pali_sentence",
        "vietnamese": "vietnamese_translation",
    }
    col = col_map.get(language, "english_translation")

    conditions = [f"LOWER({col}) LIKE LOWER(?)"]
    params: list[Any] = [f"%{keyword}%"]

    if book_id:
        conditions.append("book_id = ?")
        params.append(book_id)

    where = " AND ".join(conditions)
    sql = f"""
        SELECT book_id, para_id, line_id, {col} AS snippet
        FROM   sentences
        WHERE  {where}
        ORDER  BY book_id, para_id, line_id
        LIMIT  ?
    """
    params.append(limit)

    with _conn() as con:
        rows = con.execute(sql, params).fetchall()

    results = [
        {
            "book_id": r["book_id"],
            "para_id": r["para_id"],
            "line_id": r["line_id"],
            "snippet": (r["snippet"] or "")[:300],   # truncate long lines
        }
        for r in rows
    ]
    return {"keyword": keyword, "language": language, "results": results, "count": len(results)}


# ═══════════════════════════════════════════════════════════════════════════
#  DISPATCH TABLE
# ═══════════════════════════════════════════════════════════════════════════

_DISPATCH = {
    "list_books"                 : list_books,
    "search_books"               : search_books,
    "get_table_of_contents"      : get_table_of_contents,
    "get_paragraph"              : get_paragraph,
    "get_paragraph_range"        : get_paragraph_range,
    "get_nearby_context"         : get_nearby_context,
    "get_book_stats"             : get_book_stats,
    "find_paragraphs_by_keyword" : find_paragraphs_by_keyword,
}


def dispatch_tool(name: str, args: dict) -> dict:
    """
    Route a Gemini function_call to the matching Python function.

    Parameters
    ----------
    name : function_call.name from Gemini's response.
    args : dict(function_call.args)

    Returns
    -------
    Result dict from the tool, or {"error": "..."} on failure.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**args)
    except Exception as exc:
        return {"error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════
#  GEMINI TOOL DECLARATIONS  (google-genai SDK v1+ format)
# ═══════════════════════════════════════════════════════════════════════════
#
# Pass  `tools=[DB_TOOLS]`  to  client.models.generate_content()
# along with  `tool_config=ToolConfig(function_calling_config=
#   FunctionCallingConfig(mode="AUTO"))`.
#
# The list below uses google.genai.types.Tool / FunctionDeclaration /
# Schema objects.  Import and use exactly as shown in the usage example
# at the top of this file.

from google.genai import types as _T

def _str(desc: str, enum: list[str] | None = None) -> _T.Schema:
    s = _T.Schema(type="STRING", description=desc)
    if enum:
        s.enum = enum
    return s

def _int(desc: str) -> _T.Schema:
    return _T.Schema(type="INTEGER", description=desc)


DB_TOOLS = _T.Tool(function_declarations=[

    _T.FunctionDeclaration(
        name="list_books",
        description=(
            "List books in the Pāli Canon database, optionally filtered by "
            "nikaya, sub_nikaya, or category. Use this to discover which books "
            "are available before fetching their content."
        ),
        parameters=_T.Schema(
            type="OBJECT",
            properties={
                "nikaya"    : _str("Filter by nikaya (e.g. 'Sutta Piṭaka'). Substring match."),
                "sub_nikaya": _str("Filter by sub-nikaya (e.g. 'Majjhima Nikāya')."),
                "category"  : _str("Filter by category column."),
                "limit"     : _int("Maximum books to return (default 50)."),
            },
        ),
    ),

    _T.FunctionDeclaration(
        name="search_books",
        description=(
            "Search for books by title or classification keyword. "
            "Use when the user mentions a text by name but you don't know "
            "its exact book_id (e.g. 'Dhammapada', 'Jātaka', 'Pācittiya')."
        ),
        parameters=_T.Schema(
            type="OBJECT",
            required=["query"],
            properties={
                "query": _str("Search term to match against book name, nikaya, sub_nikaya, or category."),
                "limit": _int("Maximum results (default 20)."),
            },
        ),
    ),

    _T.FunctionDeclaration(
        name="get_table_of_contents",
        description=(
            "Return the structured table of contents (all section headings) "
            "for a given book. Use to orient the model before fetching specific "
            "paragraphs, or to answer questions about the book's structure."
        ),
        parameters=_T.Schema(
            type="OBJECT",
            required=["book_id"],
            properties={
                "book_id"  : _str("Book identifier, e.g. 'dn1'."),
                "max_depth": _int("Only include headings at this depth or shallower (1–9). Default 9 = all headings."),
            },
        ),
    ),

    _T.FunctionDeclaration(
        name="get_paragraph",
        description=(
            "Fetch the full text and statistics (line count, word count) of "
            "a single paragraph. Use when you need the exact wording of a "
            "specific passage identified by (book_id, para_id)."
        ),
        parameters=_T.Schema(
            type="OBJECT",
            required=["book_id", "para_id"],
            properties={
                "book_id" : _str("Book identifier."),
                "para_id" : _int("Paragraph number."),
                "language": _str(
                    "Which text to return: 'english', 'pali', 'vietnamese', or 'all'.",
                    enum=["english", "pali", "vietnamese", "all"],
                ),
            },
        ),
    ),

    _T.FunctionDeclaration(
        name="get_paragraph_range",
        description=(
            "Fetch multiple consecutive paragraphs from a book. "
            "Useful for retrieving a full sutta section or vinaya rule "
            "that spans several paragraphs. Maximum span: 20 paragraphs."
        ),
        parameters=_T.Schema(
            type="OBJECT",
            required=["book_id", "para_start", "para_end"],
            properties={
                "book_id"   : _str("Book identifier."),
                "para_start": _int("First paragraph ID (inclusive)."),
                "para_end"  : _int("Last paragraph ID (inclusive)."),
                "language"  : _str(
                    "Language: 'english', 'pali', 'vietnamese', or 'all'. Default 'english'.",
                    enum=["english", "pali", "vietnamese", "all"],
                ),
            },
        ),
    ),

    _T.FunctionDeclaration(
        name="get_nearby_context",
        description=(
            "Return paragraphs immediately before and after a given paragraph. "
            "Use after RAG retrieval to expand context around a matched chunk "
            "and check whether the surrounding text changes the meaning."
        ),
        parameters=_T.Schema(
            type="OBJECT",
            required=["book_id", "para_id"],
            properties={
                "book_id" : _str("Book identifier."),
                "para_id" : _int("The central paragraph ID."),
                "before"  : _int("Paragraphs to include before the centre (default 2, max 5)."),
                "after"   : _int("Paragraphs to include after the centre (default 2, max 5)."),
                "language": _str(
                    "Language: 'english', 'pali', 'vietnamese', or 'all'. Default 'english'.",
                    enum=["english", "pali", "vietnamese", "all"],
                ),
            },
        ),
    ),

    _T.FunctionDeclaration(
        name="get_book_stats",
        description=(
            "Return aggregate statistics for a book: total paragraph count, "
            "line count, and first / last para_id. "
            "Check this before fetching large ranges to avoid over-fetching."
        ),
        parameters=_T.Schema(
            type="OBJECT",
            required=["book_id"],
            properties={
                "book_id": _str("Book identifier."),
            },
        ),
    ),

    _T.FunctionDeclaration(
        name="find_paragraphs_by_keyword",
        description=(
            "Search for paragraphs that contain a specific keyword or phrase. "
            "Use for exact-match lookups of proper names (e.g. 'Devadatta'), "
            "Pāli terms (e.g. 'ānāpānasati'), or verse numbers that vector "
            "search might miss. Can be restricted to a single book."
        ),
        parameters=_T.Schema(
            type="OBJECT",
            required=["keyword"],
            properties={
                "keyword" : _str("Word or phrase to search for (case-insensitive)."),
                "book_id" : _str("Restrict to a specific book. Omit to search all."),
                "language": _str(
                    "Which column to search: 'english', 'pali', 'vietnamese'. Default 'english'.",
                    enum=["english", "pali", "vietnamese"],
                ),
                "limit"   : _int("Max results (default 20, max 100)."),
            },
        ),
    ),

])