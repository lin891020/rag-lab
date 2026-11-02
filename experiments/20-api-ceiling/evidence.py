"""Evidence for 18: 17's evidence.py (the same corpus as 16: the law, 22 clean tables, 87 raw
tables; 17's cells and 16's oracle) plus two cells that write the clean tables as kv lines.

  cards-kv          cards exactly as 17 picks it (same ranking, markdown lengths, 3,000 budget,
                    the same tables and the same surviving rows), and only then every clean table
                    rewritten as one kv line per row with short column names (short_names); raw
                    tables and articles unchanged. Same content, different writing
  cards-kv@3000     the cards ranking with the kv tables packed to 3,000 by their own length
                    (what kv costs at the same budget; retrieval track only)

17's cells:

  struct@1500       16's L1-struct exactly (16's clean tables, no subtitles): one chunk per article + one per clean row (kv) + 01's
                    raw row chunks, dense rank, whole chunks while they fit, stop at the first
                    that does not (10's budget_evidence)
  struct@3000-nosub the same with a 3,000 budget (the budget alone)
  struct@3000       the same with the clean tables' subtitles restored (the fix alone): 16's
                    clean tables dropped the PDF's second title line, so 表二五～二／三／四
                    ("…容許溫度 60／75／90℃") had one and the same title. Every cell below has it
  struct@3000-pack  the same, but skip a chunk that does not fit and keep going (the packing
                    rule cards use; retrieval track only)
  bm25@3000         07's RRF of the dense and the bigram-BM25 ranking over the struct chunks
                    (retrieval track only)
  cards             one card per table (id + title + column names + row keys) next to the
                    article chunks; dense rank; a card that comes up is replaced by its whole
                    table (clean: markdown, raw: the text) and packed (skip what does not fit)
  cards+rr          dense top 30 of the cards index reranked by bge-reranker-v2-m3, then as cards
  decomp            the question plus qwen3's 1–3 sub-queries, each ranks the cards index, the
                    rankings are interleaved (round-robin, first seen wins), then as cards
  struct@4000       struct / cards / decomp with a 4,000 budget: added after the retrieval track showed
  cards@4000        C3's two tables (1,454 + 1,019 characters) do not fit next to the first
  decomp@4000       hit in 3,000 (not pre-registered; described, no verdict)

Gold is judged on labels kept in the evidence: article numbers and `<table>|<row key>`. A whole
table carries a label per line, so when the first unit is cut to the budget only the rows that
survive count.
"""

from __future__ import annotations

import logging
from pathlib import Path

from chunkers import Chunk, article_chunks, kv_row_chunks, kv_row_line, parse_articles
from chunkers import table_row_chunks
from rag_lab.pdf import extract_text
from retrievers import BM25Index, DenseIndex
from rules import TABLES, row_key

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
ONE = HERE.parent / "01-chunking"
DEPTH = 50
RERANK_DEPTH = 30
EXCERPT_CHARS = 600  # oracle: this much of the gold article around its first mention of the table
RRF_K = 60
CELLS = [
    "struct@1500",
    "struct@3000-nosub",
    "struct@3000",
    "struct@3000-pack",
    "bm25@3000",
    "cards",
    "cards+rr",
    "decomp",
    "struct@4000",
    "cards@4000",
    "decomp@4000",
    "cards-kv",
    "cards-kv@3000",
]
NOSUB = {"struct@1500", "struct@3000-nosub"}
STOP_AT_MISFIT = {"struct@1500", "struct@3000-nosub", "struct@3000", "struct@4000", "bm25@3000"}
BUDGETS = {c: int(c.split("@")[1][:4]) if "@" in c else 3000 for c in CELLS}
BUDGETS["oracle"] = 0  # no budget: 16's oracle


# -- corpus -----------------------------------------------------------------------------------


def load_corpus() -> tuple[str, dict[str, str]]:
    """(law text, raw table texts of the 87 tables that have no clean version)."""
    law = extract_text(ONE / "corpus/law.pdf")
    raw = {
        p.stem: p.read_text("utf-8")
        for p in sorted((ONE / "corpus/tables").glob("*.txt"))
        if p.stem not in TABLES
    }
    return law, raw


def subtitles(folder: Path = ONE / "corpus/tables") -> dict[str, str]:
    """The parenthesised line(s) right under each clean table's title in the PDF text."""
    out = {}
    for tid in TABLES:
        lines = (folder / f"{tid}.txt").read_text("utf-8").splitlines()
        if len(lines) < 2 or not lines[1].lstrip().startswith(("（", "(")):
            continue
        buf = []
        for ln in lines[1:4]:
            buf.append(ln.strip())
            if ")" in ln or "）" in ln:
                break
        out[tid] = "".join(buf)
    return out


def tables_view(*, sub: bool) -> dict[str, dict]:
    """TABLES, with the subtitle appended to each title when `sub`."""
    if not sub:
        return TABLES
    st = subtitles()
    return {tid: {**t, "title": f"{t['title']}{st.get(tid, '')}"} for tid, t in TABLES.items()}


def struct_chunks(law: str, raw: dict[str, str], *, sub: bool) -> list[Chunk]:
    return (
        article_chunks(parse_articles(law), max_len=1000)
        + kv_row_chunks(tables_view(sub=sub), row_key)
        + table_row_chunks(raw)
    )


def clean_table_block(tid: str, tables: dict[str, dict]) -> Chunk:
    """A clean table as markdown-ish lines, one label per line (`<tid>|<row key>`, `<tid>|註`)."""
    t = tables[tid]
    lines: list[tuple[str, str | None]] = [(f"{tid} {t['title'].removeprefix(tid).strip()}", None)]
    lines.append((" | ".join(t["names"]), None))
    for r in t["rows"]:
        lines.append((" | ".join(str(x) for x in r), f"{tid}|{row_key(tid, r)}"))
    if t.get("notes"):
        lines.append((t["notes"], f"{tid}|註"))
    return _block(lines, tid)


def short_names(names: list[str]) -> list[str]:
    """Column names cut to what tells them apart: names split on '／', a group is the columns that
    share the first part, and each column keeps the parts not shared by its whole group
    ("同一導線管或電纜內之載流導線數／3以下／安培容量（A）" -> "3以下"). A column alone in its group
    keeps its full name; when two short names clash, every clashing one gets its group back
    (names that are the same in the source, 表四○'s four 分路額定(A)設置, stay as they are)."""
    segs = [n.split("／") for n in names]
    out = []
    for s in segs:
        group = [t for t in segs if t[0] == s[0]]
        if len(group) == 1:
            out.append("／".join(s))
            continue
        keep = [x for i, x in enumerate(s) if not all(len(t) > i and t[i] == x for t in group)]
        out.append("／".join(keep) or "／".join(s))
    clash = {n for n in out if out.count(n) > 1}
    return [f"{segs[i][0]}／{n}" if n in clash and n != names[i] else n for i, n in enumerate(out)]


def kv_like(c: Chunk, tables: dict[str, dict]) -> Chunk:
    """A clean-table block (whole, or cut by `cut`) rewritten with the same lines in the same
    order: the title and full-column-name lines as they were, each row as `名=值；名=值` with
    short_names (empty cells skipped), the notes line as it was. Any other chunk passes through."""
    tid = c.meta.get("table")
    if tid not in tables or not c.labels or c.labels[0] == tid and len(c.labels) == 1:
        return c
    t = tables[tid]
    sn = short_names(t["names"])
    md = c.text.split("\n")
    n_rows = min(len(t["rows"]), max(0, len(md) - 2))
    lines: list[tuple[str, str | None]] = [(ln, None) for ln in md[:2]]
    for r in t["rows"][:n_rows]:
        kv = "；".join(f"{n}={v}" for n, v in zip(sn, r, strict=True) if str(v))
        lines.append((kv, f"{tid}|{row_key(tid, r)}"))
    if len(md) - 2 > len(t["rows"]):
        lines.append((t["notes"], f"{tid}|註"))
    labels = sorted({lab for _, lab in lines if lab} | ({tid} if tid in c.labels else set()))
    return Chunk("\n".join(s for s, _ in lines), labels, meta={"table": tid, "lines": lines})


def raw_table_block(tid: str, text: str) -> Chunk:
    return _block([(ln, tid) for ln in text.splitlines() if ln.strip()], tid)


def _block(lines: list[tuple[str, str | None]], tid: str) -> Chunk:
    labels = sorted({lab for _, lab in lines if lab} | {tid})
    return Chunk("\n".join(s for s, _ in lines), labels, meta={"table": tid, "lines": lines})


def cut(c: Chunk, budget: int) -> Chunk:
    """The leading lines of `c` that fit in `budget`; labels from the kept lines only. A chunk
    without per-line labels is cut by characters and keeps its labels (10's first-chunk rule)."""
    lines = c.meta.get("lines")
    if not lines:
        return Chunk(c.text[:budget], list(c.labels), c.start, c.end, dict(c.meta))
    kept: list[tuple[str, str | None]] = []
    used = 0
    for s, lab in lines:
        if used + len(s) + (1 if kept else 0) > budget:
            break
        used += len(s) + (1 if kept else 0)
        kept.append((s, lab))
    if not kept:  # one line longer than the budget
        kept = [(lines[0][0][:budget], lines[0][1])]
    labels = sorted({lab for _, lab in kept if lab})
    return Chunk("\n".join(s for s, _ in kept), labels, meta={"table": c.meta.get("table")})


def table_cards(
    raw: dict[str, str], tables: dict[str, dict]
) -> tuple[list[Chunk], dict[str, Chunk]]:
    """One card per table and the whole-table block it stands for. Card text: id + title, the
    column names, the row keys (clean tables); header lines + first token of each row (raw)."""
    cards: list[Chunk] = []
    blocks: dict[str, Chunk] = {}
    for tid, t in tables.items():
        keys = "、".join(row_key(tid, r) for r in t["rows"])
        title = t["title"].removeprefix(tid).strip()
        cards.append(
            Chunk(
                f"{tid} {title}\n欄：{'；'.join(t['names'])}\n列：{keys}",
                [f"card:{tid}"],
                meta={"table": tid},
            )
        )
        blocks[tid] = clean_table_block(tid, tables)
    rows = table_row_chunks(raw)
    by_tid: dict[str, list[Chunk]] = {}
    for c in rows:
        by_tid.setdefault(c.labels[0], []).append(c)
    for tid, text in raw.items():
        got = by_tid.get(tid)
        if got:
            header = got[0].text.rsplit("\n", 1)[0]
            keys = "、".join(c.text.rsplit("\n", 1)[1].split()[0] for c in got)
            body = f"{header}\n列：{keys}"
        else:
            body = text[:300]
        cards.append(Chunk(body, [f"card:{tid}"], meta={"table": tid}))
        blocks[tid] = raw_table_block(tid, text)
    return cards, blocks


# -- packing ----------------------------------------------------------------------------------


def budget_evidence(
    ranked: list[Chunk], *, budget: int, pack: bool = False
) -> tuple[str, set[str]]:
    """Whole units in rank order while they fit; the first unit cut when it alone does not fit.
    pack=False stops at the first unit that does not fit (10/16); pack=True skips it and goes on.
    Returns (text, labels in the evidence)."""
    return render(pack_units(ranked, budget=budget, pack=pack))


def render(kept: list[Chunk]) -> tuple[str, set[str]]:
    labels = {lab for c in kept for lab in c.labels}
    text = "\n\n".join(f"【段 {i + 1}】\n{c.text}" for i, c in enumerate(kept))
    return text, labels


def pack_units(ranked: list[Chunk], *, budget: int, pack: bool = False) -> list[Chunk]:
    """The units budget_evidence keeps, in order."""
    kept: list[Chunk] = []
    used = 0
    for c in ranked:
        if used + len(c.text) > budget:
            if not kept:
                kept.append(cut(c, budget))
                used = len(kept[0].text)
                if not pack:
                    break
                continue
            if pack:
                continue
            break
        used += len(c.text)
        kept.append(c)
    return kept


def expand(ranked: list[Chunk], blocks: dict[str, Chunk]) -> list[Chunk]:
    """Cards -> their whole tables (each table once); article chunks pass through."""
    out: list[Chunk] = []
    seen: set[str] = set()
    for c in ranked:
        tid = c.labels[0].removeprefix("card:") if c.labels[0].startswith("card:") else None
        if tid is None:
            out.append(c)
        elif tid not in seen:
            seen.add(tid)
            out.append(blocks[tid])
    return out


def interleave(rankings: list[list[int]]) -> list[int]:
    """Round-robin over rankings, first occurrence wins."""
    out: list[int] = []
    seen: set[int] = set()
    for i in range(max((len(r) for r in rankings), default=0)):
        for r in rankings:
            if i < len(r) and r[i] not in seen:
                seen.add(r[i])
                out.append(r[i])
    return out


def rrf(rankings: list[list[int]], *, k: int = RRF_K) -> list[int]:
    score: dict[int, float] = {}
    for r in rankings:
        for pos, i in enumerate(r):
            score[i] = score.get(i, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score, key=lambda i: (-score[i], i))


def gold_labels(q: dict) -> tuple[set[str], set[str]]:
    rows = {f"{t}|{k}" for t, k in q["gold"]["rows"]}
    arts = set(q["gold"]["articles"])
    return rows, arts


def needed_rows(q: dict) -> set[str]:
    """The gold rows the answer uses: a second band with nothing in it (c4-16, 6 kVA of
    receptacles) is in gold.rows but not needed (20, after 19's review)."""
    rows = [f"{t}|{k}" for t, k in q["gold"]["rows"]]
    inp = q.get("inputs") or {}
    return set(rows[:1] if inp.get("rest") == 0 or inp.get("band2") == 0 else rows)


def hit_stats(labels: set[str], q: dict) -> dict:
    """Needed rows and articles by label; a table counts when any of its labels is in the
    evidence."""
    _, arts = gold_labels(q)
    rows = needed_rows(q)
    tables = {r.split("|", 1)[0] for r in rows}
    seen_tables = {lab.split("|", 1)[0] for lab in labels}
    return {
        "rows_hit": len(rows & labels),
        "rows_total": len(rows),
        "tables_hit": len(tables & seen_tables),
        "tables_total": len(tables),
        "article_hit": bool(arts & labels),
    }


def rule_excerpt(body: str, needles: list[str], *, chars: int = EXCERPT_CHARS) -> str:
    """`chars` characters of the article around the first needle found (the whole body if short)."""
    if len(body) <= chars:
        return body
    at = min((body.find(n) for n in needles if n in body), default=0)
    lo = max(0, at - chars // 3)
    return body[lo : lo + chars]


# -- the evidence builder -----------------------------------------------------------------------


class Evidence:
    """Rankings for every requested cell, built once (embeddings through the cache)."""

    def __init__(
        self,
        questions: list[dict],
        *,
        cells: list[str],
        embed,
        cache,
        subqueries: dict[str, list[str]] | None = None,
        reranker=None,
    ):
        law, raw = load_corpus()
        self.tables = tables_view(sub=True)
        self.articles = {str(a.num): a for a in parse_articles(law)}
        self.ranked: dict[str, dict[str, list[Chunk]]] = {}
        need_cards = any(c.startswith(("cards", "decomp")) for c in cells)
        for sub in (False, True):
            group = [c for c in cells if c.startswith(("struct", "bm25")) and (c in NOSUB) != sub]
            if not group:
                continue
            chunks = struct_chunks(law, raw, sub=sub)
            index = DenseIndex.build(chunks, embed, cache)
            dense = {q["id"]: index.search(q["question"], DEPTH) for q in questions}
            for c in group:
                if c != "bm25@3000":
                    self.ranked[c] = {k: [chunks[i] for i in v] for k, v in dense.items()}
            if "bm25@3000" in group:
                bm = BM25Index(chunks)
                self.ranked["bm25@3000"] = {
                    q["id"]: [
                        chunks[i]
                        for i in rrf(
                            [
                                index.search(q["question"], len(chunks)),
                                bm.search(q["question"], len(chunks)),
                            ]
                        )[:DEPTH]
                    ]
                    for q in questions
                }
            cache.save()
        if need_cards:
            cards, blocks = table_cards(raw, self.tables)
            units = article_chunks(parse_articles(law), max_len=1000) + cards
            index = DenseIndex.build(units, embed, cache)
            logger.info("cards", extra={"ctx": {"units": len(units), "cards": len(cards)}})
            dense = {q["id"]: index.search(q["question"], DEPTH) for q in questions}
            if any(c.startswith("cards") and c != "cards+rr" for c in cells):
                self.ranked["cards"] = {
                    k: expand([units[i] for i in v], blocks) for k, v in dense.items()
                }
            if "cards-kv@3000" in cells:
                kv_blocks = {tid: kv_like(b, self.tables) for tid, b in blocks.items()}
                self.ranked["cards-kv@3000"] = {
                    k: expand([units[i] for i in v], kv_blocks) for k, v in dense.items()
                }
            if "cards+rr" in cells:
                assert reranker is not None, "cards+rr needs a reranker"
                out = {}
                for q in questions:
                    top = [units[i] for i in dense[q["id"]][:RERANK_DEPTH]]
                    order, _ = reranker.rerank(q["question"], top)
                    out[q["id"]] = expand([top[i] for i in order], blocks)
                reranker.save()
                self.ranked["cards+rr"] = out
            if any(c.startswith("decomp") for c in cells):
                assert subqueries is not None, "decomp needs subqueries"
                out = {}
                for q in questions:
                    queries = [q["question"], *subqueries.get(q["id"], [])]
                    merged = interleave([index.search(s, DEPTH) for s in queries])
                    out[q["id"]] = expand([units[i] for i in merged[:DEPTH]], blocks)
                self.ranked["decomp"] = out
            cache.save()

    def oracle(self, q: dict) -> str:
        """16's oracle: the gold rule excerpt + the gold rows as kv lines."""
        parts = []
        needles = [t for t, _ in q["gold"]["rows"]]
        for num in q["gold"]["articles"]:
            a = self.articles.get(num)
            if a:
                parts.append(f"【第 {num} 條】\n{rule_excerpt(a.body, needles)}")
        for tid, key in q["gold"]["rows"]:
            t = TABLES[tid]
            row = next((r for r in t["rows"] if row_key(tid, r) == key), None)
            if key == "註":
                parts.append(f"【{t['title']}】\n{t.get('notes', '')}")
            elif row is not None:
                parts.append(f"【{t['title']}】\n{kv_row_line(t['names'], row)}")
            else:  # an off-table key (4 hp): the table's own rows around it
                parts.append(
                    f"【{t['title']}】\n"
                    + "\n".join(kv_row_line(t["names"], r) for r in t["rows"][:6])
                )
        return "\n\n".join(parts)

    def for_question(self, cell: str, q: dict) -> tuple[str, dict]:
        if cell == "oracle":
            rows, _ = gold_labels(q)
            tables = {r.split("|", 1)[0] for r in rows}
            return self.oracle(q), {
                "rows_hit": len(rows),
                "rows_total": len(rows),
                "tables_hit": len(tables),
                "tables_total": len(tables),
                "article_hit": True,
            }
        if cell == "cards-kv":  # cards' own choice of units, then the clean tables rewritten
            kept = pack_units(self.ranked["cards"][q["id"]], budget=BUDGETS[cell], pack=True)
            text, labels = render([kv_like(c, self.tables) for c in kept])
            return text, hit_stats(labels, q)
        base = cell.split("@")[0] if cell.startswith(("cards@", "decomp@")) else cell
        text, labels = budget_evidence(
            self.ranked[base][q["id"]], budget=BUDGETS[cell], pack=cell not in STOP_AT_MISFIT
        )
        return text, hit_stats(labels, q)
