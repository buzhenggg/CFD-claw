#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
BLANK_RE = re.compile(r"\s+")
DESIGNATION_RE = re.compile(r"\b([A-Za-z]{1,8})[\s._-]*([0-9]{1,6}[A-Za-z]?)\b")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
CACHE_SCHEMA_VERSION = 2
DEFAULT_CHUNK_LINES = 120
DEFAULT_OVERLAP_LINES = 20
DEFAULT_CANDIDATE_LIMIT = 1200
ENGINE_DESIGN_QUERY_EXPANSION = (
    "航空发动机 总体设计 性能 推力 功率 耗油率 推重比 可靠性 维修性 寿命 "
    "使用限制 状态监测 强度 振动 转子动力学 控制 燃油系统 空气系统 传热 冷却 "
    "compressor combustor turbine nozzle reliability maintainability life"
)


@dataclass
class Chunk:
    path: Path
    heading: str
    start_line: int
    end_line: int
    text: str
    token_count: int = 0
    tokens: list[str] = field(default_factory=list)
    tf: Counter[str] = field(default_factory=Counter)

    @property
    def length(self) -> int:
        return max(self.token_count or len(self.tokens), 1)

    def relative_path(self, root: Path) -> str:
        return str(self.path.resolve().relative_to(root.resolve()))


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search Markdown files in a local RAG-data folder and return the most relevant chunks."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing Markdown knowledge files.")
    parser.add_argument("--query", required=True, help="Query to search for.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of hits to print.")
    parser.add_argument("--chunk-lines", type=int, default=DEFAULT_CHUNK_LINES, help="Lines per chunk.")
    parser.add_argument("--overlap-lines", type=int, default=DEFAULT_OVERLAP_LINES, help="Line overlap between chunks.")
    parser.add_argument("--max-snippet-chars", type=int, default=280, help="Maximum snippet length.")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")
    parser.add_argument("--use-cache", action="store_true", help="Cache and reuse the local Markdown index.")
    parser.add_argument("--cache-dir", help="Optional cache directory. Defaults to <project>/.clawd/cache.")
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=DEFAULT_CANDIDATE_LIMIT,
        help="Maximum candidate chunks to rerank when using the SQLite index.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    started_at = time.perf_counter()
    timings: dict[str, float] = {}
    args = parse_args(argv or sys.argv[1:])
    requested_data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir = resolve_data_dir(requested_data_dir)
    query = args.query.strip()

    if not query:
        print("error: --query must not be empty", file=sys.stderr)
        return 2
    if data_dir is None:
        print(f"error: data directory not found: {requested_data_dir}", file=sys.stderr)
        return 2

    files_started = time.perf_counter()
    files = sorted(data_dir.rglob("*.md"))
    timings["file_scan_ms"] = elapsed_ms(files_started)
    if not files:
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "query": query,
                        "data_dir": str(data_dir),
                        "markdown_files_scanned": 0,
                        "chunks_indexed": 0,
                        "candidate_chunks": 0,
                        "cache": {"enabled": bool(args.use_cache), "hit": False, "path": None, "type": "sqlite"},
                        "timings": finish_timings(timings, started_at),
                        "hits": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        print(f"No Markdown files found under: {data_dir}")
        return 0

    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
    candidate_limit = max(args.candidate_limit, max(args.top_k, 1))
    if args.use_cache:
        hits, chunks_indexed, cache_info, candidate_count = search_with_sqlite_cache(
            files=files,
            data_dir=data_dir,
            query=query,
            chunk_lines=args.chunk_lines,
            overlap_lines=args.overlap_lines,
            candidate_limit=candidate_limit,
            cache_dir=cache_dir,
            timings=timings,
        )
    else:
        build_started = time.perf_counter()
        chunks, document_frequency, avg_length = build_index(
            files,
            chunk_lines=args.chunk_lines,
            overlap_lines=args.overlap_lines,
        )
        timings["index_build_ms"] = elapsed_ms(build_started)
        search_started = time.perf_counter()
        hits = search_chunks(
            chunks,
            query,
            document_frequency=document_frequency,
            avg_length=avg_length,
        )
        timings["search_ms"] = elapsed_ms(search_started)
        chunks_indexed = len(chunks)
        cache_info = {"enabled": False, "hit": False, "path": None, "type": "memory"}
        candidate_count = len(chunks)

    selected_hits = hits[: max(args.top_k, 1)]
    finished_timings = finish_timings(timings, started_at)
    if args.format == "json":
        print(
            json.dumps(
                build_json_response(
                    query=query,
                    data_dir=data_dir,
                    files=files,
                    chunks_indexed=chunks_indexed,
                    hits=selected_hits,
                    max_snippet_chars=args.max_snippet_chars,
                    cache_info=cache_info,
                    timings=finished_timings,
                    candidate_count=candidate_count,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"Query: {query}")
    print(f"Data directory: {data_dir}")
    print(f"Markdown files scanned: {len(files)}")
    print(f"Chunks indexed: {chunks_indexed}")
    print(f"Candidate chunks reranked: {candidate_count}")
    if cache_info.get("enabled"):
        print(
            f"Index cache: {'hit' if cache_info.get('hit') else 'miss'} "
            f"{cache_info.get('type') or ''} ({cache_info.get('path')})"
        )
    print(f"Retrieval time: {finished_timings.get('total_ms', 0):.1f} ms")
    print()

    if not hits:
        print("No relevant chunks found.")
        return 0

    print("Top hits:")
    for index, hit in enumerate(selected_hits, start=1):
        chunk = hit.chunk
        snippet = make_snippet(chunk.text, args.max_snippet_chars)
        rel_path = chunk.relative_path(data_dir.parent)
        print(
            f"{index}. score={hit.score:.3f} file={rel_path} "
            f"lines={chunk.start_line}-{chunk.end_line} heading={chunk.heading or '-'}"
        )
        print(f"   snippet: {snippet}")
    return 0


def build_json_response(
    *,
    query: str,
    data_dir: Path,
    files: Sequence[Path],
    chunks_indexed: int,
    hits: Sequence[SearchHit],
    max_snippet_chars: int,
    cache_info: dict | None = None,
    timings: dict[str, float] | None = None,
    candidate_count: int = 0,
) -> dict:
    return {
        "query": query,
        "data_dir": str(data_dir),
        "markdown_files_scanned": len(files),
        "chunks_indexed": chunks_indexed,
        "candidate_chunks": candidate_count,
        "cache": cache_info or {"enabled": False, "hit": False, "path": None},
        "timings": timings or {},
        "hits": [
            {
                "rank": index,
                "score": round(hit.score, 6),
                "file": hit.chunk.relative_path(data_dir.parent),
                "start_line": hit.chunk.start_line,
                "end_line": hit.chunk.end_line,
                "heading": hit.chunk.heading or "",
                "snippet": make_snippet(hit.chunk.text, max_snippet_chars),
            }
            for index, hit in enumerate(hits, start=1)
        ],
    }


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


def finish_timings(timings: dict[str, float], started_at: float) -> dict[str, float]:
    finished = dict(timings)
    finished["total_ms"] = elapsed_ms(started_at)
    return finished


def build_index_with_cache(
    *,
    files: Sequence[Path],
    data_dir: Path,
    chunk_lines: int,
    overlap_lines: int,
    use_cache: bool,
    cache_dir: Path | None = None,
) -> tuple[list[Chunk], Counter[str], float, dict]:
    """Compatibility helper for tests or callers that still expect an in-memory index."""
    cache_info = {"enabled": use_cache, "hit": False, "path": None}
    chunks, document_frequency, avg_length = build_index(
        files,
        chunk_lines=chunk_lines,
        overlap_lines=overlap_lines,
    )
    if use_cache:
        cache_info["path"] = str(get_cache_path(data_dir=data_dir, chunk_lines=chunk_lines, overlap_lines=overlap_lines, cache_dir=cache_dir))
        cache_info["type"] = "sqlite"
    return chunks, document_frequency, avg_length, cache_info


def get_cache_path(
    *,
    data_dir: Path,
    chunk_lines: int,
    overlap_lines: int,
    cache_dir: Path | None = None,
) -> Path:
    target_dir = cache_dir or data_dir.parent / ".clawd" / "cache"
    key = json.dumps(
        {
            "schema": CACHE_SCHEMA_VERSION,
            "data_dir": str(data_dir.resolve()),
            "chunk_lines": chunk_lines,
            "overlap_lines": overlap_lines,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return target_dir / f"rag_index_{digest}.sqlite"


def manifest_hash(manifest: list[dict]) -> str:
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def search_with_sqlite_cache(
    *,
    files: Sequence[Path],
    data_dir: Path,
    query: str,
    chunk_lines: int,
    overlap_lines: int,
    candidate_limit: int,
    cache_dir: Path | None,
    timings: dict[str, float],
) -> tuple[list[SearchHit], int, dict, int]:
    manifest_started = time.perf_counter()
    manifest = build_file_manifest(files, data_dir)
    expected_manifest_hash = manifest_hash(manifest)
    timings["manifest_ms"] = elapsed_ms(manifest_started)

    cache_path = get_cache_path(
        data_dir=data_dir,
        chunk_lines=chunk_lines,
        overlap_lines=overlap_lines,
        cache_dir=cache_dir,
    )
    cache_info = {
        "enabled": True,
        "hit": False,
        "path": str(cache_path),
        "type": "sqlite",
        "candidate_limit": candidate_limit,
    }

    valid_started = time.perf_counter()
    is_valid = sqlite_cache_is_valid(
        cache_path,
        data_dir=data_dir,
        chunk_lines=chunk_lines,
        overlap_lines=overlap_lines,
        expected_manifest_hash=expected_manifest_hash,
    )
    timings["cache_validate_ms"] = elapsed_ms(valid_started)
    if not is_valid:
        build_started = time.perf_counter()
        build_sqlite_index(
            cache_path,
            files=files,
            data_dir=data_dir,
            manifest=manifest,
            expected_manifest_hash=expected_manifest_hash,
            chunk_lines=chunk_lines,
            overlap_lines=overlap_lines,
        )
        timings["index_build_ms"] = elapsed_ms(build_started)
    else:
        cache_info["hit"] = True

    search_started = time.perf_counter()
    with sqlite3.connect(cache_path) as conn:
        conn.row_factory = sqlite3.Row
        hits, chunk_count, candidate_count = search_sqlite_index(
            conn,
            data_dir=data_dir,
            query=query,
            candidate_limit=candidate_limit,
        )
    timings["search_ms"] = elapsed_ms(search_started)
    cache_info["hit"] = bool(is_valid)
    return hits, chunk_count, cache_info, candidate_count


def sqlite_cache_is_valid(
    cache_path: Path,
    *,
    data_dir: Path,
    chunk_lines: int,
    overlap_lines: int,
    expected_manifest_hash: str,
) -> bool:
    if not cache_path.exists():
        return False
    try:
        with sqlite3.connect(cache_path) as conn:
            rows = dict(conn.execute("SELECT key, value FROM meta"))
    except Exception:
        return False
    return (
        rows.get("schema_version") == str(CACHE_SCHEMA_VERSION)
        and rows.get("data_dir") == str(data_dir.resolve())
        and rows.get("chunk_lines") == str(chunk_lines)
        and rows.get("overlap_lines") == str(overlap_lines)
        and rows.get("manifest_hash") == expected_manifest_hash
    )


def build_sqlite_index(
    cache_path: Path,
    *,
    files: Sequence[Path],
    data_dir: Path,
    manifest: list[dict],
    expected_manifest_hash: str,
    chunk_lines: int,
    overlap_lines: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    document_frequency: Counter[str] = Counter()
    chunk_count = 0
    total_length = 0

    with sqlite3.connect(tmp_path) as conn:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        create_sqlite_schema(conn)

        chunk_rows: list[tuple] = []
        term_rows: list[tuple] = []

        def flush_chunks() -> None:
            if chunk_rows:
                conn.executemany(
                    "INSERT INTO chunks(id, path, heading, start_line, end_line, text, length) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    chunk_rows,
                )
                chunk_rows.clear()

        def flush_terms() -> None:
            if term_rows:
                conn.executemany(
                    "INSERT INTO chunk_terms(token, chunk_id, tf) VALUES (?, ?, ?)",
                    term_rows,
                )
                term_rows.clear()

        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for chunk in split_markdown(path, text, chunk_lines=chunk_lines, overlap_lines=overlap_lines):
                chunk_count += 1
                tokens = tokenize(chunk.text)
                token_counts = Counter(tokens)
                length = max(len(tokens), 1)
                total_length += length
                relative_path = str(chunk.path.resolve().relative_to(data_dir.resolve()))
                chunk_rows.append(
                    (
                        chunk_count,
                        relative_path,
                        chunk.heading,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.text,
                        length,
                    )
                )
                for token, tf in token_counts.items():
                    term_rows.append((token, chunk_count, int(tf)))
                for token in token_counts:
                    document_frequency[token] += 1
                if len(chunk_rows) >= 5000:
                    flush_chunks()
                if len(term_rows) >= 50000:
                    flush_terms()

        flush_chunks()
        flush_terms()

        avg_length = total_length / max(chunk_count, 1)
        total_chunks = max(chunk_count, 1)
        conn.executemany(
            "INSERT INTO term_stats(token, df, idf) VALUES (?, ?, ?)",
            [
                (
                    token,
                    int(df),
                    math.log(1 + (total_chunks - df + 0.5) / (df + 0.5)),
                )
                for token, df in document_frequency.items()
            ],
        )
        conn.executemany(
            "INSERT INTO manifest(path, mtime_ns, size) VALUES (?, ?, ?)",
            [(item["path"], item["mtime_ns"], item["size"]) for item in manifest],
        )
        conn.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            [
                ("schema_version", str(CACHE_SCHEMA_VERSION)),
                ("data_dir", str(data_dir.resolve())),
                ("chunk_lines", str(chunk_lines)),
                ("overlap_lines", str(overlap_lines)),
                ("manifest_hash", expected_manifest_hash),
                ("chunk_count", str(chunk_count)),
                ("avg_length", str(avg_length)),
            ],
        )
        create_sqlite_indexes(conn)
        conn.execute("ANALYZE")

    tmp_path.replace(cache_path)


def create_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE manifest (
            path TEXT PRIMARY KEY,
            mtime_ns INTEGER NOT NULL,
            size INTEGER NOT NULL
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            heading TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            text TEXT NOT NULL,
            length INTEGER NOT NULL
        );
        CREATE TABLE chunk_terms (
            token TEXT NOT NULL,
            chunk_id INTEGER NOT NULL,
            tf INTEGER NOT NULL
        );
        CREATE TABLE term_stats (
            token TEXT PRIMARY KEY,
            df INTEGER NOT NULL,
            idf REAL NOT NULL
        );
        """
    )


def create_sqlite_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX idx_chunk_terms_token ON chunk_terms(token);
        CREATE INDEX idx_chunk_terms_chunk ON chunk_terms(chunk_id);
        """
    )


def search_sqlite_index(
    conn: sqlite3.Connection,
    *,
    data_dir: Path,
    query: str,
    candidate_limit: int,
) -> tuple[list[SearchHit], int, int]:
    query_tokens = tokenize(expand_query_for_retrieval(query))
    if not query_tokens:
        chunk_count = int(fetch_meta(conn).get("chunk_count", "0"))
        return [], chunk_count, 0

    unique_query_tokens = list(dict.fromkeys(query_tokens))
    meta = fetch_meta(conn)
    chunk_count = int(meta.get("chunk_count", "0") or 0)
    avg_length = float(meta.get("avg_length", "1.0") or 1.0)
    candidate_rows = fetch_candidate_rows(conn, unique_query_tokens, candidate_limit)
    candidate_ids = [int(row["chunk_id"]) for row in candidate_rows]
    if not candidate_ids:
        return [], chunk_count, 0

    chunks = fetch_chunks_by_id(conn, data_dir=data_dir, chunk_ids=candidate_ids)
    term_frequency_by_chunk: dict[int, Counter[str]] = {chunk_id: Counter() for chunk_id in candidate_ids}
    document_frequency = fetch_document_frequency(conn, unique_query_tokens)
    fill_candidate_term_frequencies(
        conn,
        query_tokens=unique_query_tokens,
        candidate_ids=candidate_ids,
        term_frequency_by_chunk=term_frequency_by_chunk,
    )

    query_compact = compact_text(query)
    hits: list[SearchHit] = []
    for chunk_id in candidate_ids:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            continue
        chunk.tf = term_frequency_by_chunk.get(chunk_id, Counter())
        score = bm25_score(
            chunk=chunk,
            query_tokens=unique_query_tokens,
            document_frequency=document_frequency,
            total_chunks=max(chunk_count, 1),
            avg_length=avg_length,
        )
        score += heuristic_boost(chunk, unique_query_tokens, query_compact)
        if score > 0:
            hits.append(SearchHit(chunk=chunk, score=score))

    hits.sort(
        key=lambda item: (
            round(item.score, 6),
            item.chunk.start_line * -1,
            item.chunk.path.name,
        ),
        reverse=True,
    )
    return hits, chunk_count, len(candidate_ids)


def fetch_meta(conn: sqlite3.Connection) -> dict[str, str]:
    return {str(row["key"]): str(row["value"]) for row in conn.execute("SELECT key, value FROM meta")}


def fetch_document_frequency(conn: sqlite3.Connection, query_tokens: Sequence[str]) -> Counter[str]:
    document_frequency: Counter[str] = Counter()
    if not query_tokens:
        return document_frequency
    placeholders = ", ".join("?" for _ in query_tokens)
    for row in conn.execute(f"SELECT token, df FROM term_stats WHERE token IN ({placeholders})", list(query_tokens)):
        document_frequency[str(row["token"])] = int(row["df"])
    return document_frequency


def fill_candidate_term_frequencies(
    conn: sqlite3.Connection,
    *,
    query_tokens: Sequence[str],
    candidate_ids: Sequence[int],
    term_frequency_by_chunk: dict[int, Counter[str]],
) -> None:
    if not query_tokens or not candidate_ids:
        return
    candidate_id_set = set(candidate_ids)
    for token in query_tokens:
        rows = conn.execute("SELECT chunk_id, tf FROM chunk_terms WHERE token = ?", (token,))
        for row in rows:
            chunk_id = int(row["chunk_id"])
            if chunk_id in candidate_id_set:
                term_frequency_by_chunk[chunk_id][token] = int(row["tf"])


def fetch_candidate_rows(
    conn: sqlite3.Connection,
    query_tokens: Sequence[str],
    candidate_limit: int,
) -> list[sqlite3.Row]:
    placeholders = ", ".join("?" for _ in query_tokens)
    sql = f"""
        SELECT chunk_terms.chunk_id AS chunk_id,
               SUM(chunk_terms.tf * term_stats.idf) AS rough_score
        FROM chunk_terms
        JOIN term_stats ON term_stats.token = chunk_terms.token
        WHERE chunk_terms.token IN ({placeholders})
        GROUP BY chunk_terms.chunk_id
        ORDER BY rough_score DESC
        LIMIT ?
    """
    return list(conn.execute(sql, [*query_tokens, int(candidate_limit)]))


def fetch_chunks_by_id(
    conn: sqlite3.Connection,
    *,
    data_dir: Path,
    chunk_ids: Sequence[int],
) -> dict[int, Chunk]:
    chunks: dict[int, Chunk] = {}
    if not chunk_ids:
        return chunks
    for offset in range(0, len(chunk_ids), 800):
        batch = list(chunk_ids[offset : offset + 800])
        placeholders = ", ".join("?" for _ in batch)
        rows = conn.execute(
            f"""
            SELECT id, path, heading, start_line, end_line, text, length
            FROM chunks
            WHERE id IN ({placeholders})
            """,
            batch,
        )
        for row in rows:
            chunk_id = int(row["id"])
            chunks[chunk_id] = Chunk(
                path=(data_dir / str(row["path"])).resolve(),
                heading=str(row["heading"]),
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                text=str(row["text"]),
                token_count=int(row["length"]),
            )
    return chunks


def build_file_manifest(files: Sequence[Path], data_dir: Path) -> list[dict]:
    manifest: list[dict] = []
    for path in files:
        stat = path.stat()
        manifest.append(
            {
                "path": str(path.resolve().relative_to(data_dir.resolve())),
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        )
    return manifest


def resolve_data_dir(requested: Path, *, script_path: Path | None = None) -> Path | None:
    requested = requested.expanduser().resolve()
    if requested.exists() and requested.is_dir():
        return requested

    source = (script_path or Path(__file__)).expanduser().resolve()
    for parent in source.parents:
        candidate = parent / "RAG-data"
        if candidate.exists() and candidate.is_dir():
            print(
                f"warning: data directory not found: {requested}; using {candidate.resolve()}",
                file=sys.stderr,
            )
            return candidate.resolve()

    return None


def build_index(
    paths: Iterable[Path],
    *,
    chunk_lines: int,
    overlap_lines: int,
) -> tuple[list[Chunk], Counter[str], float]:
    chunks: list[Chunk] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(split_markdown(path, text, chunk_lines=chunk_lines, overlap_lines=overlap_lines))

    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        chunk.tokens = tokenize(chunk.text)
        chunk.tf = Counter(chunk.tokens)
        for token in set(chunk.tokens):
            document_frequency[token] += 1

    avg_length = sum(chunk.length for chunk in chunks) / max(len(chunks), 1)
    return chunks, document_frequency, avg_length


def search_chunks(
    chunks: Sequence[Chunk],
    query: str,
    *,
    document_frequency: Counter[str],
    avg_length: float,
) -> list[SearchHit]:
    query_tokens = tokenize(expand_query_for_retrieval(query))
    if not query_tokens:
        return []

    total_chunks = max(len(chunks), 1)
    unique_query_tokens = list(dict.fromkeys(query_tokens))
    query_compact = compact_text(query)
    hits: list[SearchHit] = []

    for chunk in chunks:
        score = bm25_score(
            chunk=chunk,
            query_tokens=unique_query_tokens,
            document_frequency=document_frequency,
            total_chunks=total_chunks,
            avg_length=avg_length,
        )
        score += heuristic_boost(chunk, unique_query_tokens, query_compact)
        if score > 0:
            hits.append(SearchHit(chunk=chunk, score=score))

    hits.sort(
        key=lambda item: (
            round(item.score, 6),
            item.chunk.start_line * -1,
            item.chunk.path.name,
        ),
        reverse=True,
    )
    return hits


def bm25_score(
    *,
    chunk: Chunk,
    query_tokens: Sequence[str],
    document_frequency: Counter[str],
    total_chunks: int,
    avg_length: float,
) -> float:
    score = 0.0
    k1 = 1.5
    b = 0.75

    for token in query_tokens:
        tf = chunk.tf.get(token, 0)
        if tf <= 0:
            continue
        df = document_frequency.get(token, 0)
        idf = math.log(1 + (total_chunks - df + 0.5) / (df + 0.5))
        denom = tf + k1 * (1 - b + b * (chunk.length / max(avg_length, 1.0)))
        score += idf * ((tf * (k1 + 1)) / max(denom, 1e-9))
    return score


def heuristic_boost(chunk: Chunk, query_tokens: Sequence[str], query_compact: str) -> float:
    if not query_tokens:
        return 0.0

    lowered_text = chunk.text.lower()
    compact_chunk = compact_text(chunk.text)
    heading = chunk.heading.lower()
    compact_heading = compact_text(chunk.heading)
    compact_designation_chunk = compact_designation_text(chunk.text)
    compact_designation_heading = compact_designation_text(chunk.heading)
    path_str = str(chunk.path).lower()
    boost = 0.0

    joined_query = " ".join(query_tokens)
    if joined_query and joined_query in lowered_text:
        boost += 1.5
    if query_compact and query_compact in compact_chunk:
        boost += 6.0
    if query_compact and query_compact in compact_heading:
        boost += 6.0

    matched_terms = sum(1 for token in query_tokens if token in chunk.tf)
    if matched_terms == len(query_tokens):
        boost += 2.5
    else:
        boost += matched_terms * 0.3

    designation_tokens = [
        compact_designation_token(token) or token
        for token in query_tokens
        if has_designation_shape(token)
    ]
    for token in query_tokens:
        compact_token = compact_designation_token(token) or token
        if token in heading:
            boost += 0.7
        if token in path_str:
            boost += 0.4
        if has_designation_shape(token) and token in heading:
            boost += 3.0
        if has_designation_shape(token) and compact_token in compact_designation_chunk:
            boost += 1.2
    for token in designation_tokens:
        if token in compact_designation_heading:
            boost += 25.0
        elif token in compact_designation_chunk:
            boost += 15.0
        else:
            boost -= 8.0

    query_designation_compact = compact_designation_text(query_compact)
    if appears_exact_designation(query_designation_compact, compact_designation_chunk):
        boost += 3.0
    boost -= image_noise_penalty(chunk.text)

    return boost


def appears_exact_designation(query_compact: str, compact_chunk: str) -> bool:
    if not query_compact:
        return False
    has_digit = any(char.isdigit() for char in query_compact)
    has_alpha = any(char.isalpha() for char in query_compact)
    if has_digit and has_alpha and query_compact in compact_chunk:
        return True
    return False


def has_designation_shape(token: str) -> bool:
    return any(char.isdigit() for char in token) and any(char.isalpha() for char in token)


def image_noise_penalty(text: str) -> float:
    summary_count = text.count("图片摘要")
    image_count = text.count("![](")
    caption_count = text.count("系统图")
    return summary_count * 2.2 + image_count * 1.2 + caption_count * 0.4


def split_markdown(path: Path, text: str, *, chunk_lines: int, overlap_lines: int) -> list[Chunk]:
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    step = max(chunk_lines - overlap_lines, 1)

    sections: list[tuple[str, int, int]] = []
    current_heading = path.stem
    section_start = 1

    for line_number, line in enumerate(lines, start=1):
        match = HEADING_RE.match(line)
        if not match:
            continue

        heading_text = clean_heading(match.group(1)) or path.stem
        if line_number > section_start:
            sections.append((current_heading, section_start, line_number - 1))
        current_heading = heading_text
        section_start = line_number

    sections.append((current_heading, section_start, len(lines)))

    for heading, start_line, end_line in sections:
        section_lines = lines[start_line - 1 : end_line]
        start_index = 0

        while start_index < len(section_lines):
            end_index = min(start_index + chunk_lines, len(section_lines))
            chunk_start_line = start_line + start_index
            chunk_end_line = start_line + end_index - 1
            chunk_text = "\n".join(section_lines[start_index:end_index]).strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        path=path,
                        heading=heading,
                        start_line=chunk_start_line,
                        end_line=chunk_end_line,
                        text=chunk_text,
                    )
                )

            if end_index >= len(section_lines):
                break
            start_index += step

    return chunks


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = []
    for token in WORD_RE.findall(lowered):
        tokens.append(token)
        compact_designation = compact_designation_token(token)
        if compact_designation and compact_designation != token:
            tokens.append(compact_designation)

    for prefix, suffix in DESIGNATION_RE.findall(lowered):
        compact_designation = compact_designation_token(prefix + suffix)
        if compact_designation:
            tokens.append(compact_designation)

    for span in CJK_RE.findall(text):
        compact_span = compact_text(span)
        if not compact_span:
            continue

        if len(compact_span) <= 4:
            tokens.append(compact_span)

        max_n = 3
        for n in range(2, max_n + 1):
            if len(compact_span) < n:
                continue
            for index in range(len(compact_span) - n + 1):
                tokens.append(compact_span[index : index + n])

    return tokens


def expand_query_for_retrieval(query: str) -> str:
    lowered = query.lower()
    compact = compact_text(query)
    engine_related = (
        "发动机" in query
        or "推进" in query
        or "engine" in lowered
        or "propulsion" in lowered
        or "turbine" in lowered
    )
    design_related = (
        "设计" in query
        or "指标" in query
        or "方案" in query
        or "要求" in query
        or "好的" in query
        or "好" in query
        or "如何" in query
        or "怎么" in query
        or "how" in lowered
        or "design" in lowered
        or "good" in lowered
    )
    exact_designation = bool(DESIGNATION_RE.search(lowered)) or any(token for token in WORD_RE.findall(lowered) if has_designation_shape(token))
    if engine_related and design_related and not exact_designation and "航空发动机" not in compact:
        return f"{query} {ENGINE_DESIGN_QUERY_EXPANSION}"
    if engine_related and design_related and not exact_designation:
        return f"{query} {ENGINE_DESIGN_QUERY_EXPANSION}"
    return query


def compact_text(text: str) -> str:
    return BLANK_RE.sub("", text.lower())


def compact_designation_token(token: str) -> str:
    compact = NON_ALNUM_RE.sub("", token.lower())
    if has_designation_shape(compact):
        return compact
    return ""


def compact_designation_text(text: str) -> str:
    return NON_ALNUM_RE.sub("", text.lower())


def clean_heading(text: str) -> str:
    return BLANK_RE.sub(" ", text.strip())


def make_snippet(text: str, max_chars: int) -> str:
    kept_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("图片摘要："):
            continue
        if "![](" in line:
            continue
        kept_lines.append(line)

    cleaned = BLANK_RE.sub(" ", " ".join(kept_lines)).strip()
    if not cleaned:
        cleaned = BLANK_RE.sub(" ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())
