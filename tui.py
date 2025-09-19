# netwatch_tui.py
"""
Netwatch TUI — read-only terminal UI that continuously renders loss stats and approximate route groups
from the SQLite database produced by the agent.

Features (MVP):
- Read-only SQLite (WAL) connection with retry-on-busy
- Rolling windows: 24h, 6h, 1h, 5m
- Sort by 5m loss desc; tie-break by 1h loss, then name
- Colorized loss % thresholds
- Route similarity view (last successful trace per target within 2h)
  - Jaccard similarity on hop sets, greedy clustering with threshold 0.6
  - Shows representative route and members
- Live refresh (default 1s)
- Press 'q' to quit gracefully; Ctrl-C also works

Requirements:
  rich
  typer

Usage:
  python netwatch_tui.py --db ./netwatch.db --refresh 1.0
"""
from __future__ import annotations

import math
import os
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import typer
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer(add_completion=False)
console = Console()

WINDOWS = [
    (24 * 3600, "24h"),
    (6 * 3600, "6h"),
    (3600, "1h"),
    (300, "5m"),
]
ROUTE_LOOKBACK = 2 * 3600
SIM_THRESHOLD = 0.6


# --------------------
# SQLite helpers
# --------------------

def db_connect_ro(path: str) -> sqlite3.Connection:
    # Read-only, WAL-friendly
    uri = f"file:{os.path.abspath(path)}?mode=ro&immutable=0"
    conn = sqlite3.connect(uri, timeout=0.3, isolation_level=None, uri=True)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def query_with_retry(conn: sqlite3.Connection, sql: str, params: Tuple=(), retries: int = 5, backoff: float = 0.05):
    for i in range(retries):
        try:
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
            cur.close()
            return rows
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                time.sleep(backoff * (1.5 ** i))
                continue
            raise
    # last attempt
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    return rows


# --------------------
# Data types
# --------------------

@dataclass
class Target:
    id: str
    name: str


@dataclass
class Row:
    target: Target
    totals: Dict[int, Tuple[int, int, Optional[float]]]  # window_secs -> (total, fails, loss_pct)


# --------------------
# Data access
# --------------------

def load_targets(conn: sqlite3.Connection) -> List[Target]:
    rows = query_with_retry(conn, "SELECT id,name FROM targets ORDER BY name")
    return [Target(id=r[0], name=r[1]) for r in rows]


def load_loss_for_window(conn: sqlite3.Connection, window_secs: int) -> Dict[str, Tuple[int, int, Optional[float]]]:
    cutoff = int(time.time()) - window_secs
    rows = query_with_retry(
        conn,
        """
        SELECT target_id,
               COUNT(*) as total,
               SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END) as fails
          FROM ping_samples
         WHERE ts >= ?
         GROUP BY target_id
        """,
        (cutoff,),
    )
    out: Dict[str, Tuple[int, int, Optional[float]]] = {}
    for tid, total, fails in rows:
        loss = (100.0 * fails / total) if total else None
        out[tid] = (total, fails, loss)
    return out


def fmt_loss(loss: Optional[float], total: int) -> Text:
    if total < 2 or loss is None:
        t = Text("--", style="dim")
        return t
    val = max(0.0, min(100.0, loss))
    s = f"{val:.1f}%"
    if val <= 1.0:
        style = "bold green"
    elif val <= 5.0:
        style = "yellow"
    elif val <= 20.0:
        style = "red"
    else:
        style = "bold red"
    return Text(s, style=style)


# --------------------
# Route similarity
# --------------------

def load_latest_routes(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    rows = query_with_retry(
        conn,
        """
        WITH latest AS (
          SELECT target_id, MAX(ts) ts
            FROM trace_samples
           WHERE ts >= strftime('%s','now') - ?
        GROUP BY target_id)
        SELECT h.target_id, h.ts, h.hop_no, h.ip
          FROM latest l
          JOIN trace_hops h ON h.target_id = l.target_id AND h.ts = l.ts
      ORDER BY h.target_id, h.hop_no
        """,
        (ROUTE_LOOKBACK,),
    )
    routes: Dict[str, List[str]] = {}
    for target_id, _ts, hop_no, ip in rows:
        if target_id not in routes:
            routes[target_id] = []
        if ip is None:
            continue
        routes[target_id].append(ip)
    return routes


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cluster_routes(routes: Dict[str, List[str]], threshold: float = SIM_THRESHOLD):
    # Greedy clustering: pick seed with most neighbors >= threshold
    remaining = set(routes.keys())
    groups: List[List[str]] = []
    while remaining:
        # compute neighbor counts
        best_tid = None
        best_neighbors: List[str] = []
        for tid in list(remaining):
            neighbors = [tid]
            for other in remaining:
                if other == tid:
                    continue
                if jaccard(routes[tid], routes[other]) >= threshold:
                    neighbors.append(other)
            if len(neighbors) > len(best_neighbors):
                best_neighbors = neighbors
                best_tid = tid
        groups.append(best_neighbors)
        for x in best_neighbors:
            remaining.discard(x)
    # order groups by size desc
    groups.sort(key=len, reverse=True)
    return groups


# --------------------
# Rendering
# --------------------

def build_table(targets: List[Target], window_maps: Dict[int, Dict[str, Tuple[int, int, Optional[float]]]]) -> Table:
    tbl = Table(box=box.SIMPLE_HEAVY)
    tbl.add_column("Location", width=30)
    for _w, label in WINDOWS:
        tbl.add_column(label, justify="right")

    # Build rows with sorting
    def get_loss(tid: str, w: int) -> Tuple[int, Optional[float]]:
        rec = window_maps[w].get(tid)
        if rec is None:
            return 0, None
        total, _fails, loss = rec
        return total, loss

    sorted_targets = sorted(
        targets,
        key=lambda t: (
            -(get_loss(t.id, 300)[1] if get_loss(t.id, 300)[1] is not None else -1.0),
            -(get_loss(t.id, 3600)[1] if get_loss(t.id, 3600)[1] is not None else -1.0),
            t.name.lower(),
        ),
    )

    for t in sorted_targets:
        row = [Text(f"{t.name} [{t.id}]", style="bold")]
        for w, _label in WINDOWS:
            rec = window_maps[w].get(t.id)
            if rec is None:
                row.append(Text("--", style="dim"))
            else:
                total, _fails, loss = rec
                row.append(fmt_loss(loss, total))
        tbl.add_row(*row)
    return tbl


def format_route_line(hops: List[str], max_hops: int = 30) -> Text:
    if len(hops) > max_hops:
        hops = hops[:max_hops] + ["…"]
    return Text(" \u2192 ".join(hops))


def build_routes_panel(targets_by_id: Dict[str, Target], routes: Dict[str, List[str]]) -> Panel:
    if not routes:
        return Panel(Text("No recent traceroutes (last 2h)."), title="Approximate Routes (2h)")

    groups = cluster_routes(routes, SIM_THRESHOLD)
    sections: List[Text] = []
    for idx, group in enumerate(groups, start=1):
        # representative = first member
        rep_tid = group[0]
        rep_route = routes[rep_tid]
        header = Text(f"Group {idx} ", style="bold")
        header.append(f"({len(group)} targets)")
        sections.append(header)
        sections.append(format_route_line(rep_route))
        # members
        for tid in group:
            t = targets_by_id.get(tid)
            if not t:
                continue
            if tid == rep_tid:
                line = Text(f"  • {t.name} [{tid}]", style="dim")
            else:
                line = Text(f"  • {t.name} [{tid}]")
            sections.append(line)
        sections.append(Text(""))

    body = Group(*sections)
    return Panel(body, title="Approximate Routes (2h)")


def layout_render(conn: sqlite3.Connection) -> Panel:
    # fetch data
    targets = load_targets(conn)
    window_maps: Dict[int, Dict[str, Tuple[int, int, Optional[float]]]] = {}
    for w, _label in WINDOWS:
        window_maps[w] = load_loss_for_window(conn, w)

    routes = load_latest_routes(conn)

    # compose
    table = build_table(targets, window_maps)
    top = Panel(table, title="Netwatch — Loss by Window (sorted by 5m)")

    targets_by_id = {t.id: t for t in targets}
    routes_panel = build_routes_panel(targets_by_id, routes)

    return Panel(Group(top, routes_panel), box=box.SQUARE)


# --------------------
# Keyboard handling (q to quit)
# --------------------

def read_stdin_nonblocking() -> Optional[str]:
    import select
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip() or None
    return None


# --------------------
# Main loop
# --------------------

@app.command()
def tui(db: str = typer.Option(..., help="Path to SQLite database"),
        refresh: float = typer.Option(1.0, help="Refresh interval seconds")):
    conn = db_connect_ro(db)

    stop = False
    def _stop(*_):
        nonlocal stop
        stop = True
    try:
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
    except Exception:
        pass

    with Live(Align.left(layout_render(conn)), console=console, refresh_per_second=(1.0 / refresh if refresh > 0 else 4.0), screen=True) as live:
        while not stop:
            try:
                view = Align.left(layout_render(conn))
            except Exception as e:
                view = Panel(Text(f"Error: {e}", style="red"), title="Netwatch")
            live.update(view, refresh=True)  # in-place screen update
            # simple key handling: 'q' to quit
            key = read_stdin_nonblocking()
            if key and key.lower().startswith('q'):
                break
            time.sleep(refresh)

    try:
        conn.close()
    except Exception:
        pass


if __name__ == "__main__":
    app()
