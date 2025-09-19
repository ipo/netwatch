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
    totals: Dict[int, Tuple[int, int, Optional[float], Optional[float]]]  # window_secs -> (total, fails, loss_pct, avg_rtt)


# --------------------
# Data access
# --------------------

def load_targets(conn: sqlite3.Connection) -> List[Target]:
    rows = query_with_retry(conn, "SELECT id,name FROM targets ORDER BY name")
    return [Target(id=r[0], name=r[1]) for r in rows]


def load_loss_for_window(conn: sqlite3.Connection, window_secs: int) -> Dict[str, Tuple[int, int, Optional[float], Optional[float]]]:
    cutoff = int(time.time()) - window_secs
    rows = query_with_retry(
        conn,
        """
        SELECT target_id,
               COUNT(*) as total,
               SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END) as fails,
               AVG(CASE WHEN ok=1 AND rtt_ms IS NOT NULL THEN rtt_ms ELSE NULL END) as avg_rtt
          FROM ping_samples
         WHERE ts >= ?
         GROUP BY target_id
        """,
        (cutoff,),
    )
    out: Dict[str, Tuple[int, int, Optional[float], Optional[float]]] = {}
    for tid, total, fails, avg_rtt in rows:
        loss = (100.0 * fails / total) if total else None
        out[tid] = (total, fails, loss, avg_rtt)
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


def calculate_rtt_thresholds(window_maps: Dict[int, Dict[str, Tuple[int, int, Optional[float], Optional[float]]]]) -> Tuple[float, float]:
    # Collect all valid RTT values from all windows
    all_rtts = []
    for window_data in window_maps.values():
        for target_id, (total, fails, loss, avg_rtt) in window_data.items():
            if avg_rtt is not None and total > 0:
                all_rtts.append(avg_rtt)

    if len(all_rtts) < 3:
        # Fallback to reasonable defaults if not enough data
        return 50.0, 100.0

    # Sort and calculate median-based thresholds
    all_rtts.sort()
    n = len(all_rtts)

    # Use 33rd and 67th percentiles as thresholds
    thresh_low = all_rtts[n // 3]
    thresh_high = all_rtts[(2 * n) // 3]

    return thresh_low, thresh_high


def fmt_avg_rtt(avg_rtt: Optional[float], total: int, thresh_low: float, thresh_high: float) -> Text:
    if total < 1 or avg_rtt is None:
        return Text("--", style="dim")
    val = avg_rtt

    # Format precision based on value
    if val < 10.0:
        s = f"{val:.1f}ms"
    else:
        s = f"{val:.0f}ms"

    # Dynamic color thresholds
    if val <= thresh_low:
        style = "bold green"
    elif val <= thresh_high:
        style = "yellow"
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

def build_table(targets: List[Target], window_maps: Dict[int, Dict[str, Tuple[int, int, Optional[float], Optional[float]]]]) -> Table:
    tbl = Table(box=box.SIMPLE_HEAVY)
    tbl.add_column("Location", width=30)
    for _w, label in WINDOWS:
        tbl.add_column(label, justify="right")

    # Calculate dynamic RTT thresholds
    thresh_low, thresh_high = calculate_rtt_thresholds(window_maps)

    # Build rows with sorting
    def get_sort_key(target: Target) -> Tuple[float, float, float, str]:
        # Get 5m stats
        rec_5m = window_maps[300].get(target.id)
        loss_5m = rec_5m[2] if rec_5m and rec_5m[2] is not None else -1.0
        rtt_5m = rec_5m[3] if rec_5m and rec_5m[3] is not None else float('inf')

        # Get 1h stats
        rec_1h = window_maps[3600].get(target.id)
        loss_1h = rec_1h[2] if rec_1h and rec_1h[2] is not None else -1.0

        return (
            -loss_5m,      # 5m loss descending (higher loss first)
            rtt_5m,        # 5m ping ascending (lower ping first)
            -loss_1h,      # 1h loss descending (tie breaker)
            target.name.lower()  # name ascending
        )

    sorted_targets = sorted(targets, key=get_sort_key)

    for t in sorted_targets:
        row = [Text(f"{t.name} [{t.id}]", style="bold")]
        for w, _label in WINDOWS:
            rec = window_maps[w].get(t.id)
            if rec is None:
                row.append(Text("--", style="dim"))
            else:
                total, _fails, loss, avg_rtt = rec
                loss_text = fmt_loss(loss, total)
                rtt_text = fmt_avg_rtt(avg_rtt, total, thresh_low, thresh_high)
                combined = Text()
                combined.append_text(loss_text)
                combined.append(" / ")
                combined.append_text(rtt_text)
                row.append(combined)
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
    window_maps: Dict[int, Dict[str, Tuple[int, int, Optional[float], Optional[float]]]] = {}
    for w, _label in WINDOWS:
        window_maps[w] = load_loss_for_window(conn, w)

    routes = load_latest_routes(conn)

    # compose
    table = build_table(targets, window_maps)
    top = Panel(table, title="Netwatch — Loss% / Avg RTT by Window (sorted by 5m loss, then ping)")

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
