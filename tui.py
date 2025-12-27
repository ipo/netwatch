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
  PyYAML (via probe.py import)

Usage:
  python netwatch_tui.py --db ./netwatch.db --refresh 1.0
  python netwatch_tui.py --db ./netwatch.db --config ./config.yaml --refresh 1.0
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
# Config loading
# --------------------

def load_config_targets(config_path: Optional[str]) -> Optional[List[str]]:
    """Load target IDs from config file using probe.py's config loader."""
    # None means "no config supplied" so caller can fall back to all targets.
    if not config_path:
        return None

    try:
        # Import probe.py's config loader to avoid duplication
        from probe import load_config
        cfg = load_config(config_path)
        return [t.id for t in cfg.targets]
    except (FileNotFoundError, ImportError, ValueError, Exception):
        return []


# --------------------
# Data access
# --------------------

def load_targets(conn: sqlite3.Connection, config_targets: List[str] = None) -> List[Target]:
    if config_targets is None:
        # If no config targets provided, return all targets (backward compatibility)
        rows = query_with_retry(conn, "SELECT id,name FROM targets ORDER BY name")
        return [Target(id=r[0], name=r[1]) for r in rows]

    # Filter to only include targets that are in the config
    if not config_targets:
        return []

    placeholders = ','.join('?' * len(config_targets))
    rows = query_with_retry(
        conn,
        f"SELECT id,name FROM targets WHERE id IN ({placeholders}) ORDER BY name",
        tuple(config_targets)
    )
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
# Offline detection
# --------------------

def get_live_nodes(conn: sqlite3.Connection) -> set:
    """
    Get the set of target IDs that have responded successfully at least once in the last hour.
    These are considered "live" nodes for offline determination purposes.
    """
    cutoff = int(time.time()) - 3600  # 1 hour ago
    rows = query_with_retry(
        conn,
        """
        SELECT DISTINCT target_id
          FROM ping_samples
         WHERE ts >= ? AND ok = 1
        """,
        (cutoff,),
    )
    return {row[0] for row in rows}

def check_offline_status(conn: sqlite3.Connection, offline_threshold: int, live_nodes: set) -> Optional[float]:
    # Get the most recent pings across all targets, filtered to live nodes only
    if not live_nodes:
        return None  # No live nodes to check

    placeholders = ','.join('?' * len(live_nodes))
    rows = query_with_retry(
        conn,
        f"""
        SELECT ts, ok
          FROM ping_samples
         WHERE target_id IN ({placeholders})
      ORDER BY ts DESC
         LIMIT ?
        """,
        tuple(live_nodes) + (offline_threshold,),
    )

    if len(rows) < offline_threshold:
        return None

    # Check if all recent pings failed
    all_failed = all(row[1] == 0 for row in rows)
    if not all_failed:
        return None

    # Calculate offline duration from the oldest failed ping
    oldest_failed_ts = rows[-1][0]
    current_time = time.time()
    offline_minutes = (current_time - oldest_failed_ts) / 60.0

    return offline_minutes


def check_hourly_offline_status(conn: sqlite3.Connection, offline_threshold: int, live_nodes: set, hours_back: int = 168) -> List[Tuple[int, bool]]:
    """
    Check offline status for each hour going back from current hour (clock-aligned).
    Returns list of tuples where (consecutive_failures, has_data).
    consecutive_failures: 0 = no downtime, >0 = max consecutive failures in that hour.
    has_data: True if there were any ping samples in that hour.
    Only returns non-zero consecutive_failures if >= offline_threshold.
    Index 0 is most recent hour. Only considers pings to live nodes.
    """
    if not live_nodes:
        return [(0, False)] * hours_back  # No live nodes to check

    import datetime
    now = datetime.datetime.now()
    # Round down to the nearest hour (clock-aligned)
    current_hour_start = now.replace(minute=0, second=0, microsecond=0)

    hourly_status = []
    placeholders = ','.join('?' * len(live_nodes))

    for hour_offset in range(hours_back):
        # Define clock-aligned hour boundaries
        hour_start_dt = current_hour_start - datetime.timedelta(hours=hour_offset)
        hour_end_dt = hour_start_dt + datetime.timedelta(hours=1)

        hour_start = int(hour_start_dt.timestamp())
        hour_end = int(hour_end_dt.timestamp())

        # Get ALL pings for this hour in chronological order (oldest first), filtered to live nodes
        rows = query_with_retry(
            conn,
            f"""
            SELECT ts, ok
              FROM ping_samples
             WHERE ts >= ? AND ts < ? AND target_id IN ({placeholders})
          ORDER BY ts ASC
            """,
            (hour_start, hour_end) + tuple(live_nodes),
        )

        # Find the maximum consecutive failures in this hour
        max_consecutive_failures = 0
        current_consecutive = 0

        if len(rows) >= offline_threshold:
            for _, ok in rows:
                if ok == 0:  # Failed ping
                    current_consecutive += 1
                    max_consecutive_failures = max(max_consecutive_failures, current_consecutive)
                else:  # Successful ping
                    current_consecutive = 0

        # Check for cross-boundary downtime if we have failures at the edges
        enhanced_max_consecutive = max_consecutive_failures

        if len(rows) > 0:
            # Check if downtime starts at the very beginning of the hour
            if rows[0][1] == 0:  # First ping is a failure
                # Count consecutive failures at the start of this hour
                start_failures = 0
                for _, ok in rows:
                    if ok == 0:
                        start_failures += 1
                    else:
                        break

                # Look back to previous hour to see if there are more failures
                prev_hour_start = hour_start - 3600
                prev_hour_end = hour_start
                prev_rows = query_with_retry(
                    conn,
                    f"""
                    SELECT ts, ok
                      FROM ping_samples
                     WHERE ts >= ? AND ts < ? AND target_id IN ({placeholders})
                  ORDER BY ts DESC
                    """,
                    (prev_hour_start, prev_hour_end) + tuple(live_nodes),
                )

                # Count consecutive failures at the end of previous hour
                prev_end_failures = 0
                for _, ok in prev_rows:
                    if ok == 0:
                        prev_end_failures += 1
                    else:
                        break

                cross_boundary_failures = prev_end_failures + start_failures
                enhanced_max_consecutive = max(enhanced_max_consecutive, cross_boundary_failures)

            # Check if downtime ends at the very end of the hour
            if rows[-1][1] == 0:  # Last ping is a failure
                # Count consecutive failures at the end of this hour
                end_failures = 0
                for _, ok in reversed(rows):
                    if ok == 0:
                        end_failures += 1
                    else:
                        break

                # Look forward to next hour to see if there are more failures
                next_hour_start = hour_end
                next_hour_end = hour_end + 3600
                next_rows = query_with_retry(
                    conn,
                    f"""
                    SELECT ts, ok
                      FROM ping_samples
                     WHERE ts >= ? AND ts < ? AND target_id IN ({placeholders})
                  ORDER BY ts ASC
                    """,
                    (next_hour_start, next_hour_end) + tuple(live_nodes),
                )

                # Count consecutive failures at the start of next hour
                next_start_failures = 0
                for _, ok in next_rows:
                    if ok == 0:
                        next_start_failures += 1
                    else:
                        break

                cross_boundary_failures = end_failures + next_start_failures
                enhanced_max_consecutive = max(enhanced_max_consecutive, cross_boundary_failures)

        # Only report if it meets the threshold
        has_data = len(rows) > 0
        if enhanced_max_consecutive >= offline_threshold:
            hourly_status.append((enhanced_max_consecutive, has_data))
        else:
            hourly_status.append((0, has_data))

    return hourly_status


# --------------------
# Rendering
# --------------------

def build_offline_hours_line(hourly_status: List[Tuple[int, bool]], max_width: int = 200) -> Text:
    """
    Build the offline hours visualization line.
    Shows red characters representing consecutive failure count for offline hours,
    dark grey '-' for hours with data but no offline criteria fulfilled,
    spaces for hours with no data, with '|' at midnight boundaries.
    Fits as many hours as possible within the available width.
    """
    if not hourly_status:
        return Text(" Offline hours: (no data)")

    prefix = " Offline hours: "
    result = Text(prefix)

    # Calculate available width for hour indicators
    available_width = max_width - len(prefix)

    # Calculate current time info for midnight boundary detection
    import datetime
    now = datetime.datetime.now()
    current_hour_start = now.replace(minute=0, second=0, microsecond=0)

    # Each hour takes 1 character, plus boundary markers at midnight
    hours_displayed = 0
    chars_used = 0

    for i, (consecutive_failures, has_data) in enumerate(hourly_status):
        # Calculate the hour this represents (going back from current hour start)
        hour_dt = current_hour_start - datetime.timedelta(hours=i)
        hour_of_day = hour_dt.hour

        # Check if we need a midnight boundary marker
        # This happens when transitioning from hour 0 to hour 23 (crossing midnight going backwards)
        needs_boundary = False
        if i > 0:
            prev_hour_dt = current_hour_start - datetime.timedelta(hours=i-1)
            prev_hour_of_day = prev_hour_dt.hour
            # Midnight boundary when going from 0 to 23
            needs_boundary = (prev_hour_of_day == 0 and hour_of_day == 23)

        chars_needed = 1 + (1 if needs_boundary else 0)

        # Stop if we'd exceed available width
        if chars_used + chars_needed > available_width:
            break

        # Add midnight boundary marker
        if needs_boundary:
            result.append("|", style="dim")
            chars_used += 1

        # Add hour status
        if consecutive_failures > 0:
            # Show count up to 9, then '+' for 10+
            if consecutive_failures <= 9:
                result.append(str(consecutive_failures), style="bold red")
            else:
                result.append("+", style="bold red")
        elif has_data:
            # Has data but no offline criteria fulfilled - show dark grey dash
            result.append("-", style="dim")
        else:
            # No data - show space
            result.append(" ")

        chars_used += 1
        hours_displayed += 1

    return result

def _measure_window_cell(rec: Optional[Tuple[int, int, Optional[float], Optional[float]]],
                        thresh_low: float,
                        thresh_high: float) -> int:
    """
    Return the printable length of a loss/rtt cell (plain characters, no ANSI),
    e.g. '0.1% / 34ms'. Keeps measurement logic in one place.
    """
    if rec is None:
        return len("--")

    total, _fails, loss, avg_rtt = rec
    loss_text = fmt_loss(loss, total)
    rtt_text = fmt_avg_rtt(avg_rtt, total, thresh_low, thresh_high)
    combined = f"{loss_text.plain} / {rtt_text.plain}"
    return len(combined)


def _truncate_label(label: str, max_width: int) -> str:
    """Truncate label to fit within max_width, appending '..' when needed."""
    if len(label) <= max_width:
        return label
    if max_width <= 2:
        return label[:max_width]
    return label[: max_width - 2] + ".."


def build_table(
    targets: List[Target],
    window_maps: Dict[int, Dict[str, Tuple[int, int, Optional[float], Optional[float]]]],
    console_width: int = 120,
) -> Table:
    # Calculate dynamic RTT thresholds first
    thresh_low, thresh_high = calculate_rtt_thresholds(window_maps)

    # Determine required width for each metric column (content only, no padding)
    window_col_widths: Dict[int, int] = {}
    for w, label in WINDOWS:
        max_len = len(label)
        for t in targets:
            rec = window_maps[w].get(t.id)
            max_len = max(max_len, _measure_window_cell(rec, thresh_low, thresh_high))
        # Add buffer so loss/rtt text isn't clipped; user asked for extra room
        window_col_widths[w] = max_len + 4

    # Accurate width budget: Rich table width ~= sum(col_widths)
    # + padding(2 per col) + vertical borders (num_cols + 1).
    num_cols = 1 + len(WINDOWS)
    padding_lr = 1  # from padding=(0,1) set below
    fixed_overhead = (padding_lr * 2 * num_cols) + (num_cols + 1)

    metrics_total = sum(window_col_widths.values())
    remaining = console_width - fixed_overhead - metrics_total
    # Keep the label reasonably wide; fall back to 10 chars minimum.
    label_width = max(10, remaining)

    # If still over budget (console very narrow), cap at remaining space.
    total_estimated_width = metrics_total + label_width + fixed_overhead
    if total_estimated_width > console_width:
        label_width = max(10, console_width - fixed_overhead - metrics_total)

    tbl = Table(box=box.SIMPLE_HEAVY, padding=(0, 1))
    tbl.add_column("Location", width=label_width, no_wrap=True)
    for w, label in WINDOWS:
        tbl.add_column(label, justify="right", width=window_col_widths[w], no_wrap=True)

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
        # Show tag/ID first for quicker scanning, format: "tag: name"
        raw_label = f"{t.id}: {t.name}"
        label = _truncate_label(raw_label, label_width)
        row = [Text(label, style="bold")]
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


def build_offline_banner(offline_minutes: float) -> Text:
    message = f" === OFFLINE FOR {offline_minutes:.1f} MINUTES ==="
    return Text(message, style="bold white on red")


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


def layout_render(conn: sqlite3.Connection, offline_threshold: int, config_targets: List[str] = None) -> Panel:
    # fetch data for normal display
    targets = load_targets(conn, config_targets)
    window_maps: Dict[int, Dict[str, Tuple[int, int, Optional[float], Optional[float]]]] = {}
    for w, _label in WINDOWS:
        window_maps[w] = load_loss_for_window(conn, w)

    routes = load_latest_routes(conn)

    # get live nodes for offline determination (call once per update)
    live_nodes = get_live_nodes(conn)

    # check offline status
    offline_minutes = check_offline_status(conn, offline_threshold, live_nodes)
    hourly_status = check_hourly_offline_status(conn, offline_threshold, live_nodes)

    # compose layout elements
    console_width = console.size.width if console.size else 120
    table = build_table(targets, window_maps, console_width)

    # Create offline hours line as separate element
    # Account for panel borders (typically 2-4 chars) to prevent overflow
    panel_border_width = 8  # Conservative estimate for panel borders + extra margin
    available_width_for_offline_line = max(50, console_width - panel_border_width)
    offline_hours_line = build_offline_hours_line(hourly_status, available_width_for_offline_line)

    # Always show the standard header; offline banner is disabled
    top = Panel(Group(offline_hours_line, table), title="Netwatch — Loss% / Avg RTT by Window (sorted by 5m loss, then ping)")
    content = Group(top)

    targets_by_id = {t.id: t for t in targets}
    routes_panel = build_routes_panel(targets_by_id, routes)

    return Panel(Group(content, routes_panel), box=box.SQUARE)


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
        refresh: float = typer.Option(1.0, help="Refresh interval seconds"),
        offline_threshold: int = typer.Option(3, help="Number of consecutive failed pings to trigger offline mode"),
        config: Optional[str] = typer.Option(None, help="Path to config.yaml (filters displayed targets)")):
    conn = db_connect_ro(db)
    config_targets = load_config_targets(config)

    stop = False
    def _stop(*_):
        nonlocal stop
        stop = True
    try:
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
    except Exception:
        pass

    with Live(Align.left(layout_render(conn, offline_threshold, config_targets)), console=console, refresh_per_second=(1.0 / refresh if refresh > 0 else 4.0), screen=True) as live:
        while not stop:
            try:
                view = Align.left(layout_render(conn, offline_threshold, config_targets))
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
