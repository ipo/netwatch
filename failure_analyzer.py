#!/usr/bin/env python3
"""
Failure Analyzer - Identifies 5-minute sections with high failure rates
and shows consecutive success/failure patterns for each section.

Usage:
    python failure_analyzer.py --db ./netwatch.db --hours 36 --top 5
"""

import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import argparse
from datetime import datetime, timedelta


@dataclass
class PingSample:
    ts: int
    target_id: str
    target_name: str
    ok: bool
    rtt_ms: Optional[float]


@dataclass
class SegmentInfo:
    start_ts: int
    end_ts: int
    total_samples: int
    failed_samples: int
    failure_percentage: float

    @property
    def start_time(self) -> str:
        return datetime.fromtimestamp(self.start_ts).strftime('%Y-%m-%d %H:%M:%S')

    @property
    def end_time(self) -> str:
        return datetime.fromtimestamp(self.end_ts).strftime('%Y-%m-%d %H:%M:%S')


def db_connect_ro(path: str) -> sqlite3.Connection:
    """Create read-only connection to SQLite database"""
    import os
    uri = f"file:{os.path.abspath(path)}?mode=ro&immutable=0"
    conn = sqlite3.connect(uri, timeout=0.3, isolation_level=None, uri=True)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def get_live_nodes(conn: sqlite3.Connection) -> set:
    """Get target IDs that have responded successfully at least once in the last hour"""
    cutoff = int(time.time()) - 3600
    cursor = conn.execute("""
        SELECT DISTINCT target_id
        FROM ping_samples
        WHERE ts >= ? AND ok = 1
    """, (cutoff,))
    return {row[0] for row in cursor.fetchall()}


def find_high_failure_segments(conn: sqlite3.Connection, hours_back: int, top_n: int) -> List[SegmentInfo]:
    """Find 5-minute segments with highest failure rates"""
    live_nodes = get_live_nodes(conn)
    if not live_nodes:
        return []

    cutoff = int(time.time()) - (hours_back * 3600)
    placeholders = ','.join('?' * len(live_nodes))

    cursor = conn.execute(f"""
        SELECT
            (ts / 300) * 300 as segment_start,
            COUNT(*) as total_samples,
            SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) as failed_samples,
            100.0 * SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) / COUNT(*) as failure_percentage
        FROM ping_samples
        WHERE ts >= ? AND target_id IN ({placeholders})
        GROUP BY segment_start
        HAVING total_samples >= 5
        ORDER BY failure_percentage DESC, total_samples DESC
        LIMIT ?
    """, (cutoff,) + tuple(live_nodes) + (top_n,))

    segments = []
    for row in cursor.fetchall():
        segment_start, total, failed, failure_pct = row
        segments.append(SegmentInfo(
            start_ts=segment_start,
            end_ts=segment_start + 300,
            total_samples=total,
            failed_samples=failed,
            failure_percentage=failure_pct
        ))

    return segments


def get_segment_samples(conn: sqlite3.Connection, segment: SegmentInfo) -> List[PingSample]:
    """Get all ping samples for a specific 5-minute segment, ordered by timestamp"""
    live_nodes = get_live_nodes(conn)
    if not live_nodes:
        return []

    placeholders = ','.join('?' * len(live_nodes))

    cursor = conn.execute(f"""
        SELECT p.ts, p.target_id, t.name, p.ok, p.rtt_ms
        FROM ping_samples p
        JOIN targets t ON p.target_id = t.id
        WHERE p.ts >= ? AND p.ts < ? AND p.target_id IN ({placeholders})
        ORDER BY p.ts, p.target_id
    """, (segment.start_ts, segment.end_ts) + tuple(live_nodes))

    samples = []
    for row in cursor.fetchall():
        ts, target_id, target_name, ok, rtt_ms = row
        samples.append(PingSample(
            ts=ts,
            target_id=target_id,
            target_name=target_name,
            ok=bool(ok),
            rtt_ms=rtt_ms
        ))

    return samples


def format_timestamp(ts: int) -> str:
    """Format timestamp for display"""
    return datetime.fromtimestamp(ts).strftime('%H:%M:%S')


def print_segment_analysis(segment: SegmentInfo, samples: List[PingSample]):
    """Print detailed analysis of a 5-minute segment"""
    print(f"\n{'='*80}")
    print(f"SEGMENT: {segment.start_time} to {segment.end_time}")
    print(f"Overall: {segment.failure_percentage:.2f}% failure rate ({segment.failed_samples}/{segment.total_samples} samples)")
    print(f"{'='*80}")

    if not samples:
        print("No samples found in this segment.")
        print(f"{'='*80}")
        return

    # Print each ping result line by line in chronological order
    for sample in samples:
        ts_str = format_timestamp(sample.ts)
        rtt_str = f"{sample.rtt_ms:.1f}ms" if sample.rtt_ms is not None else "timeout"

        if sample.ok:
            status_text = "ok  "
            status_color = "\033[92m"  # Green
            line = f"{ts_str} {status_color}{status_text}\033[0m {sample.target_name} ({sample.target_id}) - {rtt_str}"
        else:
            status_text = "FAIL"
            status_color = "\033[91m"  # Red
            line = f"{ts_str} {status_color}{status_text}\033[0m {sample.target_name} ({sample.target_id}) - {rtt_str}"

        print(line)

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Analyze network failure patterns in 5-minute segments")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--hours", type=int, default=36, help="Hours to analyze (default: 36)")
    parser.add_argument("--top", type=int, default=5, help="Number of top failure segments to analyze (default: 5)")

    args = parser.parse_args()

    try:
        conn = db_connect_ro(args.db)

        print(f"Analyzing top {args.top} failure segments from last {args.hours} hours...")

        # Find segments with highest failure rates
        segments = find_high_failure_segments(conn, args.hours, args.top)

        if not segments:
            print("No segments with failures found.")
            return

        print(f"\nFound {len(segments)} segments with high failure rates:")
        for i, segment in enumerate(segments, 1):
            print(f"{i}. {segment.start_time}: {segment.failure_percentage:.2f}% failure rate")

        # Analyze each segment in detail
        for i, segment in enumerate(segments, 1):
            samples = get_segment_samples(conn, segment)
            print_segment_analysis(segment, samples)

        conn.close()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())