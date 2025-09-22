# netwatch_agent.py
"""
Netwatch Agent — data-probe side of the two-component (agent + TUI) design.

MVP features:
- Loads YAML config of targets and intervals
- Initializes SQLite (WAL) with the specified schema
- Schedules concurrent ICMP ping and traceroute subprocesses (system binaries)
- Parses outputs and writes samples to SQLite
- Periodic pruning (36h retention)
- Optional dry-run mode (no DB writes)

CLI:
  python netwatch_agent.py agent --db ./netwatch.db --config ./config.yaml
  python netwatch_agent.py init  --db ./netwatch.db --config ./config.yaml
  python netwatch_agent.py check --config ./config.yaml

Requirements (see requirements.txt):
  PyYAML
  typer

Stdlib only otherwise (asyncio, sqlite3, re, time, random, shutil, ipaddress)
"""
from __future__ import annotations

import asyncio
import dataclasses
import enum
import os
import random
import re
import shutil
import signal
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import typer
import yaml

app = typer.Typer(add_completion=False, help="Netwatch data-probe agent (MVP)")

# -------------------------
# Config models (lightweight)
# -------------------------

@dataclass
class Concurrency:
    ping: int = 30
    traceroute: int = 3

@dataclass
class Target:
    id: str
    name: str
    host: str
    family: str = "auto"  # auto|v4|v6

@dataclass
class Config:
    probe_interval_secs: int = 60
    trace_interval_secs: int = 600
    ping_timeout_secs: int = 2
    traceroute_timeout_secs: int = 15
    traceroute_per_hop_timeout: int = 1
    traceroute_max_hops: int = 30
    concurrency: Concurrency = dataclasses.field(default_factory=Concurrency)
    targets: List[Target] = dataclasses.field(default_factory=list)


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # Basic validation
    def _require(k: str, container: dict):
        if k not in container:
            raise ValueError(f"Missing required config key: {k}")
    _require("targets", raw)
    targets = []
    for t in raw["targets"]:
        for k in ("id", "name", "host"):
            if k not in t:
                raise ValueError(f"Target missing key {k}: {t}")
        fam = t.get("family", "auto")
        if fam not in ("auto", "v4", "v6"):
            raise ValueError(f"Invalid family '{fam}' for target {t['id']}")
        targets.append(Target(id=t["id"], name=t["name"], host=t["host"], family=fam))
    conc_raw = raw.get("concurrency", {})
    conc = Concurrency(ping=int(conc_raw.get("ping", 30)), traceroute=int(conc_raw.get("traceroute", 3)))
    cfg = Config(
        probe_interval_secs=int(raw.get("probe_interval_secs", 60)),
        trace_interval_secs=int(raw.get("trace_interval_secs", 600)),
        ping_timeout_secs=int(raw.get("ping_timeout_secs", 2)),
        traceroute_timeout_secs=int(raw.get("traceroute_timeout_secs", 15)),
        traceroute_per_hop_timeout=int(raw.get("traceroute_per_hop_timeout", 1)),
        traceroute_max_hops=int(raw.get("traceroute_max_hops", 30)),
        concurrency=conc,
        targets=targets,
    )
    return cfg


# -------------------------
# SQLite schema & helpers
# -------------------------

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS targets (
  id        TEXT PRIMARY KEY,
  name      TEXT NOT NULL,
  host      TEXT NOT NULL,
  family    TEXT NOT NULL DEFAULT 'auto'
);

CREATE TABLE IF NOT EXISTS ping_samples (
  ts        INTEGER NOT NULL,
  target_id TEXT NOT NULL REFERENCES targets(id) ON DELETE CASCADE,
  ok        INTEGER NOT NULL,
  rtt_ms    REAL,
  PRIMARY KEY (ts, target_id)
);
CREATE INDEX IF NOT EXISTS idx_ping_target_ts ON ping_samples(target_id, ts);

CREATE TABLE IF NOT EXISTS trace_samples (
  ts        INTEGER NOT NULL,
  target_id TEXT NOT NULL REFERENCES targets(id) ON DELETE CASCADE,
  duration_ms INTEGER,
  PRIMARY KEY (ts, target_id)
);
CREATE INDEX IF NOT EXISTS idx_trace_target_ts ON trace_samples(target_id, ts);

CREATE TABLE IF NOT EXISTS trace_hops (
  ts        INTEGER NOT NULL,
  target_id TEXT NOT NULL,
  hop_no    INTEGER NOT NULL,
  ip        TEXT,
  rtt_ms    REAL,
  PRIMARY KEY (ts, target_id, hop_no),
  FOREIGN KEY (ts, target_id) REFERENCES trace_samples(ts, target_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_hops_target_ts ON trace_hops(target_id, ts);
"""

RETENTION_SECONDS = 30 * 24 * 3600  # 30 days


def db_connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=5, isolation_level=None)  # autocommit
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def db_init(conn: sqlite3.Connection, cfg: Config) -> None:
    conn.executescript(SCHEMA_SQL)
    typer.secho("✓ Database schema initialized", fg=typer.colors.GREEN)
    # Upsert targets
    with conn:
        for t in cfg.targets:
            conn.execute(
                "INSERT INTO targets(id,name,host,family) VALUES(?,?,?,?)\n"
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name, host=excluded.host, family=excluded.family",
                (t.id, t.name, t.host, t.family),
            )
        typer.secho(f"✓ Upserted {len(cfg.targets)} targets", fg=typer.colors.GREEN)


def db_insert_ping(conn: sqlite3.Connection, ts: int, target_id: str, ok: bool, rtt_ms: Optional[float], dry_run: bool=False) -> None:
    if dry_run:
        return
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO ping_samples(ts,target_id,ok,rtt_ms) VALUES (?,?,?,?)",
            (ts, target_id, 1 if ok else 0, rtt_ms),
        )


def db_insert_trace(conn: sqlite3.Connection, ts: int, target_id: str, duration_ms: Optional[int], hops: List[Tuple[int, Optional[str], Optional[float]]], dry_run: bool=False) -> None:
    if dry_run:
        return
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO trace_samples(ts,target_id,duration_ms) VALUES (?,?,?)",
            (ts, target_id, duration_ms),
        )
        for hop_no, ip, rtt in hops:
            conn.execute(
                "INSERT OR REPLACE INTO trace_hops(ts,target_id,hop_no,ip,rtt_ms) VALUES (?,?,?,?,?)",
                (ts, target_id, hop_no, ip, rtt),
            )


def db_prune(conn: sqlite3.Connection) -> None:
    cutoff = int(time.time()) - RETENTION_SECONDS
    with conn:
        cursor = conn.execute("DELETE FROM ping_samples WHERE ts < ?", (cutoff,))
        ping_deleted = cursor.rowcount
        cursor = conn.execute("DELETE FROM trace_hops WHERE ts < ?", (cutoff,))
        cursor = conn.execute("DELETE FROM trace_samples WHERE ts < ?", (cutoff,))
        trace_deleted = cursor.rowcount
    if ping_deleted > 0 or trace_deleted > 0:
        typer.secho(f"🧹 Pruned {ping_deleted} ping samples, {trace_deleted} trace samples", fg=typer.colors.BLUE)


# -------------------------
# Probing subprocess wrappers
# -------------------------

PING_RTT_RE = re.compile(r"time=(?P<ms>[0-9]+\.?[0-9]*) ms")
TRACE_LINE_RE = re.compile(r"^\s*(?P<hop>\d+)\s+(?P<ip>(?:\d+\.){3}\d+|[0-9a-fA-F:]+|\*)\s+(?P<rtt>[0-9]+\.?[0-9]*)\s+ms")


class IPFamily(enum.Enum):
    AUTO = "auto"
    V4 = "v4"
    V6 = "v6"


async def run_ping(host: str, timeout: int, family: IPFamily) -> Tuple[bool, Optional[float]]:
    """Return (ok, rtt_ms)."""
    cmds = []
    if family == IPFamily.V4:
        cmds.append(["ping", "-n", "-c", "1", "-w", str(timeout), host])
    elif family == IPFamily.V6:
        cmds.append(["ping6", "-n", "-c", "1", "-w", str(timeout), host])
    else:  # AUTO: try ping (v4) then ping6
        cmds.append(["ping", "-n", "-c", "1", "-w", str(timeout), host])
        cmds.append(["ping6", "-n", "-c", "1", "-w", str(timeout), host])

    for cmd in cmds:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout + 1)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                continue
            ok = proc.returncode == 0
            if not ok:
                continue
            text = stdout.decode(errors="ignore")
            m = PING_RTT_RE.search(text)
            rtt_ms = float(m.group("ms")) if m else None
            return True, rtt_ms
        except FileNotFoundError:
            return False, None
    return False, None


async def run_traceroute(host: str, per_hop_w: int, max_hops: int, family: IPFamily, overall_timeout: int) -> Tuple[List[Tuple[int, Optional[str], Optional[float]]], int]:
    """Return (hops, duration_ms). hops: list of (hop_no, ip|None, rtt_ms|None)."""
    base = ["traceroute", "-n", "-q", "1", "-w", str(per_hop_w), "-m", str(max_hops)]
    if family == IPFamily.V6:
        base.insert(1, "-6")
    cmd = [*base, host]
    start = time.time()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError:
        return [], int((time.time() - start) * 1000)
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=overall_timeout)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        stdout = b""
    text = stdout.decode(errors="ignore")
    hops: List[Tuple[int, Optional[str], Optional[float]]] = []
    for line in text.splitlines():
        m = TRACE_LINE_RE.match(line)
        if not m:
            continue
        hop_no = int(m.group("hop"))
        ip_raw = m.group("ip")
        ip = None if ip_raw == "*" else ip_raw
        rtt_ms = float(m.group("rtt")) if m.group("rtt") else None
        hops.append((hop_no, ip, rtt_ms))
    duration_ms = int((time.time() - start) * 1000)
    return hops, duration_ms


# -------------------------
# Agent scheduler
# -------------------------

@dataclass
class SchedState:
    next_ping: Dict[str, float]
    next_trace: Dict[str, float]


def jitter(base: int, pct: float = 0.1) -> float:
    d = base * pct
    return base + random.uniform(-d, d)


async def agent_loop(db_path: str, cfg: Config, dry_run: bool=False) -> None:
    conn = db_connect(db_path)
    db_init(conn, cfg)
    typer.secho(f"🚀 Agent starting with {len(cfg.targets)} targets", fg=typer.colors.CYAN, bold=True)

    # Verify binaries
    missing = []
    if not shutil.which("ping") and not shutil.which("ping6"):
        missing.append("ping/ping6")
    if not shutil.which("traceroute"):
        missing.append("traceroute")
    if missing:
        typer.secho(f"Warning: missing binaries: {', '.join(missing)}", fg=typer.colors.YELLOW)

    sem_ping = asyncio.Semaphore(cfg.concurrency.ping)
    sem_trace = asyncio.Semaphore(cfg.concurrency.traceroute)

    state = SchedState(next_ping={}, next_trace={})
    now = time.time()
    for t in cfg.targets:
        state.next_ping[t.id] = now + random.uniform(0, cfg.probe_interval_secs)
        state.next_trace[t.id] = now + random.uniform(0, cfg.trace_interval_secs)

    stop_event = asyncio.Event()

    def _handle_sig():
        stop_event.set()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_sig)
        except NotImplementedError:
            pass

    async def do_ping(t: Target):
        fam = IPFamily(t.family)
        async with sem_ping:
            ts = int(time.time())
            ok, rtt = await run_ping(t.host, cfg.ping_timeout_secs, fam)
            if ok:
                msg = f"📡 ping {t.id} ({t.host}): {rtt:.1f}ms" if rtt else f"📡 ping {t.id} ({t.host}): ok"
                typer.secho(msg, fg=typer.colors.GREEN)
            else:
                typer.secho(f"❌ ping {t.id} ({t.host}): failed", fg=typer.colors.RED)
            db_insert_ping(conn, ts, t.id, ok, rtt, dry_run=dry_run)

    async def do_trace(t: Target):
        fam = IPFamily(t.family)
        async with sem_trace:
            ts = int(time.time())
            hops, dur = await run_traceroute(
                t.host,
                cfg.traceroute_per_hop_timeout,
                cfg.traceroute_max_hops,
                fam,
                cfg.traceroute_timeout_secs,
            )
            if hops:
                typer.secho(f"🛤️  trace {t.id} ({t.host}): {len(hops)} hops in {dur}ms", fg=typer.colors.MAGENTA)
            else:
                typer.secho(f"❌ trace {t.id} ({t.host}): failed (timeout or no response)", fg=typer.colors.RED)
            db_insert_trace(conn, ts, t.id, dur, hops, dry_run=dry_run)

    last_prune = 0

    try:
        while not stop_event.is_set():
            now = time.time()
            # Dispatch due tasks
            tasks: List[asyncio.Task] = []
            for t in cfg.targets:
                if now >= state.next_ping[t.id]:
                    tasks.append(asyncio.create_task(do_ping(t)))
                    state.next_ping[t.id] = now + jitter(cfg.probe_interval_secs)
                if now >= state.next_trace[t.id]:
                    tasks.append(asyncio.create_task(do_trace(t)))
                    state.next_trace[t.id] = now + jitter(cfg.trace_interval_secs)
            if tasks:
                # Let tasks progress but don't block the loop entirely
                await asyncio.sleep(0)  # yield

            # Periodic prune
            if now - last_prune > 3600:  # every hour
                db_prune(conn)
                last_prune = now

            await asyncio.wait([asyncio.create_task(asyncio.sleep(0.25))], return_when=asyncio.ALL_COMPLETED)
    finally:
        conn.close()


# -------------------------
# CLI commands
# -------------------------

@app.command()
def init(db: str = typer.Option(..., help="Path to SQLite database"),
         config: str = typer.Option(..., help="Path to config.yaml")):
    """Initialize database schema and upsert targets from config."""
    cfg = load_config(config)
    conn = db_connect(db)
    db_init(conn, cfg)
    conn.close()
    typer.echo("Initialized DB and targets.")


@app.command()
def agent(db: str = typer.Option(..., help="Path to SQLite database"),
          config: str = typer.Option(..., help="Path to config.yaml"),
          dry_run: bool = typer.Option(False, help="Run probes but skip DB writes")):
    """Run the headless probing agent."""
    cfg = load_config(config)
    typer.echo("Starting netwatch agent (MVP)…")
    asyncio.run(agent_loop(db, cfg, dry_run=dry_run))


@app.command()
def check(config: str = typer.Option(..., help="Path to config.yaml")):
    """Check presence of required binaries and print config summary."""
    cfg = load_config(config)
    ping_ok = shutil.which("ping") or shutil.which("ping6")
    tr_ok = shutil.which("traceroute")
    typer.echo(f"ping present: {'yes' if ping_ok else 'NO'}")
    typer.echo(f"traceroute present: {'yes' if tr_ok else 'NO'}")
    typer.echo(f"Targets: {len(cfg.targets)} | ping interval: {cfg.probe_interval_secs}s | trace interval: {cfg.trace_interval_secs}s")


if __name__ == "__main__":
    # Lazy import for contextlib used only in run_ping/kill path
    import contextlib  # noqa: F401
    app()
