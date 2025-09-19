# Netwatch

Netwatch is a lightweight network monitoring tool that continuously monitors network connectivity using ping and traceroute. It consists of two components: a data collection agent that probes configured targets and stores results in SQLite, and a rich terminal UI that displays real-time loss statistics and route analysis.

## Features
- Real-time ping and traceroute monitoring with configurable intervals
- SQLite database with WAL mode for concurrent read/write access
- Color-coded loss percentages across multiple time windows (5m, 1h, 6h, 24h)
- Route similarity analysis with automatic clustering of similar paths
- Concurrent probing with configurable limits

## Requirements
Python 3.6+ and system binaries: `ping`, `ping6`, `traceroute`

## Installation
```bash
pip install -r requirements.txt
```

## Configuration
Edit `config.yaml` to configure monitoring targets, intervals, and concurrency settings.

## Usage

To run prober:
```pip install -r requirements.txt
python3 probe.py init --db ./netwatch.db --config ./config.yaml
python3 probe.py agent --db ./netwatch.db --config ./config.yaml```

To run reporter:
`python3 tui.py --db ./netwatch.db --refresh 1.0`
