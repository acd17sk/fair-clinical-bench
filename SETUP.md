# Setup

## One-time

```bash
cd ~/Documents/epic-project-test/fair-clinical-bench
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                       # smoke test should pass
```

## Running the swarm to build the project

1. Open opencode in this directory:
   ```bash
   cd ~/Documents/epic-project-test/fair-clinical-bench
   opencode
   ```
2. Enable Epic + Lean Turbo for the session:
   ```
   /swarm turbo epic on
   /swarm epic status
   ```
3. Plan the build (the architect will pick up `docs/architecture.md` as the source of truth):
   ```
   /swarm plan Build fair-clinical-bench per docs/architecture.md.
   ```
4. Execute phase by phase:
   ```
   continue
   ```
   The architect should call `epic_run_phase` before each phase and surface:
   > Epic Mode: <DECISION> (p=X.XX) — <reason>

## Watching Epic Mode's decisions

While the swarm runs:

```bash
# Most recent decision (durable, cross-session)
/swarm epic last

# Full calibration state (learned threshold + hot modules + recent divergent tasks)
/swarm epic calibration

# Read-only what-if against current plan (does not write evidence)
/swarm epic decide
```

Files to watch on disk:

| Path | What it holds |
|---|---|
| `.swarm/evidence/epic-promotions.jsonl` | One JSON line per `epic_run_phase` call |
| `.swarm/epic/divergence.jsonl` | One JSON line per completed task (declared vs actual scope) |
| `.swarm/epic/calibration.json` | Current learned threshold + hot modules |

Append to `docs/epic-mode-tracking.md` after each phase to keep the public log in sync.

