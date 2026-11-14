# MemoryScapes workers

The processing pipeline that turns an upload into a memory. The DAG is defined in
[`memoryscapes_workers/pipeline.py`](memoryscapes_workers/pipeline.py):

```
ingest ─┬─> poses ──> splat ─┬─> localize ──> soundscape ──> publish
        └─> separate ────────┘
```

| Stage | Module | Hardware |
|---|---|---|
| Ingest | `stages/ingest.py` | CPU |
| Camera poses | `stages/poses.py` | GPU |
| Splat training | `stages/splat.py` | GPU |
| Separation + labeling | `stages/separate.py` | GPU |
| Localization | `stages/localize.py` | GPU |
| Soundscape (SATIE) | `stages/soundscape.py` + `providers/soundscape.py` | external |
| Privacy + publish | `stages/publish.py` | GPU/CPU |

```bash
cd services/workers
pip install -e ".[dev]"            # core only
pip install -e ".[visual,audio]"   # GPU/ML stacks
```

Set `SOUNDSCAPE_PROVIDER=stub` to develop without SATIE.
