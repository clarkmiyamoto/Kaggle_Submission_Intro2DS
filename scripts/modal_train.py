"""Run configs on Modal GPUs (https://modal.com), so long runs don't need a laptop.

One-time setup, from the repo root:
    uv run modal setup                     # log in; writes ~/.modal.toml
    bash scripts/modal_upload_data.sh      # data/ -> the kaggle-rl-data volume (~3.4 GB)

Train: commit first, since the container runs HEAD. Each config gets its own GPU container
(`--module src.retrain` runs src/retrain.py instead of src/run.py):
    uv run modal run --detach scripts/modal_train.py::main --configs configs/host/a.yaml,configs/host/b.yaml --gpu A10
--detach keeps the runs going if this terminal closes. Logs stream here and go to the
outputs volume (logs/<config name>.log; `uv run modal volume get kaggle-rl-outputs logs/x.log -`).

Fetch results/*.json and runs/<run_id>/ that this checkout doesn't have yet (--name keeps only
paths containing that text; without it, every run ever written to the volume comes down):
    uv run modal run scripts/modal_train.py::fetch --name final_full24_wide2_drop0.05_trval

How it works: the local side packs HEAD into a git bundle and sends it with each call. The
container clones it, links data/ to the data volume and runs `python -m src.run <config>`
exactly as on a laptop, so results/<run_id>.json gets the same SHA and dirty flag. New
results/ and runs/ files are copied to the kaggle-rl-outputs volume.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/root/repo")  # inside the container
DATA = Path("/data")
OUT = Path("/outputs")

data_vol = modal.Volume.from_name("kaggle-rl-data", create_if_missing=True)
out_vol = modal.Volume.from_name("kaggle-rl-outputs", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.13")
         .apt_install("git")
         .uv_sync(str(ROOT), extra_options="--no-dev"))
app = modal.App("kaggle-rl", image=image)


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _bundle() -> tuple[bytes, str]:
    """HEAD as a git bundle, refusing uncommitted changes to tracked code (as run.py's dirty flag)."""
    if _git("status", "--porcelain", "--", ".", ":!results", ":!runs"):
        sys.exit("uncommitted changes: commit them first; the container runs HEAD")
    sha = _git("rev-parse", "HEAD")
    bundle = subprocess.run(["git", "bundle", "create", "-", "HEAD"], cwd=ROOT, capture_output=True,
                            check=True).stdout
    return bundle, sha


def _checkout(bundle: bytes, sha: str) -> None:
    shutil.rmtree(REPO, ignore_errors=True)  # Modal may reuse a container for the next queued config
    Path("/tmp/repo.bundle").write_bytes(bundle)
    subprocess.run(["git", "clone", "-q", "/tmp/repo.bundle", str(REPO)], check=True)
    subprocess.run(["git", "checkout", "-q", sha], cwd=REPO, check=True)
    (REPO / "data").symlink_to(DATA)  # gitignored by `/data`, as in every worktree


def _stream(cmd: list[str], log: Path) -> int:
    """Run cmd in the repo, echoing its output and writing it to log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
    return proc.wait()


@app.function(gpu="L4", volumes={DATA: data_vol, OUT: out_vol}, timeout=24 * 3600, cpu=4.0, memory=24 * 1024)
def train(config: str, bundle: bytes, sha: str, module: str = "src.run") -> list[str]:
    _checkout(bundle, sha)
    tracked = set((REPO / "results").glob("*.json"))
    rc = _stream([sys.executable, "-u", "-m", module, config], OUT / "logs" / f"{Path(config).stem}.log")
    new = sorted(set((REPO / "results").glob("*.json")) - tracked)
    for f in new:
        (OUT / "results").mkdir(exist_ok=True)
        shutil.copy2(f, OUT / "results" / f.name)
        shutil.copytree(REPO / "runs" / f.stem, OUT / "runs" / f.stem, dirs_exist_ok=True)
    dumps = REPO / "runs" / "snapdump"  # HostNet snapshot_dump files, if the config wrote any
    if dumps.is_dir():
        shutil.copytree(dumps, OUT / "runs" / "snapdump", dirs_exist_ok=True)
    out_vol.commit()
    if rc:
        raise RuntimeError(f"{config}: src.run exited with {rc}")
    return [f.name for f in new]


@app.local_entrypoint()
def main(configs: str, gpu: str = "L4", module: str = "src.run"):
    bundle, sha = _bundle()
    paths = [c.strip() for c in configs.split(",") if c.strip()]
    for p in paths:
        assert (ROOT / p).is_file(), f"no such config: {p}"
    print(f"HEAD {sha[:7]}: {len(paths)} run(s) of {module} on {gpu}")
    calls = [(p, train.with_options(gpu=gpu).spawn(p, bundle, sha, module)) for p in paths]
    for p, call in calls:
        try:
            print(f"done {p}: {call.get()}")
        except Exception as e:  # keep waiting on the others
            print(f"FAILED {p}: {e}")
    print("fetch with: uv run modal run scripts/modal_train.py::fetch")


@app.local_entrypoint()
def fetch(name: str = ""):
    """Copy results/*.json and runs/<run_id>/ files from the outputs volume that are missing here,
    only those whose path contains `name` when it is given."""
    n = 0
    for top in ("results", "runs"):
        try:
            entries = out_vol.listdir(top, recursive=True)
        except Exception:  # nothing written yet
            continue
        for e in entries:
            if e.type != modal.volume.FileEntryType.FILE or name not in e.path:
                continue
            dest = ROOT / e.path
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                for chunk in out_vol.read_file(e.path):
                    f.write(chunk)
            n += 1
            print(f"  {e.path}")
    print(f"fetched {n} file(s)")
