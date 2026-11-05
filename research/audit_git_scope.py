"""Check Git contents are source, hyperparameters and curated result documents."""
import argparse
import collections
import csv
import fnmatch
import io
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {".py", ".sh", ".md", ".yaml", ".yml", ".json", ".csv", ".txt"}
LOCAL_ROOTS = {"artifacts", "outputs", "results", "res", "dataset", "logs", "log", "reports", "plots", "figures",
               "checkpoints", ".cache", ".venv", "venv", ".aws", ".ssh", ".codex", ".agents", "wandb", "mlruns"}
RUNTIME_NAMES = {"portfolio_return.csv", "qlib_report.csv", "orders.csv", "trades.csv", "holdings.csv", "positions.csv",
                 "epoch_metrics.csv", "training_progress.json", "training_fit.json", "training_result.json", "progress.json",
                 "monitoring.json", "watchdog.json", "resource_samples.csv", "resource_peaks.json"}
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "api_secret": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}"),
}
SENSITIVE_KEY = re.compile(r"^(?:password|api[_-]?key|client[_-]?secret|secret[_-]?key|access[_-]?key|private[_-]?key)$", re.I)


def git(*args, input=None):
    return subprocess.check_output(["git", *args], cwd=ROOT, input=input)


def candidates(index):
    if not index:
        names=set(git("ls-files", "-z").decode().strip("\0").split("\0"))
        names.update(git("ls-files", "--others", "--exclude-standard", "-z").decode().strip("\0").split("\0"))
        for name in sorted(names - {""}):
            path=ROOT/name
            if path.exists():
                yield name, path.read_bytes() if not path.is_symlink() else b"", path.is_symlink()
        return
    entries=[]
    for item in git("ls-files", "--stage", "-z").decode().strip("\0").split("\0"):
        metadata, name=item.split("\t",1)
        mode, sha, stage=metadata.split()
        if stage!="0":
            raise RuntimeError("Unmerged index entry: "+name)
        entries.append((name,sha,mode=="120000"))
    payload="".join(sha+"\n" for _,sha,_ in entries).encode()
    stream=io.BytesIO(git("cat-file", "--batch", input=payload))
    for name,sha,symlink in entries:
        header=stream.readline().decode().split()
        assert header[0]==sha and header[1]=="blob"
        data=stream.read(int(header[2])); assert stream.read(1)==b"\n"
        yield name,data,symlink


def secret_keys(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if SENSITIVE_KEY.match(key) and item:
                text=str(item).upper()
                if not any(marker in text for marker in ("EXAMPLE","REDACTED","YOUR_","PLACEHOLDER")):
                    return True
            if secret_keys(item): return True
    elif isinstance(value,list):
        return any(secret_keys(item) for item in value)
    return False


def inspect(name, data, symlink):
    path=Path(name)
    if symlink: return "symlink is not portable source/configuration"
    if path.suffix not in ALLOWED and name not in ("LICENSE", ".gitignore"):
        return "unsupported file type"
    if path.parts[0] in LOCAL_ROOTS and not (path.parts[0]=="dataset" and len(path.parts)==2 and path.suffix==".py"):
        return "local artifact/storage directory"
    if name.startswith("research/baseline/"): return "immutable local baseline evidence"
    if "分批实验清单" in path.name or fnmatch.fnmatch(path.name,".env*") and path.name!=".env.example":
        return "private plan or environment file"
    if path.name in RUNTIME_NAMES or (path.suffix in (".csv", ".json") and fnmatch.fnmatch(path.name,"*_predictions.*")):
        return "raw prediction/portfolio/runtime product"
    if len(data)>2*1024*1024: return "file exceeds 2 MiB curated-text budget"
    try: text=data.decode("utf-8-sig")
    except UnicodeDecodeError: return "non-UTF8/binary file"
    if "\0" in text: return "binary content"
    if str(Path.home()) in text: return "machine-specific home path"
    for label,pattern in SECRET_PATTERNS.items():
        if pattern.search(text): return "possible credential: "+label
    if path.suffix==".json":
        if secret_keys(json.loads(text)): return "nonempty credential configuration key"
    if path.suffix==".csv":
        header=next(csv.reader(io.StringIO(text)),[])
        columns={column.lower() for column in header}
        if "instrument" in columns and columns.intersection({"date","datetime"}):
            return "per-stock per-day data schema"
        if columns.intersection({"date","datetime"}) and "return" in columns:
            return "daily-return data schema"
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index",action="store_true",help="Audit actual staged blobs for the entire repository")
    args=parser.parse_args()
    violations=[]; count=0; size=0; extensions=collections.Counter()
    for name,data,symlink in candidates(args.index):
        count+=1; size+=len(data); extensions[Path(name).suffix or "<none>"]+=1
        reason=inspect(name,data,symlink)
        if reason: violations.append(dict(file=name,reason=reason))
    report=dict(status="passed" if not violations else "failed",view="index" if args.index else "worktree_candidates",
                files=count,bytes=size,extensions=dict(extensions),violations=violations)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if violations: raise SystemExit(1)


if __name__=="__main__":
    main()
