from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
RECORDS_DIR = ROOT / "simulation-records"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": None,
        "torch_version": None,
        "cuda_available": False,
        "torch_npu": False,
        "torch_npu_version": None,
        "npu_available": False,
    }
    try:
        import torch

        env["torch"] = True
        env["torch_version"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        env["torch_error"] = str(exc)
    try:
        import torch_npu

        env["torch_npu"] = True
        env["torch_npu_version"] = getattr(torch_npu, "__version__", "unknown")
        try:
            import torch

            env["npu_available"] = bool(torch.npu.is_available())
        except Exception:
            pass
    except Exception as exc:
        env["torch_npu_error"] = str(exc)
    return env


class RunRecorder:
    def __init__(self, task_id: str, meta: dict[str, Any]):
        self.task_id = task_id
        self.meta = meta
        self.steps: list[dict[str, Any]] = []
        self.started_at = utc_now()

    def add(self, kind: str, text: str, **extra: Any) -> None:
        step = {"kind": kind, "text": text, "ts": utc_now(), **extra}
        self.steps.append(step)

    def finish(self, outcome: str) -> dict[str, Any]:
        record = {
            **self.meta,
            "id": self.task_id,
            "live": True,
            "started_at": self.started_at,
            "finished_at": utc_now(),
            "environment": detect_environment(),
            "steps": self.steps,
            "outcome": outcome,
        }
        RECORDS_DIR.mkdir(parents=True, exist_ok=True)
        path = RECORDS_DIR / f"{self.task_id}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 120) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def fetch_url(url: str, timeout: int = 20, _depth: int = 0) -> dict[str, Any]:
    import re
    import urllib.error
    import urllib.request
    from urllib.parse import urljoin

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; cuda-cann-workflow-runner/1.1)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            final_url = resp.geturl()
            body = resp.read(200_000).decode("utf-8", errors="replace")
            title = _extract_title(body)
            snippet = _extract_snippet(body)

            # PyTorch docs 等页面常有 meta/JS 跳转，再抓一层
            if _depth < 2 and ("Redirecting" in title or "location.replace" in body):
                nxt = _extract_redirect_target(body, final_url)
                if nxt and nxt != url:
                    deeper = fetch_url(nxt, timeout=timeout, _depth=_depth + 1)
                    deeper["redirect_from"] = url
                    return deeper

            return {
                "url": url,
                "final_url": final_url,
                "status": resp.status,
                "title": title,
                "snippet": snippet,
                "found_keywords": _scan_keywords(body, ["add", "element", "tensor", "custom", "operator"]),
            }
    except urllib.error.HTTPError as exc:
        return {"url": url, "status": exc.code, "error": str(exc)}
    except Exception as exc:
        return {"url": url, "status": None, "error": str(exc)}


def _extract_title(body: str) -> str:
    import re

    m = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1)).strip()


def _extract_snippet(body: str, limit: int = 420) -> str:
    import re

    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", body)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _extract_redirect_target(body: str, base: str) -> str | None:
    import re
    from urllib.parse import urljoin

    for pat in [
        r'location\.replace\("([^"]+)"',
        r'<link rel="canonical" href="([^"]+)"',
        r'<a href="([^"]+generated/torch[^"]+)"',
    ]:
        m = re.search(pat, body)
        if m:
            return urljoin(base, m.group(1))
    return None


def _scan_keywords(body: str, keywords: list[str]) -> list[str]:
    low = body.lower()
    return [k for k in keywords if k in low]


def search_pages(
    rec: RunRecorder,
    plan: str,
    pages: list[tuple[str, str]],
    keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    """多页面检索：每页单独一步，贴近 Claude 的 Web Search 过程。"""
    kw = keywords or ["add", "operator", "custom", "support"]
    rec.add(
        "think",
        plan
        + f"\n\n🔎 检索计划：共 {len(pages)} 个页面（Claude 类助手通常也会跨页面对照，而非只看一页）。",
    )
    hits: list[dict[str, Any]] = []
    for i, (name, url) in enumerate(pages, 1):
        fetched = fetch_url(url)
        if "found_keywords" not in fetched or not fetched["found_keywords"]:
            fetched["found_keywords"] = _scan_keywords(
                fetched.get("snippet", ""), kw
            )
        rec.add(
            "search",
            f"页面 {i}/{len(pages)} · {name}",
            page_index=i,
            page_total=len(pages),
            page_name=name,
            **fetched,
        )
        hits.append({"name": name, **fetched})
    rec.add(
        "tool",
        f"汇总 {len(hits)} 页检索结果",
        pages_checked=len(hits),
        ok_pages=sum(1 for h in hits if h.get("status") == 200),
        page_summary=[{"name": h["name"], "url": h.get("final_url") or h.get("url"), "status": h.get("status")} for h in hits],
    )
    return hits


def run_python(code: str, timeout: int = 120) -> dict[str, Any]:
    return run_cmd([sys.executable, "-c", code], timeout=timeout)


def grep_repo(pattern: str, path: str) -> dict[str, Any]:
    target = ROOT / path
    rg = run_cmd(["rg", "-n", pattern, path])
    if rg["exit_code"] == 0 or rg.get("stdout"):
        return rg
    return run_cmd(["grep", "-Rn", pattern, str(target)])


def run_task(task_id: str, fn: Callable[[RunRecorder], str]) -> dict[str, Any]:
    from runner.tasks import TASKS

    if task_id not in TASKS:
        raise KeyError(f"Unknown task id: {task_id}")
    meta = TASKS[task_id]["meta"]
    rec = RunRecorder(task_id, meta)
    outcome = fn(rec)
    return rec.finish(outcome)
