"""Score 10 focus metrics from simulation records (phase 1 · s1-q1 pair)."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]

# --- Gold sets for 环节 1 · 查官方文档 · vector add ---

CUDA_GOLD_URLS = [
    "docs.pytorch.org",
    "pytorch.org/docs",
    "pytorch.org/tutorials",
]

CANN_GOLD_URLS = [
    "hiascend.com",
    "gitee.com/ascend/pytorch",
]

CUDA_OFFICIAL = CUDA_GOLD_URLS
CANN_OFFICIAL = ["hiascend.com"]

S1_CHECKLIST = {
    "CUDA": [
        "torch.add 原生 API",
        "官方文档链接",
        "自研/组合决策结论",
        "本机/环境说明",
        "Custom Op 边界说明",
    ],
    "CANN": [
        "torch_npu / Add 支持说明",
        "版本/CANN 配套要求",
        "官方文档链接",
        "NPU 试跑/环境说明",
        "明确支持/不支持结论",
    ],
}

S1_BASELINE_STS = 4  # 专家最短路径：检索→汇总→结论


def load_record(record_id: str) -> dict:
    fp = ROOT / "simulation-records" / f"{record_id}.json"
    return json.loads(fp.read_text(encoding="utf-8"))


def _urls_from_record(rec: dict) -> list[str]:
    urls: list[str] = []
    for step in rec.get("steps", []):
        if step.get("kind") == "search":
            u = step.get("final_url") or step.get("url") or ""
            if u:
                urls.append(u)
    return urls


def _full_text(rec: dict) -> str:
    parts = [rec.get("outcome", ""), rec.get("prompt", "")]
    for step in rec.get("steps", []):
        parts.append(step.get("text", ""))
        parts.append(step.get("snippet", "") or "")
        parts.append(step.get("url", "") or "")
        parts.append(step.get("final_url", "") or "")
    return "\n".join(parts)


def _reply_text(rec: dict) -> str:
    for step in reversed(rec.get("steps", [])):
        if step.get("kind") == "reply":
            return step.get("text", "")
    return rec.get("outcome", "")


def _stack(rec: dict) -> str:
    return rec.get("stack", "CUDA")


def _url_matches_gold(url: str, patterns: list[str]) -> bool:
    low = url.lower()
    return any(p in low for p in patterns)


def _url_official(url: str, stack: str) -> bool:
    patterns = CUDA_OFFICIAL if stack == "CUDA" else CANN_OFFICIAL
    return _url_matches_gold(url, patterns)


def score_recall_at_k(rec: dict, k: int = 5) -> dict:
    stack = _stack(rec)
    gold = CUDA_GOLD_URLS if stack == "CUDA" else CANN_GOLD_URLS
    searches = [s for s in rec.get("steps", []) if s.get("kind") == "search"][:k]
    if not gold:
        return {"value": None, "detail": "无 Gold 集"}
    hits = sum(
        1
        for s in searches
        if s.get("status") == 200 and _url_matches_gold(s.get("final_url") or s.get("url", ""), gold)
    )
    # 相关文档池大小 = 该栈 s1 预期检索页数
    relevant_pool = len(gold)
    val = min(1.0, hits / relevant_pool)
    return {
        "value": round(val, 3),
        "detail": f"Top-{k} 检索中 {hits} 页命中 Gold 域（池 {relevant_pool}）",
        "auto": "semi",
    }


def score_mrr(rec: dict) -> dict:
    stack = _stack(rec)
    gold = CUDA_GOLD_URLS if stack == "CUDA" else CANN_GOLD_URLS
    rank = 0
    for step in rec.get("steps", []):
        if step.get("kind") != "search":
            continue
        rank += 1
        url = step.get("final_url") or step.get("url", "")
        if step.get("status") == 200 and _url_matches_gold(url, gold):
            return {"value": round(1.0 / rank, 3), "detail": f"首个相关检索为第 {rank} 次 search", "auto": "semi"}
    return {"value": 0.0, "detail": "未命中 Gold 官方页", "auto": "semi"}


def score_npl(rec: dict) -> dict:
    sts = sum(1 for s in rec.get("steps", []) if s.get("kind") in ("search", "tool", "reply"))
    val = sts / S1_BASELINE_STS
    return {
        "value": round(val, 3),
        "detail": f"STS={sts}，baseline={S1_BASELINE_STS}",
        "auto": "semi",
    }


def score_ttc(rec: dict) -> dict:
    start = rec.get("started_at")
    reply_ts = None
    for step in rec.get("steps", []):
        if step.get("kind") == "reply":
            reply_ts = step.get("ts")
    if not start or not reply_ts:
        return {"value": None, "detail": "缺少时间戳", "auto": "semi"}
    t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
    t1 = datetime.fromisoformat(reply_ts.replace("Z", "+00:00"))
    sec = (t1 - t0).total_seconds()
    return {"value": round(sec, 2), "detail": f"{sec:.1f}s（started→reply）", "auto": "semi", "unit": "s"}


def score_osr(rec: dict) -> dict:
    stack = _stack(rec)
    urls = _urls_from_record(rec)
    if not urls:
        return {"value": 0.0, "detail": "无引用 URL", "auto": "full"}
    official = sum(1 for u in urls if _url_official(u, stack))
    val = official / len(urls)
    return {
        "value": round(val, 3),
        "detail": f"{official}/{len(urls)} URL 为栈官方域",
        "auto": "full",
    }


def score_ver_acc(rec: dict) -> dict:
    """Version Accuracy：回答中版本/配套陈述与检索到的官方页是否一致。"""
    text = _reply_text(rec)
    prompt = rec.get("prompt", "")
    requires_version = "版本" in prompt

    def _gold_versions() -> set[str]:
        found: set[str] = set()
        for step in rec.get("steps", []):
            blob = " ".join(
                filter(
                    None,
                    [
                        step.get("final_url") or "",
                        step.get("url") or "",
                        step.get("title") or "",
                        step.get("snippet") or "",
                    ],
                )
            )
            if "60RC1" in blob or "6.0.RC1" in blob:
                found.add("6.0.RC1")
            if "7.0.0" in blob or "商用版7.0" in blob:
                found.add("7.0.0")
            if "2.12" in blob:
                found.add("2.12")
            for m in re.finditer(r"\d+\.\d+(?:\.\d+)?(?:\.RC\d+)?", blob):
                found.add(m.group(0))
        return found

    claims = re.findall(
        r"\d+\.\d+(?:\.\d+)?(?:\.RC\d+)?|CANN\s*[\d.]+|PyTorch\s*[\d.]+|6\.0\.?RC\d",
        text,
        re.I,
    )
    gold = _gold_versions()

    if requires_version and not claims:
        return {
            "value": 0.0,
            "detail": "任务要求注明版本但未给出可核查版本陈述",
            "auto": "semi",
        }

    if not claims:
        return {
            "value": 1.0,
            "detail": "无版本陈述且无错误版本（或未要求版本）",
            "auto": "semi",
        }

    def _claim_ok(claim: str) -> bool:
        c = re.sub(r"\s+", "", claim.lower())
        for g in gold:
            gn = re.sub(r"\s+", "", g.lower())
            if gn in c or c in gn:
                return True
        nums = re.findall(r"\d+\.\d+", claim)
        return any(any(n in g for g in gold) for n in nums)

    correct = sum(1 for c in claims if _claim_ok(c))
    val = correct / len(claims)
    wrong = [c for c in claims if not _claim_ok(c)]
    return {
        "value": round(val, 3),
        "detail": f"版本陈述 {correct}/{len(claims)} 与检索页一致"
        + (f"；未对齐：{', '.join(wrong)}" if wrong else ""),
        "auto": "semi",
    }


def _checklist_hits(rec: dict) -> list[bool]:
    stack = _stack(rec)
    text = _full_text(rec).lower()
    items = S1_CHECKLIST[stack]
    hits = []
    rules = {
        "CUDA": [
            lambda t: "torch.add" in t or "add_" in t,
            lambda t: "pytorch.org" in t or "docs.pytorch" in t,
            lambda t: any(x in t for x in ("不必自研", "无需自研", "原生", "不必", "不需要")),
            lambda t: "torch" in t or "环境" in t or "冒烟" in t,
            lambda t: "custom" in t or "extension" in t or "教程" in t or "docs.pytorch" in t,
        ],
        "CANN": [
            lambda t: "add" in t and ("torch" in t or "npu" in t or "映射" in t),
            lambda t: any(x in t for x in ("版本", "rc", "cann", "6.0", "7.0")),
            lambda t: "hiascend" in t or "ascend" in t,
            lambda t: "npu" in t or "试跑" in t or "环境" in t,
            lambda t: any(x in t for x in ("支持", "确认", "映射", "需", "结论")),
        ],
    }
    for fn in rules[stack]:
        hits.append(bool(fn(text)))
    return hits


def score_cov_checklist(rec: dict) -> dict:
    hits = _checklist_hits(rec)
    val = sum(hits) / len(hits) if hits else 0
    checklist = S1_CHECKLIST[_stack(rec)]
    missed = [checklist[i] for i, h in enumerate(hits) if not h]
    return {
        "value": round(val, 3),
        "detail": f"命中 {sum(hits)}/{len(hits)}" + (f"；漏：{', '.join(missed)}" if missed else ""),
        "auto": "semi",
    }


def score_pass_at_1(rec: dict) -> dict:
    has_code = bool(re.search(r"```|def |import torch", _full_text(rec)))
    if not has_code:
        return {"value": None, "detail": "环节 1 无代码块 · N/A", "auto": "full", "na": True}
    return {"value": 0.0, "detail": "待 sandbox 试跑", "auto": "full"}


def score_ttfd(rec: dict) -> dict:
    reply = _reply_text(rec)
    if not reply:
        return {"value": None, "detail": "无 reply", "auto": "semi"}
    decision_patterns = [
        r"不必自研", r"无需自研", r"不需要", r"用原生", r"→", r"结论",
        r"需在有 NPU", r"再试跑", r"确认",
    ]
    pos = len(reply)
    for pat in decision_patterns:
        m = re.search(pat, reply)
        if m:
            pos = min(pos, m.start())
    # 粗略 token：字符比
    val = pos / max(len(reply), 1)
    return {
        "value": round(val, 3),
        "detail": f"决策句约在 reply 前 {int(val*100)}% 位置（字符近似）",
        "auto": "semi",
    }


def score_csc(rec: dict) -> dict:
    has_code = bool(re.search(r"```", _full_text(rec)))
    if not has_code:
        return {"value": None, "detail": "环节 1 无代码 · N/A", "auto": "full", "na": True}
    return {"value": None, "detail": "N/A", "auto": "full", "na": True}


METRIC_ORDER = [
    ("Recall@k", score_recall_at_k),
    ("MRR", score_mrr),
    ("NPL", score_npl),
    ("TTC", score_ttc),
    ("OSR", score_osr),
    ("VerAcc", score_ver_acc),
    ("Cov@checklist", score_cov_checklist),
    ("pass@1", score_pass_at_1),
    ("TTFD", score_ttfd),
    ("CSC", score_csc),
]

METRIC_META = {
    "Recall@k": {"module": "检索深度与精准度", "layer": "执行层", "layer_key": "exec"},
    "MRR": {"module": "检索深度与精准度", "layer": "执行层", "layer_key": "exec"},
    "NPL": {"module": "路径效率", "layer": "执行层", "layer_key": "exec"},
    "TTC": {"module": "多轮收敛", "layer": "执行层", "layer_key": "exec"},
    "OSR": {"module": "官方信息占比", "layer": "内容层", "layer_key": "content"},
    "VerAcc": {"module": "信息时效性与 API 幻觉", "layer": "内容层", "layer_key": "content"},
    "Cov@checklist": {"module": "知识覆盖率", "layer": "内容层", "layer_key": "content"},
    "pass@1": {"module": "可执行性与完整性", "layer": "输出层", "layer_key": "output"},
    "TTFD": {"module": "快速给答", "layer": "输出层", "layer_key": "output"},
    "CSC": {"module": "代码简洁性", "layer": "输出层", "layer_key": "output"},
}


def score_record(rec: dict) -> dict[str, Any]:
    out = {}
    for name, fn in METRIC_ORDER:
        out[name] = fn(rec)
        out[name]["module"] = METRIC_META[name]["module"]
        out[name]["layer"] = METRIC_META[name]["layer"]
    return out


def score_pair(cuda_id: str, cann_id: str) -> dict:
    cuda = load_record(cuda_id)
    cann = load_record(cann_id)
    cuda_scores = score_record(cuda)
    cann_scores = score_record(cann)

    rows = []
    for name, _ in METRIC_ORDER:
        c = cuda_scores[name]
        n = cann_scores[name]
        delta = None
        if c.get("value") is not None and n.get("value") is not None and not c.get("na") and not n.get("na"):
            # NPL / TTC / TTFD：越低越好或 contextual
            if name in ("NPL", "TTFD"):
                delta = round(n["value"] - c["value"], 3)
            elif name == "TTC":
                delta = round(n["value"] - c["value"], 2)
            else:
                delta = round(n["value"] - c["value"], 3)
        rows.append({
            "metric": name,
            "module": METRIC_META[name]["module"],
            "layer": METRIC_META[name]["layer"],
            "layer_key": METRIC_META[name]["layer_key"],
            "cuda": c,
            "cann": n,
            "delta": delta,
        })

    return {
        "pair_id": "s1-q1",
        "scene": "环节 1 · 查官方文档",
        "cuda_id": cuda_id,
        "cann_id": cann_id,
        "cuda_prompt": cuda.get("prompt"),
        "cann_prompt": cann.get("prompt"),
        "scored_at": datetime.utcnow().isoformat() + "Z",
        "metrics": rows,
    }


def save_pair_result(out_path: Path | None = None) -> dict:
    result = score_pair("cuda-s1-q1", "cann-s1-q1")
    fp = out_path or ROOT / "metrics-results" / "s1-q1-pair.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = save_pair_result()
    print(json.dumps(r, ensure_ascii=False, indent=2))
