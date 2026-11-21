import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional

USER_AGENT = "DigitalHiveAgent/2.0"

KNOWN_DOC_TITLES = {
    "01_fz490_pesticides_protocol.md": "ФЗ № 490-ФЗ и пестициды",
    "02_swarming_microclimate_acoustics.md": "Микроклимат, роение и акустика",
    "03_migration_and_weather_rules.md": "Погода, осмотры и кочёвка",
    "04_varroa_biosecurity_and_limits.md": "Varroa и ветправила",
}

def parse_kb_ids(raw_value) -> list[str]:
    if not raw_value:
        return []
    if isinstance(raw_value, str):
        items = [x.strip() for x in raw_value.replace(";", ",").split(",")]
    elif isinstance(raw_value, (list, tuple, set)):
        items = [str(x).strip() for x in raw_value]
    else:
        return []
    seen = []
    for it in items:
        if it and it not in seen:
            seen.append(it)
    return seen[:5]

def build_server_file_search_tool(
    kb_ids: list[str],
    max_results: int = 4,
    rerank: Optional[bool] = None,
    max_uses: int = 2,
) -> Optional[dict]:
    clean_ids = parse_kb_ids(kb_ids)
    if not clean_ids:
        return None
    params = {
        "knowledge_base_ids": clean_ids,
        "max_results": max(1, min(20, int(max_results))),
        "max_uses": max(1, min(5, int(max_uses))),
    }
    if rerank is not None:
        params["rerank"] = bool(rerank)
    return {
        "type": "aitunnel:file_search",
        "parameters": params,
    }

def _extract_heading_from_excerpt(path: str, excerpt: str) -> str:
    if not excerpt:
        return ""
    first_line = excerpt.strip().splitlines()[0].strip()
    if " — " in first_line:
        heading_part = first_line.split(" — ", 1)[1].strip()
        if ">" in heading_part:
            heading_part = heading_part.split(">")[-1].strip()
        return heading_part[:60]
    m = re.match(r"^#{1,4}\s+(.+)$", first_line)
    if m:
        return m.group(1).strip()[:60]
    return ""

def _humanize_doc_name(path: str) -> str:
    base = os.path.basename(str(path or "").strip())
    if not base:
        return "Справочник"
    if base in KNOWN_DOC_TITLES:
        return KNOWN_DOC_TITLES[base]
    stem = os.path.splitext(base)[0]
    stem = re.sub(r"^[0-9]+[_\-\s]+", "", stem)
    stem = stem.replace("_", " ").replace("-", " ").strip()
    return stem if stem else base

def format_citation_badges(kb_res: dict, max_sources: int = 4) -> list[str]:
    if not kb_res or kb_res.get("status") != "ok":
        return []
    fragments = kb_res.get("fragments") or []
    cnt = len(fragments)
    if cnt == 0:
        return []

    badges = [f"📚 База знаний ({cnt} фрагм.)"]
    seen_docs = set()
    for frag in fragments:
        path = str(frag.get("source_document") or "").strip()
        doc_title = _humanize_doc_name(path)
        doc_key = doc_title.lower()
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)
        badges.append(f"📖 {doc_title}")
        if len(seen_docs) >= max_sources:
            break
    return badges

def search_single_knowledge_base(
    kb_id: str,
    query: str | list[str],
    api_key: str,
    base_url: str = "https://api.aitunnel.ru/v1/",
    max_results: int = 4,
    rerank: Optional[bool] = None,
    timeout: float = 10.0,
) -> dict:
    if not api_key:
        return {"status": "error", "error_type": "no_api_key", "message": "Не задан API-ключ AITUNNEL.", "results": []}
    if not kb_id or not str(kb_id).strip():
        return {"status": "error", "error_type": "no_kb_id", "message": "Не задан ID базы знаний.", "results": []}

    if isinstance(query, list):
        clean_q = [str(q).strip()[:2000] for q in query if str(q).strip()][:5]
        if not clean_q:
            return {"status": "error", "error_type": "empty_query", "message": "Пустой поисковый запрос.", "results": []}
        q_payload = clean_q if len(clean_q) > 1 else clean_q[0]
    else:
        q_str = str(query or "").strip()[:2000]
        if not q_str:
            return {"status": "error", "error_type": "empty_query", "message": "Пустой поисковый запрос.", "results": []}
        q_payload = q_str

    endpoint = f"{base_url.rstrip('/')}/knowledge-bases/{kb_id.strip()}/search"
    body = {
        "query": q_payload,
        "max_results": max(1, min(20, int(max_results))),
    }
    if rerank is not None:
        body["rerank"] = bool(rerank)

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        flat_results = []
        for item in payload.get("data") or []:
            q_used = item.get("query", "")
            for hit in item.get("results") or []:
                flat_results.append({
                    "knowledge_base_id": hit.get("knowledge_base_id", kb_id),
                    "path": hit.get("path", ""),
                    "score": round(float(hit.get("score", 0.0)), 3),
                    "text": str(hit.get("text", "")).strip(),
                    "matched_query": q_used,
                })
        return {
            "status": "ok",
            "results": flat_results,
            "rerank_model": payload.get("rerank_model"),
            "usage": payload.get("usage") or {},
        }
    except urllib.error.HTTPError as he:
        err_type = f"http_{he.code}"
        err_msg = f"Ошибка поиска по базе знаний (HTTP {he.code})."
        try:
            err_body = json.loads(he.read().decode("utf-8"))
            err_obj = err_body.get("error") or {}
            err_type = err_obj.get("type") or err_type
            if err_obj.get("message"):
                err_msg = str(err_obj["message"])
        except Exception:
            pass
        if he.code == 404:
            err_msg = f"База знаний «{kb_id}» не найдена. Проверьте ID базы в настройках."
        elif he.code == 402:
            err_msg = "Недостаточно баланса или исчерпан бюджет API-ключа для поиска по базе знаний."
        elif he.code == 503:
            err_msg = "Поиск по базе знаний временно недоступен."
        return {"status": "error", "error_type": err_type, "message": err_msg, "results": []}
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": "Не удалось обратиться к базе знаний (ошибка сети или таймаут).",
            "results": [],
        }

def search_knowledge_bases(
    kb_ids: list[str],
    queries: str | list[str],
    api_key: str,
    base_url: str = "https://api.aitunnel.ru/v1/",
    max_results: int = 4,
    rerank: Optional[bool] = None,
    min_score: float = 0.15,
    timeout: float = 10.0,
) -> dict:
    clean_ids = parse_kb_ids(kb_ids)
    if not clean_ids:
        return {
            "status": "not_configured",
            "message": "База знаний AITUNNEL не подключена (не задан ID вида kb_...).",
            "fragments_count": 0,
            "fragments": [],
        }

    all_hits = []
    errors = []
    used_rerank = False
    for kb_id in clean_ids:
        res = search_single_knowledge_base(
            kb_id=kb_id,
            query=queries,
            api_key=api_key,
            base_url=base_url,
            max_results=max_results,
            rerank=rerank,
            timeout=timeout,
        )
        if res["status"] == "ok":
            all_hits.extend(res["results"])
            if res.get("rerank_model"):
                used_rerank = True
        else:
            errors.append(res["message"])

    if not all_hits and errors:
        return {
            "status": "error",
            "message": " ".join(errors),
            "fragments_count": 0,
            "fragments": [],
        }

    all_hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
    dedup = []
    seen_keys = set()
    for hit in all_hits:
        dedup_key = (hit.get("path", ""), hit.get("text", "")[:140])
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        dedup.append(hit)

    # При включённом реранкере шкала оценок может быть ниже косинусной близости — учитываем это
    effective_min = min(min_score, 0.05) if used_rerank else min_score
    filtered = [h for h in dedup if h.get("score", 0.0) >= effective_min]
    if not filtered and dedup and dedup[0].get("score", 0.0) >= 0.08:
        filtered = dedup[:1]

    top_fragments = filtered[: max(1, min(10, int(max_results)))]
    compact_fragments = [
        {
            "source_document": f["path"],
            "source_title": _humanize_doc_name(f["path"]),
            "section_title": _extract_heading_from_excerpt(f["path"], f["text"]),
            "relevance_score": f["score"],
            "excerpt": f["text"][:1800],
        }
        for f in top_fragments
    ]

    if not compact_fragments:
        return {
            "status": "empty",
            "message": "В подключённой базе знаний не найдено релевантных фрагментов по этому вопросу.",
            "fragments_count": 0,
            "fragments": [],
        }

    return {
        "status": "ok",
        "fragments_count": len(compact_fragments),
        "fragments": compact_fragments,
        "partial_warnings": errors if errors else [],
    }
