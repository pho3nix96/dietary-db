#!/usr/bin/env python3
"""Download the Swedish Food Agency food composition database (Livsmedelsdatabasen).

Data licence: CC BY 4.0. Credit: "Source: Swedish Food Agency (Livsmedelsverket),
Livsmedelsdatabasen, <version date>".

API documentation: https://dataportal.livsmedelsverket.se/livsmedel/swagger/index.html

Usage (Python 3.9+, standard library only):

    python scripts/fetch_livsmedelsverket.py              # writes to data/raw/livsmedelsverket/
    python scripts/fetch_livsmedelsverket.py --out DIR --workers 4

Responses are cached per food in DIR/cache/, so an interrupted run resumes
where it stopped. Delete the cache folder to force a fresh download.

Outputs:
    foods.csv            one row per food: number, English and Swedish names, type,
                         scientific name, FoodEx2 code(s), main group, data version
    nutrients_long.csv   one row per food x component, with value type and method
    nutrients_wide.csv   one row per food, one column per component (CODE_unit)
    langual.csv          LanguaL facets per food (processing, cooking, part, ...)
    components.csv       code, name, unit for every component column
    download_info.json   API version, date, number of foods
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://dataportal.livsmedelsverket.se/livsmedel/api/v1"
ENGLISH, SWEDISH = 2, 1
USER_AGENT = "food-space-map research download (open data, CC BY 4.0)"


def get_json(url: str, retries: int = 4, pause: float = 1.5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed after {retries} tries: {url}\n  {e}") from e
            time.sleep(pause * (attempt + 1))


def list_foods(lang: int, page: int = 500) -> list[dict]:
    foods, offset = [], 0
    while True:
        data = get_json(f"{BASE}/livsmedel?offset={offset}&limit={page}&sprak={lang}")
        batch = data.get("livsmedel", [])
        foods += batch
        total = data.get("_meta", {}).get("totalRecords", len(foods))
        offset += page
        if not batch or offset >= total:
            return foods


def fetch_food_details(nr: int, cache: Path) -> dict:
    f = cache / f"{nr}.json"
    try:
        d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except (json.JSONDecodeError, UnicodeDecodeError):  # truncated by an interrupted run: fetch again
        d = {}
    changed = False
    for key, ep in (("nutrients", "naringsvarden"), ("classifications", "klassificeringar"), ("raw_commodities", "ravaror")):
        if key not in d:  # older caches lack raw_commodities: fetch only what is missing
            d[key] = get_json(f"{BASE}/livsmedel/{nr}/{ep}?sprak={ENGLISH}")
            changed = True
    if changed:
        tmp = f.with_suffix(".json.tmp")  # write then rename, so a killed run never leaves a half file
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        tmp.replace(f)
    return d


def raw_commodity_rows(details: dict[int, dict]) -> list[dict]:
    """Recipe of each food in raw commodities: share of the recipe (%), the factor
    converting it to raw commodity, and grams of raw commodity per 100 g food."""
    rows = []
    for nr in sorted(details):
        for r in details[nr].get("raw_commodities") or []:
            rows.append({"food_id": nr, "foodex2_code": r.get("foodEx2"), "commodity": (r.get("namn") or "").strip(),
                         "share_pct": r.get("andel"), "factor": r.get("faktor"),
                         "raw_equivalent_g": r.get("omraknadTillRa"), "preparation": r.get("tillagning")})
    return rows


def component_key(n: dict) -> str:
    """Column name from the EuroFIR code and unit, e.g. FAT_g, ENERC_kcal, F18:2_g."""
    code = n.get("euroFIRkod") or n.get("forkortning") or n.get("namn", "unknown")
    unit = (n.get("enhet") or "").replace("µ", "u").replace("/", "per")
    return f"{code}_{unit}" if unit else code


def tidy(foods_en: list[dict], foods_sv: list[dict], details: dict[int, dict]):
    sv_name = {f["nummer"]: f.get("namn") for f in foods_sv}
    food_rows, long_rows, langual_rows, components = [], [], [], {}
    for f in foods_en:
        nr = f["nummer"]
        d = details.get(nr)
        if d is None:
            continue
        cls = d["classifications"] or []
        foodex2 = [c.get("kod") for c in cls if c.get("typ") == "FoodEx2" and c.get("kod")]
        main = [c.get("kod") for c in cls if c.get("typ") == "Main group" and c.get("kod")]
        food_rows.append({
            "food_id": nr,
            "name": f.get("namn"),
            "name_sv": sv_name.get(nr),
            "food_type": f.get("livsmedelsTyp"),
            "scientific_name": f.get("vetenskapligtNamn"),
            "project": f.get("projekt"),
            "analysis_note": f.get("analys"),
            "foodex2_code": "|".join(foodex2),
            "main_group": "|".join(main),
            "data_version": f.get("version"),
        })
        for c in cls:
            if c.get("typ") == "LanguaL":
                langual_rows.append({"food_id": nr, "facet": c.get("fasett"), "langual_id": c.get("langualId"),
                                     "facet_code": c.get("fasettkod"), "term": (c.get("namn") or "").strip()})
        for n in d["nutrients"] or []:
            key = component_key(n)
            components.setdefault(key, {"column": key, "code": n.get("euroFIRkod"), "name": n.get("namn"),
                                        "unit": n.get("enhet"), "basis": n.get("matrisenhet")})
            long_rows.append({
                "food_id": nr, "component": key, "name": n.get("namn"), "value": n.get("varde"),
                "unit": n.get("enhet"), "value_type": n.get("vardetyp"), "method_type": n.get("metodtyp"),
                "basis": n.get("matrisenhet"),
            })
    return food_rows, long_rows, langual_rows, components


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    fields = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_wide(path: Path, food_rows: list[dict], long_rows: list[dict], components: dict):
    cols = sorted(components)
    values: dict[int, dict] = {}
    for r in long_rows:
        values.setdefault(r["food_id"], {})[r["component"]] = r["value"]
    rows = []
    for f in food_rows:
        row = {"food_id": f["food_id"], "name": f["name"], "foodex2_code": f["foodex2_code"], "main_group": f["main_group"]}
        row.update({c: values.get(f["food_id"], {}).get(c, "") for c in cols})
        rows.append(row)
    write_csv(path, rows, ["food_id", "name", "foodex2_code", "main_group"] + cols)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/livsmedelsverket", help="output folder")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (keep it small; it is a public service)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N foods (for a quick test)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    info = get_json(f"{BASE}/api-info")
    print(f"API: {info.get('apiName')} {info.get('apiVersion')} ({info.get('apiStatus')})")
    foods_en = list_foods(ENGLISH)
    foods_sv = list_foods(SWEDISH)
    if args.limit:
        foods_en = foods_en[: args.limit]
    print(f"{len(foods_en)} foods listed; downloading nutrients, classifications and raw commodities...")

    details, failed = {}, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fetch_food_details, f["nummer"], cache): f["nummer"] for f in foods_en}
        for i, fut in enumerate(as_completed(futures), 1):
            nr = futures[fut]
            try:
                details[nr] = fut.result()
            except Exception as e:  # keep going; report at the end
                failed.append(nr)
                print(f"  food {nr}: {e}", file=sys.stderr)
            if i % 100 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} foods ({time.time() - t0:.0f} s)")

    food_rows, long_rows, langual_rows, components = tidy(foods_en, foods_sv, details)
    write_csv(out / "foods.csv", food_rows)
    write_csv(out / "nutrients_long.csv", long_rows)
    write_csv(out / "langual.csv", langual_rows)
    write_csv(out / "components.csv", list(components.values()))
    write_csv(out / "raw_commodities.csv", raw_commodity_rows(details),
              ["food_id", "foodex2_code", "commodity", "share_pct", "factor", "raw_equivalent_g", "preparation"])
    write_wide(out / "nutrients_wide.csv", food_rows, long_rows, components)
    (out / "download_info.json").write_text(json.dumps({
        "source": "Swedish Food Agency (Livsmedelsverket), Livsmedelsdatabasen",
        "licence": "CC BY 4.0",
        "api": info,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "foods": len(food_rows),
        "components": len(components),
        "failed_food_ids": failed,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Done: {len(food_rows)} foods, {len(components)} components -> {out}/")
    if failed:
        print(f"{len(failed)} foods failed; run again to retry them (cached foods are skipped).")


if __name__ == "__main__":
    main()
