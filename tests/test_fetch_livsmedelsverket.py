"""Offline test of the Livsmedelsverket download script, using responses shaped
like the real API (checked against the live API on 2026-10-18)."""

import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_livsmedelsverket.py"
spec = importlib.util.spec_from_file_location("fetch_lv", SCRIPT)
lv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lv)


def nutrient(code, name, unit, value, vtype="Best estimate"):
    return {"euroFIRkod": code, "forkortning": code, "namn": name, "enhet": unit, "varde": value,
            "vardetyp": vtype, "metodtyp": "Analytical result(s)", "matrisenhet": "per 100 g edible portion"}


FOODS = {
    1: ("Beef tallow", "Talg, nöt", "A037X", "Other fats (lard, tallow, coconut oil)",
        [nutrient("FAT", "Fat, total", "g", 99.0), nutrient("FASAT", "Fatty acids, total saturated", "g", 50.1),
         nutrient("ENERC", "Energy (kJ)", "kJ", 3700), nutrient("ENERC", "Energy (kcal)", "kcal", 890)]),
    2: ("Milk 3% fat", "Mjölk 3%", "A02MC", "Milk",
        [nutrient("FAT", "Fat, total", "g", 3.0), nutrient("PROT", "Protein", "g", 3.4),
         nutrient("DISAC", "Disaccharides", "g", 4.8), nutrient("ENERC", "Energy (kcal)", "kcal", 60)]),
}


def fake_get_json(url, retries=4, pause=1.5):
    if url.endswith("/api-info"):
        return {"apiName": "LivsmedelData.API", "apiVersion": "1.0.0", "apiStatus": "active"}
    if "/livsmedel?" in url:
        lang = 1 if "sprak=1" in url else 2
        offset = int(url.split("offset=")[1].split("&")[0])
        items = [] if offset else [{"nummer": k, "namn": v[1] if lang == 1 else v[0], "livsmedelsTyp": "Analysed",
                                    "version": "2026-06-29"} for k, v in FOODS.items()]
        return {"_meta": {"totalRecords": len(FOODS)}, "livsmedel": items}
    nr = int(url.rsplit("/livsmedel/", 1)[1].split("/")[0])
    if "/naringsvarden" in url:
        return FOODS[nr][4]
    if "/klassificeringar" in url:
        return [{"typ": "FoodEx2", "kod": FOODS[nr][2]}, {"typ": "Main group", "kod": FOODS[nr][3]},
                {"typ": "LanguaL", "fasett": "F. EXTENT OF HEAT TREATMENT", "langualId": "F0003",
                 "fasettkod": "F0011", "namn": "NOT HEAT-TREATED "}]
    if "/ravaror" in url:
        return [{"foodEx2": "A02LV", "namn": "Milk, cow", "andel": 100, "faktor": 1, "omraknadTillRa": 100,
                 "tillagning": "Uncooked"}] if nr == 2 else []
    raise AssertionError(url)


def test_corrupt_cache_is_refetched(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("fetch", Path(__file__).parents[1] / "scripts" / "fetch_livsmedelsverket.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    (tmp_path / "7.json").write_text('{"nutrients": [')  # truncated by an interrupted run
    monkeypatch.setattr(m, "get_json", lambda url, *a, **k: [{"url": url}])
    d = m.fetch_food_details(7, tmp_path)
    assert set(d) == {"nutrients", "classifications", "raw_commodities"}
    assert not list(tmp_path.glob("*.tmp"))


def test_download_writes_tidy_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(lv, "get_json", fake_get_json)
    lv.main(["--out", str(tmp_path), "--workers", "2"])
    foods = list(csv.DictReader(open(tmp_path / "foods.csv", encoding="utf-8")))
    assert [f["name"] for f in foods] == ["Beef tallow", "Milk 3% fat"]
    assert foods[1]["name_sv"] == "Mjölk 3%" and foods[1]["foodex2_code"] == "A02MC"
    wide = list(csv.DictReader(open(tmp_path / "nutrients_wide.csv", encoding="utf-8")))
    assert "ENERC_kJ" in wide[0] and "ENERC_kcal" in wide[0]          # energy units kept apart
    assert wide[1]["DISAC_g"] == "4.8" and wide[0]["DISAC_g"] == ""   # missing stays empty
    langual = list(csv.DictReader(open(tmp_path / "langual.csv", encoding="utf-8")))
    assert langual[0]["term"] == "NOT HEAT-TREATED"
    rav = list(csv.DictReader(open(tmp_path / "raw_commodities.csv", encoding="utf-8")))
    assert len(rav) == 1 and rav[0]["foodex2_code"] == "A02LV" and rav[0]["raw_equivalent_g"] == "100"
    info = json.loads((tmp_path / "download_info.json").read_text())
    assert info["foods"] == 2 and info["licence"] == "CC BY 4.0"
    # second run uses the cache and does not refetch details
    calls = []
    monkeypatch.setattr(lv, "get_json", lambda u, **k: calls.append(u) or fake_get_json(u))
    lv.main(["--out", str(tmp_path)])
    assert not any("naringsvarden" in u for u in calls)
