#!/usr/bin/env python3
"""Simulate 3-day food records in the FAO/WHO GIFT microdata layout.

    python scripts/simulate_food_records.py [--subjects 100] [--days 3] [--seed 42]
        [--out data/simulated/food_records]

Purpose: test data for the co-intake graph and dietary-pattern code until real
records (e.g. INRAN-SCAI 2005-06 via FAO/WHO GIFT) are available. Nothing here is
real intake.

Design
- Foods come from the Swedish Food Agency table, so every row has a real
  FoodEx2 code, GIFT group/subgroup and nutrient values per 100 g.
- Each subject has a latent dietary pattern (traditional Nordic, health-conscious,
  convenience, vegetarian); the pattern shifts which foods are picked in each
  meal slot (keyword weights) and how often snacks, sweets and alcohol occur.
- Days are consecutive, starting on a random weekday; weekends shift breakfast
  later, add sweets and alcohol and reduce lunch at work.
- Meals are built from slots (e.g. breakfast = bread + spread + topping + hot
  drink, or porridge + milk + berries); each slot draws a food from its pool,
  and the subject reuses favourite foods across days (habitual choices).
- Portions: slot base weight x energy scale (sex, age, body size) x lognormal
  day-to-day noise; amounts rounded to whole grams.
- Each subject has a person-level reporting factor (mean 0.9, so ~10% under-
  reporting on average), which scales portions.

Outputs (GIFT column names where they exist)
- subjects.csv     SUBJECT, SEX, AGE_YEAR, WEIGHT, HEIGHT, AREA_TYPE
- consumption.csv  SUBJECT, SURVEY_DAY, WEEK_DAY, CONSUMPTION_TIME_HOUR/_MINUTES,
                   MEAL_NAME, PLACE, FOODEX2_INGR_CODE, INGREDIENT_ENG, food_id (Swedish),
                   gift_group, gift_subgroup, FOOD_AMOUNT_CONS, ENERGY_kcal, PROTEIN_g,
                   CARBOH_g, SUGAR_g, FIBTG_g, FAT_g, SAT_FAT_g, MUFA_g, PUFA_g, ALCOHOL_g
- truth.csv        SUBJECT, pattern, reporting_factor (hidden, for checking methods)
- unique_foods_truth.csv
                   unique products (not in the food table, each reported once): their
                   label values per 100 g (or none), and the hidden base food and true GIFT
                   group, for checking how well placement recovers them
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

POOLS = {  # slot: (Swedish food groups, optional name regex, base grams)
    "bread": (["Soft bread", "Crisp bread"], r"", 40),
    "spread": (["Fat of mixed origin >=80%fat", "Mixed origin fat", "Butter"], r"[Ss]pread|[Bb]utter|Bregott|[Mm]argarine", 8),
    "topping": (["Hard cheese", "Processed meat", "Sausages"], r"[Cc]heese|[Hh]am|[Ss]alami|[Pp]âté|[Ss]moked", 20),
    "veg_topping": (["Vegetables"], r"^Cucumber|^Tomato|^Sweet pepper|^Lettuce", 25),
    "porridge": (["Porridge"], r"[Pp]orridge", 250),
    "cereal": (["Breakfast cereals"], r"", 40),
    "ferm": (["Natural fermented products", "Flavoured fermented milk"], r"", 200),
    "berry": (["Berries fresh frozen", "Jam, marmelade, jelly, apple sauce"], r"", 60),
    "fruit": (["Fruit fresh frozen"], r"Apple|Banana|Orange|Pear|Grape|Kiwi|Mandarin|Clementine|Plum", 130),
    "hotdrink": (["Coffee infusion", "Tea infusion"], r"brewed|infusion|[Cc]offee|[Tt]ea", 200),
    "milk": (["Milk"], r"^Milk \d|^Milk 0|^Milk 1|^Milk 3", 200),
    "juice": (["Fruit juices etc"], r"juice", 200),
    "meat": (["Meat products meat dishes", "Poultry products and dishes", "Sausage dishes"], r"", 150),
    "fish": (["Fish and shellfish dishes", "Fish fresh frozen cooked"], r"fried|boiled|baked|[Gg]ratin|[Ss]oufflé|[Pp]oached|oven|[Ss]moked", 140),
    "veggie": (["Replacement for meat protein products and dishes", "Vegetable dishes and products"], r"", 160),
    "soup": (["Soup"], r"", 300),
    "fastfood": (["Pasta dishes", "Pizza pie pirog", "Hamburger with bread"], r"", 300),
    "starch": (["Potatoes", "Rice rice noodles", "Pasta"], r"boiled|Boiled|[Mm]ashed", 180),
    "veg": (["Vegetables", "Root vegetables", "Mixed salad"], r"", 80),
    "sauce": (["Sauce dressing mayonnaise"], r"[Ss]auce|[Gg]ravy|[Dd]ressing", 40),
    "bun": (["Buns cakes cookies"], r"", 60),
    "sweet": (["Sweets with chocolate", "Chocolate", "Sweets excluding chocolate", "Ice cream"], r"", 50),
    "crisps": (["Crisps, popcorn, etc.", "Nuts and seeds"], r"", 35),
    "soft": (["Fruit syrup soft drink, non-alcoholic cider"], r"", 330),
    "alcohol": (["Beer", "Wine"], r"", 300),
    "water": (["Water, mineral water"], r"Tap water|Mineral water", 250),
}

# keyword weights per pattern (multiplicative on the food's draw weight)
PATTERN_WORDS = {
    "traditional": {r"[Pp]otato": 3, r"[Mm]eatball|[Ss]ausage|[Pp]ork|[Bb]eef|[Hh]erring": 3, r"[Rr]ye|[Cc]risp bread": 2,
                    r"[Oo]at|[Pp]orridge": 2, r"[Bb]utter|Bregott": 2, r"[Ff]ermented milk|filmjölk": 2, r"[Cc]heese hard": 2},
    "health": {r"[Ww]holegrain|[Ff]ibre": 3, r"[Ss]almon|[Cc]od|[Ff]ish": 3, r"[Yy]oghurt|[Ff]ermented milk 0": 2,
               r"[Mm]uesli|[Oo]at": 3, r"[Ss]alad|[Bb]roccoli|[Ss]pinach|[Cc]arrot": 3, r"[Bb]erries|[Bb]lueberr|[Rr]aspberr": 2,
               r"0\.5%|1% fat|[Ll]ight": 2, r"[Tt]ea": 2, r"[Ss]ugar|[Cc]hocolate|[Ff]ried": 0.4},
    "convenience": {r"[Pp]izza|[Hh]amburger|[Pp]asta|[Ll]asagne|[Kk]ebab|[Nn]ugget": 4, r"[Ss]ugar|[Cc]ocoa|[Hh]oney|[Ff]lavour": 2,
                    r"[Cc]hips|[Cc]risps|[Cc]hocolate|[Cc]andy|[Ss]weets": 2, r"[Cc]ola|[Ss]oft drink|[Ss]oda": 3,
                    r"[Ff]ried|[Dd]eep": 2, r"[Ww]hite|[Ww]heat bread": 2, r"[Ww]holegrain": 0.5},
    "vegetarian": {r"[Bb]ean|[Ll]entil|[Cc]hickpea|[Ss]oy|[Tt]ofu|[Qq]uorn|[Vv]eget|[Ff]alafel|[Hh]alloumi": 4,
                   r"[Oo]at drink|[Ss]oy drink": 3, r"[Nn]ut|[Ss]eed": 2},
}
EXCLUDE = {"vegetarian": r"[Mm]eat|[Bb]eef|[Pp]ork|[Cc]hicken|[Hh]am|[Ss]ausage|[Ss]alami|[Ll]iver|[Ff]ish|[Ss]almon|[Cc]od|[Hh]erring|[Ss]hrimp|[Tt]una|[Bb]acon|[Kk]ebab|[Hh]amburger|[Pp]âté|[Ll]amb|[Tt]urkey"}

PATTERN_SHARE = {"traditional": 0.35, "health": 0.25, "convenience": 0.25, "vegetarian": 0.15}
MEALS = ["Breakfast", "Snack or drink between breakfast and lunch", "Lunch",
         "Snack or drink between lunch and dinner", "Dinner", "Snack or drink after dinner"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# foods that are not eaten as such: powders, concentrates, dry/uncooked forms, infant and diet products
NOT_AS_EATEN = (r"[Pp]owder|[Cc]onc\.|[Cc]oncentrat|[Uu]ncooked|[Dd]ried uncooked|[Ii]nfant|[Cc]hildren|[Bb]aby|"
                r"[Ff]ermented w/|[Ii]nstant coffee powder|[Dd]iet |[Ss]weetener|[Rr]aw\b")


def build_pools(foods: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    foods = foods[~foods["name"].str.contains(NOT_AS_EATEN, regex=True, na=False)]
    for slot, (groups, rx, _) in POOLS.items():
        p = foods[foods["main_group"].isin(groups)]
        if rx:
            p = p[p["name"].str.contains(rx, regex=True, na=False)]
        p = p[pd.to_numeric(p["ENERC_kcal"], errors="coerce").notna()]
        out[slot] = p.reset_index(drop=True)
    return out


def pattern_weights(pool: pd.DataFrame, pattern: str) -> np.ndarray:
    w = np.ones(len(pool))
    names = pool["name"].fillna("")
    for rx, f in PATTERN_WORDS.get(pattern, {}).items():
        w *= np.where(names.str.contains(rx, regex=True), f, 1.0)
    if pattern in EXCLUDE:
        w *= np.where(names.str.contains(EXCLUDE[pattern], regex=True), 0.0, 1.0)
    return w / w.sum() if w.sum() > 0 else np.ones(len(pool)) / len(pool)


class Person:
    def __init__(self, sid, rng, pools):
        self.sid, self.rng, self.pools = sid, rng, pools
        self.sex = rng.choice(["Male", "Female"])
        self.age = int(rng.integers(18, 75))
        self.height = round(float(rng.normal(179 if self.sex == "Male" else 166, 7)), 1)
        bmi = float(np.clip(rng.lognormal(np.log(25.5), 0.15), 17, 45))
        self.weight = round(bmi * (self.height / 100) ** 2, 1)
        self.pattern = rng.choice(list(PATTERN_SHARE), p=list(PATTERN_SHARE.values()))
        self.report = float(np.clip(rng.normal(0.9, 0.12), 0.55, 1.2))
        # energy scale ~ basal needs relative to an average adult
        bmr = (10 * self.weight + 6.25 * self.height - 5 * self.age + (5 if self.sex == "Male" else -161))
        self.scale = 1.2 * bmr / 1550.0  # calibrated so median reported energy is ~1,800 (F) / ~2,200 (M) kcal
        self.favs = {}  # habitual choices per slot
        self.skip_breakfast = rng.random() < (0.25 if self.pattern == "convenience" else 0.08)
        self.snacker = rng.random()
        self.bf_type = rng.choice(["sandwich", "porridge", "yoghurt", "cereal"],
                                  p={"traditional": [.5, .35, .1, .05], "health": [.2, .3, .4, .1],
                                     "convenience": [.35, .05, .15, .45], "vegetarian": [.3, .3, .3, .1]}[self.pattern])

    def pick(self, slot):
        pool = self.pools[slot]
        if len(pool) == 0:
            return None
        if slot not in self.favs:
            w = pattern_weights(pool, self.pattern)
            k = min(3, len(pool))
            self.favs[slot] = (self.rng.choice(len(pool), size=k, replace=False, p=w), w)
        favs, w = self.favs[slot]
        i = self.rng.choice(favs) if self.rng.random() < 0.7 else self.rng.choice(len(pool), p=w)
        return pool.iloc[int(i)]

    def grams(self, slot):
        base = POOLS[slot][2]
        drink = slot in ("hotdrink", "milk", "juice", "soft", "alcohol", "water")
        s = 1.0 if drink else self.scale
        return max(5, round(base * s * self.report * self.rng.lognormal(0, 0.3)))


def meal_slots(p: Person, meal: str, weekend: bool, rng) -> list[str]:
    if meal == "Breakfast":
        if p.skip_breakfast and rng.random() < 0.8:
            return []
        t = p.bf_type if rng.random() < 0.75 else rng.choice(["sandwich", "porridge", "yoghurt", "cereal"])
        s = {"sandwich": ["bread", "spread", "topping"] + (["veg_topping"] if rng.random() < .4 else []),
             "porridge": ["porridge", "milk"] + (["berry"] if rng.random() < .6 else []),
             "yoghurt": ["ferm", "cereal"] + (["berry"] if rng.random() < .5 else []),
             "cereal": ["cereal", "milk"]}[t]
        s += ["hotdrink"] + (["juice"] if rng.random() < .25 else [])
        if weekend and rng.random() < 0.4:
            s += ["bread", "spread", "topping"]
        return s
    if meal == "Lunch":
        if rng.random() < 0.05:
            return []
        if rng.random() < (0.35 if p.pattern == "convenience" else 0.12):
            return ["fastfood"] + (["soft"] if rng.random() < .5 else ["water"])
        if rng.random() < 0.15:
            return ["soup", "bread", "spread"]
        main = rng.choice(["meat", "fish", "veggie"], p={"traditional": [.6, .25, .15], "health": [.3, .4, .3],
                                                         "convenience": [.6, .15, .25], "vegetarian": [0, 0, 1]}[p.pattern])
        s = [main, "veg"] + (["starch"] if main != "veggie" or rng.random() < .5 else [])
        s += (["sauce"] if rng.random() < .5 else []) + (["bread", "spread"] if rng.random() < .5 else [])
        s += [rng.choice(["milk", "water"], p=[.4, .6] if p.pattern != "convenience" else [.2, .8])]
        return s
    if meal == "Dinner":
        if rng.random() < (0.3 if p.pattern == "convenience" or weekend else 0.1):
            s = ["fastfood"]
        else:
            main = rng.choice(["meat", "fish", "veggie", "soup"], p={"traditional": [.55, .25, .1, .1], "health": [.25, .4, .25, .1],
                                                                   "convenience": [.6, .1, .2, .1], "vegetarian": [0, 0, .8, .2]}[p.pattern])
            s = [main] + (["starch", "veg"] if main != "soup" else ["bread", "spread"])
            s += ["sauce"] if main in ("meat", "fish") and rng.random() < .6 else []
        drink = "alcohol" if weekend and rng.random() < .45 else rng.choice(["water", "milk", "soft"], p=[.55, .3, .15] if p.pattern != "convenience" else [.4, .1, .5])
        return s + [drink]
    # snacks
    base = 0.55 if meal.startswith("Snack or drink between lunch") else 0.35 if "breakfast and lunch" in meal else 0.4
    if rng.random() > base * (0.6 + 0.8 * p.snacker):
        return []
    if "after dinner" in meal:
        opts = [["sweet"], ["crisps"], ["bread", "spread", "topping", "hotdrink"], ["fruit"], ["ferm"]]
        pw = [.25, .15, .3, .15, .15] if not weekend else [.35, .25, .2, .1, .1]
    else:
        opts = [["hotdrink", "bun"], ["fruit"], ["hotdrink"], ["ferm"], ["bread", "spread", "topping"], ["sweet"]]
        pw = {"health": [.1, .4, .2, .2, .08, .02], "convenience": [.3, .1, .15, .05, .1, .3]}.get(p.pattern, [.35, .2, .2, .1, .1, .05])
    return list(opts[rng.choice(len(opts), p=pw)])


MEAL_TIME = {"Breakfast": (7.3, 0.7), "Snack or drink between breakfast and lunch": (10.0, 0.5), "Lunch": (12.0, 0.5),
             "Snack or drink between lunch and dinner": (15.0, 0.6), "Dinner": (18.0, 0.8), "Snack or drink after dinner": (21.0, 0.7)}


MODIFIERS = ["homemade", "organic", "store-brand", "grandma's", "farm-shop", "artisan", "high-protein",
             "lactose-free", "reduced-sugar", "extra crispy", "family recipe", "market stall", "small-batch"]
EXTRAS = {"bread": ["with seeds", "with caraway", "with sunflower seeds", "sourdough"],
          "topping": ["with herbs", "smoked", "with pepper crust"], "ferm": ["raspberry", "vanilla-cardamom", "blueberry"],
          "porridge": ["with apple and cinnamon", "with lingonberries"], "meat": ["with chanterelles", "with lingonberry sauce"],
          "fish": ["with dill", "with horseradish"], "veggie": ["with chickpeas", "with feta"], "soup": ["with dumplings"],
          "bun": ["with cardamom", "with saffron", "with almond paste"], "sweet": ["with sea salt", "with licorice"],
          "fastfood": ["with jalapeños", "with extra cheese"], "berry": ["sea-buckthorn mix"], "fruit": ["dried mix"]}


def unique_name(base: str, slot: str, rng) -> str:
    """A product name not in the food table: the base food reworded with a modifier and/or extra."""
    words = re.sub(r"\s*\b(e\.g\.|app\.|w/o\b|fortified\b|RTD\b).*$", "", base).strip(" ,")
    words = words if len(words) > 3 else base
    parts = [rng.choice(MODIFIERS)] if rng.random() < 0.8 else []
    parts.append(words.lower())
    if slot in EXTRAS and rng.random() < 0.6:
        parts.append(rng.choice(EXTRAS[slot]))
    return " ".join(parts).capitalize()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--foods", default="data/processed/foods_lv.csv")
    ap.add_argument("--subjects", type=int, default=100)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unique-share", type=float, default=0.06,
                    help="share of food rows that are unique products (not in the food table, reported once)")
    ap.add_argument("--no-values-share", type=float, default=0.3,
                    help="share of unique products recorded with a name only (no nutrient values)")
    ap.add_argument("--out", default="data/simulated/food_records")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    foods = pd.read_csv(args.foods, dtype={"foodex2_base": str})
    foods["main_group"] = foods["main_group"].astype(str).str.strip()
    pools = build_pools(foods)
    nut = {"ENERGY_kcal": "ENERC_kcal", "PROTEIN_g": "PROT_g", "CARBOH_g": "CHO_g", "SUGAR_g": "SUGAR_g",
           "FIBTG_g": "FIBT_g", "FAT_g": "FAT_g", "SAT_FAT_g": "FASAT_g", "MUFA_g": "FAMS_g", "PUFA_g": "FAPU_g",
           "ALCOHOL_g": "ALC_g"}

    subj, rows, truth, unique = [], [], [], []
    for sid in range(1, args.subjects + 1):
        p = Person(sid, rng, pools)
        subj.append({"SUBJECT": sid, "SEX": p.sex, "AGE_YEAR": p.age, "WEIGHT": p.weight, "HEIGHT": p.height,
                     "AREA_TYPE": rng.choice(["Urban", "Rural"], p=[.7, .3])})
        truth.append({"SUBJECT": sid, "pattern": p.pattern, "reporting_factor": round(p.report, 3)})
        start = int(rng.integers(0, 7))
        works = rng.random() < 0.65
        for d in range(args.days):
            wd = (start + d) % 7
            weekend = wd >= 5
            for meal in MEALS:
                slots = meal_slots(p, meal, weekend, rng)
                if not slots:
                    continue
                mu, sd = MEAL_TIME[meal]
                t = rng.normal(mu + (1.2 if weekend and meal == "Breakfast" else 0), sd)
                hh, mm = int(t) % 24, int((t % 1) * 60)
                place = "Work or school" if (meal in ("Lunch", "Snack or drink between breakfast and lunch") and works and not weekend
                                             and rng.random() < .8) else ("Restaurant" if rng.random() < .07 else "Home")
                for slot in slots:
                    f = p.pick(slot)
                    if f is None:
                        continue
                    g = p.grams(slot)
                    uniq = rng.random() < args.unique_share and slot not in ("water", "hotdrink")
                    r = {"SUBJECT": sid, "ROUND": 1, "SURVEY_DAY": d + 1, "WEEK_DAY": WEEKDAYS[wd],
                         "CONSUMPTION_TIME_HOUR": hh, "CONSUMPTION_TIME_MINUTES": mm, "MEAL_NAME": meal,
                         "PLACE": place, "slot": slot, "FOODEX2_INGR_CODE": f.get("foodex2_base"),
                         "INGREDIENT_ENG": f["name"], "food_id": int(f["food_id"]), "gift_group": f.get("gift_group"),
                         "gift_subgroup": f.get("gift_subgroup"), "FOOD_AMOUNT_CONS": g}
                    for k, c in nut.items():
                        v = pd.to_numeric(f.get(c), errors="coerce")
                        r[k] = round(float(v) * g / 100, 3) if pd.notna(v) else np.nan
                    r["in_database"] = True
                    if uniq:
                        uid = f"U{len(unique) + 1:04d}"
                        name = unique_name(f["name"], slot, rng)
                        has_vals = rng.random() >= args.no_values_share
                        # label-style values: the base food perturbed by product differences (+/- ~15%)
                        jit = {k: float(rng.lognormal(0, 0.15)) for k in nut}
                        unique.append({"unique_food_id": uid, "name": name, "base_food_id": int(f["food_id"]),
                                       "base_name": f["name"], "true_gift_group": f.get("gift_group"),
                                       "true_gift_subgroup": f.get("gift_subgroup"), "has_values": has_vals,
                                       **{f"{c}_per100": (round(float(pd.to_numeric(f.get(c), errors='coerce')) * jit[k], 2)
                                                          if has_vals and pd.notna(pd.to_numeric(f.get(c), errors='coerce')) else np.nan)
                                          for k, c in nut.items()}})
                        r.update({"FOODEX2_INGR_CODE": None, "INGREDIENT_ENG": name, "food_id": None,
                                  "gift_group": None, "gift_subgroup": None, "in_database": False, "unique_food_id": uid})
                        for k, c in nut.items():
                            v = unique[-1][f"{c}_per100"]
                            r[k] = round(v * g / 100, 3) if pd.notna(v) else np.nan
                    rows.append(r)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cons = pd.DataFrame(rows)
    pd.DataFrame(subj).to_csv(out / "subjects.csv", index=False)
    cons.to_csv(out / "consumption.csv", index=False)
    pd.DataFrame(truth).to_csv(out / "truth.csv", index=False)
    pd.DataFrame(unique).to_csv(out / "unique_foods_truth.csv", index=False)
    print(f"unique products: {len(unique)} ({sum(u['has_values'] for u in unique)} with nutrient values, "
          f"{sum(not u['has_values'] for u in unique)} name only)")

    # summary
    day = cons.groupby(["SUBJECT", "SURVEY_DAY"])[list(nut)].sum().reset_index()
    per = day.groupby("SUBJECT")[list(nut)].mean().join(pd.DataFrame(subj).set_index("SUBJECT")[["SEX"]])
    per = per.join(pd.DataFrame(truth).set_index("SUBJECT"))
    e = per["ENERGY_kcal"]
    pct = lambda c, k: (per[c] * k / e * 100).median()
    print(f"{len(cons)} food rows, {cons['food_id'].nunique()} distinct foods, "
          f"{cons.groupby(['SUBJECT','SURVEY_DAY','MEAL_NAME']).ngroups} eating occasions")
    print("energy kcal/day, median by sex:", per.groupby("SEX")["ENERGY_kcal"].median().round(0).to_dict())
    print(f"energy % (median): protein {pct('PROTEIN_g', 4):.0f}, carbohydrate {pct('CARBOH_g', 4):.0f}, "
          f"fat {pct('FAT_g', 9):.0f}, alcohol {pct('ALCOHOL_g', 7):.1f}; fibre g/day {per['FIBTG_g'].median():.0f}")
    print("eating occasions per day:", round(cons.groupby(["SUBJECT", "SURVEY_DAY"])["MEAL_NAME"].nunique().mean(), 1))
    print("patterns:", pd.DataFrame(truth)["pattern"].value_counts().to_dict())
    print(per.groupby("pattern")[["ENERGY_kcal", "FIBTG_g", "SUGAR_g"]].median().round(0).to_string())
    print(f"Wrote {out}/")


if __name__ == "__main__":
    main()
