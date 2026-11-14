"""Map free-text food-record items (another country's records) onto the food reference.

Each item is a name plus the macros its app reported for the eaten portion. Matching:

1. Clean the name: drop the brand in brackets "(Kellogg's) Frosted Flakes", lower-case,
   expand record-app shorthand, and translate US food words to the words the Nordic
   tables use (US_TO_REFERENCE: "chips" -> "crisps", "cookie" -> "biscuit", ...).
2. Name candidates: the k reference foods with the most similar names (the TF-IDF
   word + character index used to place new foods, placement.NameIndex).
3. Composition check: the reported macros give the item's energy profile (shares of
   energy from protein, fat, sugars, other available carbohydrate and fibre). Each
   candidate's profile comes from its per-100 g values. The score is
       name similarity - comp_weight * profile distance
   so among foods with similar names the one whose composition fits the report wins
   ("milk" with 5 g fat per 8 fl oz picks semi-skimmed, not whole milk). Portion size
   does not enter: shares do not depend on how much was eaten.
4. A match is accepted when the name similarity of the chosen food reaches min_name_sim.

US labels count fibre inside total carbohydrate; the Nordic tables give available
carbohydrate (CHO_g) and fibre separately, so available = total_carb - fibre here.
The reference table has no alcohol column, so the energy of alcoholic drinks is
understated on the reference side.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .placement import NameIndex

# US record wording -> wording of the Nordic (SE/FI English) food names. Applied to whole words.
US_TO_REFERENCE = {
    r"chips": "crisps", r"potato chips?": "potato crisps", r"tortilla chips?": "tortilla crisps nachos", r"corn chips?": "corn crisps",
    r"french fries|fries": "french fries chips potato", r"cookies?": "biscuit cookie",
    r"candy|candies": "sweets confectionery", r"gumdrops?": "sweets jelly", r"jelly": "jam",
    r"soda|pop": "soft drink", r"diet coke|coke zero|diet soda|diet pepsi": "soft drink cola sugar free",
    r"coke|pepsi|cola": "soft drink cola", r"diet mountain dew": "soft drink citrus sugar free",
    r"mountain dew|mello yello|sprite|7 ?up": "soft drink citrus lemonade",
    r"stevia|equal|splenda|sweet ?n ?low": "sweetener", r"cheetos": "cheese puffs maize snack",
    r"frosted flakes|corn flakes|cornflakes": "corn flakes breakfast cereal", r"cheerios": "oat rings breakfast cereal",
    r"oatmeal": "oatmeal porridge oat", r"granola": "muesli granola", r"cereal": "breakfast cereal",
    r"2% milk|2 ?% partly skimmed milk|2 percent": "milk 1.5% semi-skimmed", r"skim milk|nonfat milk": "milk skimmed",
    r"whole milk": "milk 3% whole", r"half and half|half & half": "cream 10%", r"coffee cream(er)?|creamer": "coffee cream 12%",
    r"heavy cream|whipping cream": "whipping cream 40%", r"sour cream": "sour cream",
    r"cottage cheese": "cottage cheese", r"cream cheese": "cream cheese", r"string cheese|cheese stick": "cheese mozzarella",
    r"greek yogurt": "greek yoghurt", r"yogurt": "yoghurt", r"ground beef": "beef mince",
    r"hamburger": "hamburger", r"hot dog": "hot dog sausage", r"bologna": "sausage cold cuts",
    r"deli turkey|turkey breast": "turkey cold cuts", r"jerky": "dried meat", r"bacon": "bacon",
    r"tuna": "tuna", r"zucchini": "courgette", r"eggplant": "aubergine",
    r"cilantro": "coriander", r"arugula": "rocket", r"scallions?|green onions?": "spring onion",
    r"bell peppers?": "sweet pepper", r"garbanzo": "chickpea", r"romaine": "lettuce romaine",
    r"cantaloupe": "melon cantaloupe", r"blue ?berries": "blueberry", r"raisins": "raisin",
    r"peanut butter|pb": "peanut butter", r"jam": "jam", r"syrup": "syrup", r"ketchup": "ketchup",
    r"mayo": "mayonnaise", r"ranch dressing|ranch": "dressing", r"tzatziki|tatziki": "tzatziki yoghurt sauce",
    r"salsa": "salsa tomato sauce", r"biscuit": "scone biscuit", r"muffins?": "muffin", r"bagels?": "bagel bread",
    r"tortilla": "tortilla bread", r"english muffins?": "english muffin bread", r"pancakes?": "pancake",
    r"waffles?": "waffle", r"protein bar": "protein bar", r"protein (powder|shake)|whey( protein)?": "protein powder whey",
    r"smoothie": "smoothie", r"latte": "caffe latte milk coffee", r"cappuccino": "cappuccino", r"espresso": "espresso coffee",
    r"moscato|chardonnay|pinot|cabernet|merlot": "wine", r"ice cream": "ice cream", r"popsicle": "ice lolly",
    r"pretzels?": "pretzel", r"crackers?": "crackers", r"pop ?corn": "popcorn",
    r"donuts?|doughnuts?": "doughnut", r"grits": "maize porridge polenta", r"loaf": "cake",
    r"pound cake": "sponge cake", r"pot pie": "chicken pie", r"bourbon|whiskey|whisky": "spirit whisky",
    r"vodka|gin|rum|tequila": "spirit vodka", r"martini": "dry martini", r"beer": "beer",
    r"gatorade|powerade": "sports drink", r"red bull|monster": "energy drink", r"crystal light": "soft drink lemonade sugar free",
    r"jello|jell o": "jelly dessert", r"brownies?": "brownie", r"cupcakes?": "muffin cake",
    r"m ?& ?m ?s|m and m s|mms": "chocolate sweets milk chocolate", r"reese s|reeses": "chocolate peanut butter",
    r"snickers": "chocolate bar caramel peanut", r"oreos?": "biscuit chocolate cream filling",
    r"skittles|starburst|smarties|dots": "sweets jelly candy", r"boost": "meal replacement drink",
    r"premier protein|ensure": "meal replacement protein drink", r"quest": "protein bar",
    r"cheez its?|goldfish": "crackers cheese", r"lean cuisine|banquet|stouffer s": "ready meal",
    r"jimmy dean": "sausage", r"sausage biscuit": "sausage bread", r"burritos?": "burrito wrap",
    r"tacos?": "taco", r"nuggets": "nuggets", r"wings": "wings", r"coleslaw": "cabbage salad coleslaw",
    r"hard boiled( eggs?)?": "egg boiled", r"scrambled eggs?": "egg scrambled", r"fried eggs?": "egg fried",
    r"sweet tea": "tea sweetened sugar", r"iced tea": "tea ice tea",
    r"tangerines?": "mandarin clementine", r"slaw": "coleslaw", r"peppers": "sweet pepper", r"ziti|penne|rigatoni": "pasta",
    r"rice": "rice boiled", r"coffee": "coffee brewed", r"tea": "tea brewed", r"fat free milk|nonfat milk": "milk skimmed",
    r"mac and cheese|macaroni and cheese": "macaroni cheese gratin", r"cinnamon rolls?": "cinnamon bun",
    r"sweet rolls?": "sweet bun", r"red velvet cake": "cake", r"cobb salad|chef salad": "salad with chicken egg cheese",
}
# claims that say how a food is sold, not what it is (dropped before matching)
CLAIM_WORDS = re.compile(r"\b(lactose[ -]?free|gluten[ -]?free|organic|natural|homemade|home made|generic|premium|"
                         r"original|classic|single serving|bag|fresh|real sugar)\b")
# head nouns that are not foods on their own: they say how much, not what
NOT_HEADS = {"cup", "cups", "serving", "servings", "piece", "pieces", "slice", "slices", "bowl", "plate",
             "small", "medium", "large", "box", "meal", "bar", "bars", "original", "plain", "flavor", "flavored"}
NON_FOOD = re.compile(r"\b(ibuprofen|tylenol|advil|aspirin|vitamin|multivitamin|supplement|pills?|tablets?|medication|"
                      r"metformin|melatonin|probiotic)\b", re.I)
SHORTHAND = {r"choc": "chocolate", r"std": "standard", r"b ?fast": "breakfast", r"w/": "with ", r"&": " and ",
             r"lg": "large", r"sm": "small", r"med": "medium", r"hard boiled": "egg boiled", r"oz": "ounce"}

BRAND = re.compile(r"^\s*\([^)]*\)\s*")


_COMPILED: dict[int, tuple] = {}


def _sub_all(s: str, table: dict) -> str:
    """Replace whole-word matches in one pass (longer patterns first), so a replacement
    is never translated again."""
    if id(table) not in _COMPILED:
        pats = sorted(table, key=len, reverse=True)
        rx = re.compile("|".join(f"(?<![a-z])(?:{p})(?![a-z])" for p in pats))
        comp = [(re.compile(f"^(?:{p})$"), table[p]) for p in pats]
        _COMPILED[id(table)] = (rx, comp)
    rx, comp = _COMPILED[id(table)]

    def rep(m):
        for c, r in comp:
            if c.match(m.group(0)):
                return r
        return m.group(0)
    return rx.sub(rep, s)


def _plain(name: str, translate: bool) -> str:
    s = BRAND.sub("", name).lower()
    s = re.sub(r"\([^)]*\)", " ", s)  # other bracketed notes
    s = s.replace("'", "")
    s = CLAIM_WORDS.sub(" ", s)
    s = _sub_all(s, SHORTHAND)
    if translate:
        s = _sub_all(s, US_TO_REFERENCE)
    s = re.sub(r"[^a-z0-9% ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_label(name, translate: bool = True) -> str:
    """'(Kellogg's) Frosted Flakes, Cereal' -> 'corn flakes breakfast cereal breakfast cereal'."""
    if not isinstance(name, str) or not name.strip():
        return ""
    return _plain(name, translate)


def head_label(name, translate: bool = True) -> str:
    """The head noun phrase: in English the last word of the part before 'with'/','
    ('Chicken Biscuit' -> biscuit, 'Salad with Fried Chicken' -> salad), translated."""
    if not isinstance(name, str) or not name.strip():
        return ""
    s = BRAND.sub("", name).lower().replace("'", "")
    s = re.split(r",| with | w/ | and | & |\(| - ", s)[0]
    words = [w for w in re.sub(r"[^a-z ]+", " ", s).split() if w not in NOT_HEADS]
    if not words:
        return ""
    two = " ".join(words[-2:])
    # keep two words when they form one food name ("peanut butter", "frosted flakes")
    whole = any(re.fullmatch(p, two) for p in US_TO_REFERENCE)
    head = two if (len(words) > 1 and (two in PAIRS or whole)) else words[-1]
    return _plain(head, translate) if len(head) > 2 else ""


PAIRS = {"peanut butter", "ice cream", "sour cream", "cream cheese", "cottage cheese", "green beans", "sweet potato",
         "french toast", "hot dog", "orange juice", "apple juice", "olive oil", "coffee cream", "chocolate milk",
         "potato salad", "egg salad", "tuna salad", "chicken salad", "fruit salad", "hot chocolate", "baked beans"}


# words that make a different food out of the same ingredient: a candidate that has one the
# label lacks ("Apple" -> "Apple pie") is a dish or product of the food, not the food itself
FORM_WORDS = {"pie", "soup", "cake", "cakes", "jam", "sauce", "salad", "stew", "casserole", "pudding", "gratin", "pizza",
              "juice", "drink", "dried", "powder", "chips", "chip", "compote", "dressing", "spread", "sandwich",
              "pasty", "dough", "balls", "sticks", "stock", "mousse", "smoothie", "risotto", "lasagna", "curry"}
DEFAULT_FORM = re.compile(r"\b(raw|fresh|average|plain|unspecified|w/ skin|with skin|boiled|brewed)\b", re.I)


def form_adjustment(label: str, names: pd.Series, penalty: float = 0.1, bonus: float = 0.03) -> np.ndarray:
    """-penalty for each form word in a candidate name that the label lacks; +bonus for
    a default form (raw, plain, average...) when the label is one or two words."""
    lab = set(re.findall(r"[a-z]+", label.lower()))
    out = np.zeros(len(names))
    for n, name in enumerate(names):
        words = set(re.findall(r"[a-z]+", str(name).lower()))
        out[n] -= penalty * len((words & FORM_WORDS) - lab)
        if len(lab) <= 2 and DEFAULT_FORM.search(str(name)):
            out[n] += bonus
    return out


def is_non_food(name) -> bool:
    return isinstance(name, str) and bool(NON_FOOD.search(name))


# -- composition profile -------------------------------------------------------------------
PROFILE = ["protein", "fat", "sugar", "other_carb", "fibre"]
KCAL = np.array([4.0, 9.0, 4.0, 4.0, 2.0])


def profile(protein, fat, avail_carb, sugar, fibre) -> np.ndarray:
    """Energy shares from protein, fat, sugars, other available carbohydrate and fibre (n x 5).
    Missing parts count as 0; rows with under 5 kcal from these parts are NaN (no profile)."""
    P = np.nan_to_num(np.asarray(protein, float))
    F = np.nan_to_num(np.asarray(fat, float))
    S = np.nan_to_num(np.asarray(sugar, float))
    C = np.clip(np.nan_to_num(np.asarray(avail_carb, float)) - S, 0, None)
    B = np.nan_to_num(np.asarray(fibre, float))
    E = np.column_stack([P, F, S, C, B]) * KCAL
    tot = E.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = E / tot
    out[tot.ravel() < 5] = np.nan
    return out


def reference_profiles(foods: pd.DataFrame) -> np.ndarray:
    """Per-100 g profiles of the reference foods (5 kcal floor applies per 100 g)."""
    return profile(foods["PROT_g"], foods["FAT_g"], foods["CHO_g"], foods["SUGAR_g"], foods["FIBT_g"])


def reference_energy(foods: pd.DataFrame) -> np.ndarray:
    """kcal per 100 g from the macros (no alcohol column in the table)."""
    return (4 * foods["PROT_g"].fillna(0) + 9 * foods["FAT_g"].fillna(0) + 4 * foods["CHO_g"].fillna(0)
            + 2 * foods["FIBT_g"].fillna(0)).to_numpy(float)


def record_profiles(items: pd.DataFrame) -> np.ndarray:
    fib = items["dietary_fiber"]
    avail = items["total_carb"] - fib.fillna(0)
    return profile(items["protein"], items["total_fat"], avail, items["sugar"], fib)


def profile_distance(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distance between energy-share profiles, scaled to 0..1 (max is sqrt(2))."""
    return np.sqrt(((B - a) ** 2).sum(axis=1)) / np.sqrt(2)


# -- matching ------------------------------------------------------------------------------
def match_items(items: pd.DataFrame, foods: pd.DataFrame, label_col: str = "label", k: int = 30,
                comp_weight: float = 0.5, head_weight: float = 0.4, min_name_sim: float = 0.35,
                max_profile_dist: float = 0.3, form_penalty: float = 0.1, translate: bool = True) -> pd.DataFrame:
    """One row per item: best reference food by name only and by name + composition.

    items:  label_col (free text) and the record macros (protein, total_fat, total_carb,
            sugar, dietary_fiber) for the eaten portion; any of them may be missing.
    foods:  reference table (food_id, name, gift_group, gift_subgroup, PROT_g, FAT_g, CHO_g, SUGAR_g, FIBT_g).
    Name similarity = (1 - head_weight) * similarity of the whole label
                      + head_weight * similarity of its head noun (head_label)
                      + form_adjustment (dish/product words the label lacks are penalised).
    Returns item index, cleaned label, and for "name" (name similarity only) and "hybrid"
    (name + composition): food_id, name, GIFT group/subgroup, name similarity, profile
    distance and `status`:
        matched               name similarity >= min_name_sim and, when the item has a
                              profile, profile distance <= max_profile_dist
        composition_mismatch  a similar name exists but its composition does not fit the report
        no_similar_name       best name similarity < min_name_sim
        non_food              medicine or supplement (NON_FOOD)
        no_label              empty name
    """
    foods = foods.reset_index(drop=True)
    index = NameIndex(foods["name"].fillna(""))
    ref_prof = reference_profiles(foods)
    rec_prof = record_profiles(items)
    rows = []
    for (i, r), rp in zip(items.iterrows(), rec_prof):
        raw = r[label_col]
        lab, head = clean_label(raw, translate), head_label(raw, translate)
        out = {"item": i, "clean_label": lab, "head": head}
        if not lab or is_non_food(raw):
            out["name_status"] = out["hybrid_status"] = "no_label" if not lab else "non_food"
            rows.append(out)
            continue
        sims = index.similarity(lab)
        if head and head_weight > 0 and head != lab:
            sims = (1 - head_weight) * sims + head_weight * index.similarity(head)
        cand = np.argsort(-sims)[:k]
        if form_penalty:
            sims = sims.copy()
            sims[cand] += form_adjustment(lab, foods["name"].iloc[cand], penalty=form_penalty)
            cand = cand[np.argsort(-sims[cand])]
        d = profile_distance(rp, ref_prof[cand]) if not np.isnan(rp).any() else np.full(len(cand), np.nan)
        score = sims[cand] - comp_weight * np.nan_to_num(d, nan=0.5)
        for how, j in (("name", cand[0]), ("hybrid", cand[int(np.argmax(score))])):
            pos = int(np.where(cand == j)[0][0])
            if sims[j] < min_name_sim:
                status = "no_similar_name"
            elif not np.isnan(d[pos]) and d[pos] > max_profile_dist:
                status = "composition_mismatch"
            else:
                status = "matched"
            out.update({f"{how}_food_id": int(foods.at[j, "food_id"]), f"{how}_name": foods.at[j, "name"],
                        f"{how}_gift_group": foods.at[j, "gift_group"], f"{how}_gift_subgroup": foods.at[j, "gift_subgroup"],
                        f"{how}_name_sim": float(sims[j]), f"{how}_profile_dist": float(d[pos]),
                        f"{how}_status": status})
        rows.append(out)
    return pd.DataFrame(rows).set_index("item")
