# Nordic food reference (food space map)

A reference of Swedish and Finnish foods for analysing dietary records. Every food has its
FAO/WHO GIFT food group, its composition and its "fraction" profile (what kind of fat,
sugar, carbohydrate and protein it carries). The reference is used to:

1. group and map foods within each GIFT group, and check the official grouping;
2. place foods that are not in the tables (unique products in food records);
3. compare people's diets from food records, using food similarity rather than exact food matches.

The composition side runs on the real Swedish and Finnish tables. Everything involving
people runs on **simulated** food records, apart from a first mapping test on the public BIG IDEAs food logs (docs/bigideas_mapping.md).

## Status (2026-11-06)

| Part | State | Where |
|---|---|---|
| Swedish table (Livsmedelsverket) | Done: 2,606 foods, 60 components, 2,603 with a GIFT group | `data/processed/foods_lv.csv` |
| Finnish table (Fineli) | Done: 4,238 foods, all with a GIFT group (food-use class map, plus name rules for 349) | `data/processed/foods_fi.csv` |
| Combined Nordic table | Done: 6,841 foods; SE-FI crosswalk of 744 mutual best matches | `data/processed/foods_nordic.csv`, `docs/se_fi_crosswalk_mutual.csv` |
| Protein fractions | Partly provisional: 13 of 30 food categories have values, and 9 of those 13 use provisional or textbook numbers | `data/curated/protein_fractions.csv` |
| Clustering and trees within GIFT groups | Results committed for SE, FI and Nordic | `results/within_gift*/` |
| Clustering all foods (main pipeline) | Runs; agrees weakly with GIFT (ARI 0.21) | `python -m foodspace.run` |
| Placing new foods | Works: the name alone finds the GIFT group for 82-89% of foods | `src/foodspace/placement.py` |
| Food records | **Simulated only**: 100 people x 3 days | `data/simulated/food_records/` |
| Diet comparison between people | Runs on the simulated records; weak but real signal; GIFT subgroup tree is the best tested food tree | `scripts/compare_diets.py`, `results/diet_similarity/` |
| Real free-text records (BIG IDEAs, US) | First test: 78% of logged energy maps automatically; about 83% right food and 92% right GIFT group among accepted items (hand review) | `scripts/map_bigideas.py`, `docs/bigideas_mapping.md`, `results/bigideas/` |
| Splitting composite dishes into ingredients | Not done: needs the recipe download | see [Open work](#open-work) |
| Explorer website | Live at https://pho3nix96.github.io/dietary-db/ (first deployed 2026-11-06) | `.github/workflows/pages.yml` |
| Licence | Not chosen | |

The explorer pages (`artifacts/`) are published at https://pho3nix96.github.io/dietary-db/ by the pages
workflow, on every push to `main` that changes `artifacts/`, `site/` or the workflow itself. They show the
composition data and the simulated records only.

## How it fits together

```
raw downloads (not in git)            curated tables (in git)
  Livsmedelsverket API                  GIFT map for FoodEx2 and Fineli classes
  Fineli open-data package              protein fractions with sources
  GIFT FoodEx2 matching table           protein category per Swedish food
        |                                      |
        +------------- prepare_* ------------- +
                          |
             data/processed/ food tables (in git)
               |                    |                       |
   within-group structure    placement of new foods    food records (simulated)
   (clusters, trees, maps)   (names + nutrients)       -> matching -> diet profiles
                                                        -> person-to-person distance
```

## Data

| Source | Licence | In git |
|---|---|---|
| Swedish Food Agency, Livsmedelsdatabasen (API, via `scripts/fetch_livsmedelsverket.py`) | CC BY 4.0, "Source: Swedish Food Agency" | Processed table only |
| Fineli Release 20, basic package 2 (THL; manual download from fineli.fi, "Open data") | CC BY 4.0 | Processed table only |
| FAO/WHO GIFT FoodEx2 matching table (FoodEx2_MTX.xlsx, matrix 16.2) | **Not checked** | Derived `data/curated/foodex2_to_gift.csv` only |

The raw downloads, including the Swedish recipes (`raw_commodities.csv`), stay in `data/raw/`
and are not committed; the scripts re-create them. Check the FAO licence before releasing the
derived GIFT table.

The GIFT nutrition-sensitive grouping has 19 groups and 95 subgroups. The Swedish table uses
18 of the groups (none of its foods is in "Insects, grubs and their products"); 3 Swedish foods
have no group ("unassigned"). The largest group is Composite dishes (470 Swedish foods).

## Methods in brief

**Views of a food.** Similarity between two foods combines:
- *composition*: protein, fat, available carbohydrate and fibre per 100 g (log1p, z-scored, cosine distance);
- *fractions*: each macronutrient split into its parts and turned into shares, so milk and milk
  powder get the same profile. The blocks are fat classes (SFA/MUFA/PUFA), omega-6 / ALA / long-chain
  omega-3, C4-C10 vs longer saturated fat (Swedish table only), mono- / sucrose / other disaccharides,
  starch / sugars / fibre, and protein fractions. Shares are compared with the Aitchison distance
  (zero replacement, centred log-ratio, Euclidean). Each block is scaled to [0, 1] and blocks
  are averaged with equal weight. A block counts for a pair of foods only when both have it, and
  foods with less than `min_parent` g/100 g of the parent nutrient have no profile for that block.

Composition and fractions are fused with equal weight. GIFT is the reference the results are
checked against, not an input (`fusion.gift_in_fusion: false`). Food names never shape the
clusters; they are used only to place new foods. Every choice is in `config/config.yaml`.

**Protein fractions** are not in composition tables. `data/curated/protein_fractions.csv` gives
shares per food category from the literature (casein/whey for dairy, Osborne fractions for
cereals and pulses, 11S/7S for soy, myofibrillar/sarcoplasmic/stromal for meat and fish), each
with a status and a DOI. To compare across families, fractions are summed into solubility-based
`common_class` values; that mapping is approximate. A food's profile is the protein-weighted mix
of its recipe ingredients' profiles (`src/foodspace/commodities.py` for Sweden, recipe expansion in
`scripts/prepare_fineli.py` for Finland). This gives profiles for 1,026 Swedish and 1,339 Finnish
foods. The main pipeline instead uses one curated category per food
(`data/curated/protein_category_map.csv`, 524 Swedish foods).

**Clustering and trees.** Leiden communities on a k-nearest-neighbour graph of the fused
similarity; treelet trees (Lee, Nadler & Wasserman 2008) on 1 - fused distance; UMAP layouts.
Treelet merges are not monotonic: a later merge can be more similar than an earlier one, so cut
the trees by merge order (`treelet.cut`), not by height.

**Comparing diets** (`src/foodspace/diet.py`):
1. *Matching.* Each record line is linked to a reference food. The steps, in order: by food id,
   only when the records use the reference's own ids; by FoodEx2 code (base term, facets used
   when they match exactly); by name among the foods sharing that base term; or by name alone.
   Unique products keep the GIFT group placement gave them. Unmatched lines are reported, not guessed.
2. *Nutrients.* Lines without nutrient values take them from their reference food.
3. *Profiles.* Each person becomes intake shares (energy by default) over the tree
   GIFT group -> subgroup -> food, so constant under-reporting does not change the profile.
4. *Distance.* The earth mover's distance on that tree: swapping whole milk for semi-skimmed
   milk costs little, swapping milk for cereals costs the most. A macro-nutrient (Aitchison) view is
   added with weight 0.25 by default. Composite dishes count as one food of their group unless a
   table of ingredient shares is passed (`--dish-shares`).
5. *Food tree* (`--tree`). The default is GIFT group -> subgroup -> food. `subgroup-composition`
   keeps the subgroups and, inside each, uses the composition tree of the reference
   (`results/within_gift/trees.json`), so foods of one subgroup are as close as their composition
   allows. `composition` uses the composition tree instead of the subgroups. Branch lengths come
   from the treelet merge distances, made monotonic and scaled to the same depths, so the largest
   distance and the cost of crossing GIFT groups are the same in every tree.

## Results so far

All foods, Swedish table (`python -m foodspace.run`, outputs not committed):
- Leiden finds 16 clusters. Agreement with the GIFT groups is low: ARI 0.21, NMI 0.40. 1,300 of
  the 2,606 foods sit in a cluster dominated by another group, so the GIFT groups are not
  compact in composition space. Read that list as disagreements to inspect, not as coding errors.
- Fractions keep food families together (`docs/comparison_by_group.csv`). The share of a food's
  5 nearest neighbours that are in its Livsmedelsverket food group rises from 0.32 on totals
  alone to 0.44 with fractions; for dairy (neighbours that are dairy) from 0.43 to 0.72, and for
  hard cheese from 0.31 to 0.84. Sauces (0.48 -> 0.31) and soups (0.37 -> 0.34) lose cohesion.
- 524 Swedish foods have a curated protein category, but only 355 of those categories have values
  yet. Of the 355 foods, 109 rest on sourced values alone and 246 on provisional or textbook values
  (`metrics.json`, `protein_values_by_status`).

Within GIFT groups (`results/within_gift*/groups.csv`; medians over the groups with at least 8 foods):

| Table | ARI vs subgroup | 5 nearest in same subgroup: fused | composition | fractions | names (not used) |
|---|---|---|---|---|---|
| Sweden | 0.12 | 0.71 | 0.62 | 0.59 | 0.77 |
| Finland | 0.16 | 0.76 | 0.71 | 0.67 | 0.82 |
| Nordic | 0.11 | 0.74 | 0.68 | 0.65 | 0.84 |

In the Nordic table, a food's nearest neighbours come from the other country at 0.64 of the rate
expected if the country did not matter (median over groups). The two databases still differ in
method (measured vs estimated starch, summed vs total omega-3), and part of that shows.

Cereals by protein fractions: 250 cereal foods form 6 clusters (`docs/cereal_cluster_summary.csv`).
Wheat, barley and maize foods are prolamin-rich, rice is glutelin-rich, rye is albumin-rich.
The profile is computed from the grain mix, so agreement with the dominant grain is partly built in.

Placing new foods (leave-one-out over 600 foods): the name alone recovers the GIFT group for
82% (Sweden), 89% (Finland) and 88% (Nordic) of foods, and the subgroup for 70-78%. The
TF-IDF index includes the held-out name, so these figures are slightly optimistic.

Simulated food records (`data/simulated/food_records/`, see its README): 100 people, 3 days,
4,280 record lines, with 225 unique products (60 of them name only). Placement recovers the
true GIFT group of 95% (name only) and 96% (name + values) of the unique products.

Diet comparison on the simulated records (`results/diet_similarity/summary.json`):
- all 4,280 lines are matched, and 60 lines get their nutrients from the reference;
- a person's 5 most similar people share their hidden dietary pattern 50% of the time, against
  28% by chance; the food-tree view alone gives 51%, the nutrient view alone 34%;
- Leiden clusters of people agree weakly with the 4 patterns (ARI 0.13);
- ignoring the food ids, as with real records, gives the same results; the matcher then finds
  the right food for 99.4% of lines. That is an upper bound, because the simulated names are
  copied from the reference.

Which food tree works best (`scripts/compare_diet_trees.py`, `results/diet_similarity/tree_comparison.csv`):
the committed records plus 5 more record sets simulated with other seeds, 100 people each. The
value is the share of a person's 5 most similar people with the same hidden pattern, mean over the
6 sets (chance: 0.28):

| Food tree | Tree alone | Tree + nutrients | Tree alone vs default, range over sets |
|---|---|---|---|
| Flat (no similarity between different foods) | 0.43 | 0.44 | -0.12 to -0.04 |
| GIFT groups only | 0.47 | 0.47 | -0.06 to -0.01 |
| **GIFT subgroups (default)** | **0.52** | **0.52** | |
| Subgroups + composition inside them | 0.52 | 0.51 | -0.01 to +0.02 |
| Composition instead of subgroups | 0.50 | 0.49 | -0.04 to -0.01 |

Food similarity clearly helps: every tree beats treating each food on its own. Replacing the
subgroups with the composition trees is worse in every record set. Inside Composite dishes, for
example, the subgroups separate meat, fish and vegetable dishes, which is what tells a vegetarian
from a traditional eater, while composition mixes dishes with similar macros. Using composition
inside the subgroups ties with the default. The simulated patterns are built from which kinds of
food people pick, so they cannot show whether composition matters in real diets; that needs real
records and an outside criterion (for example biomarkers).

The simulation also gives each person strong habitual favourites, which may be part of why
patterns separate weakly. Re-check on real records.

## Running

```bash
conda env create -f environment.yml        # or: pip install -e ".[dev,scripts]"
conda activate food-space-map
pip install -e .
pytest -q                                  # 65 tests
```

Without any downloads (from a fresh clone), these run on the committed tables:

```bash
python -m foodspace.run config/config.yaml                     # all foods (writes results/food_map.csv, misfits.csv, metrics.json)
python scripts/within_gift_groups.py --out results/within_gift # within GIFT groups (the website's map data also needs the GIFT matrix)
python scripts/place_food.py "Skyr, plain" --db nordic --set PROT_g=11 FAT_g=0.2 CHO_g=3.8 FIBT_g=0 SUGAR_g=3.8
python scripts/compare_diets.py                                # diet comparison on the simulated records
python scripts/compare_diet_trees.py                           # which food tree works best (simulates 5 more record sets)
python scripts/person_graphs.py && python scripts/build_cointake_graph.py && python scripts/recommendation_groups.py
```

These need the raw downloads:

| Step | Needs |
|---|---|
| `fetch_livsmedelsverket.py`, then `prepare_livsmedelsverket.py` | Livsmedelsverket API (about 5,000 small requests, resumable) |
| `within_group_cereals.py`, `food_tree.py` | `data/raw/livsmedelsverket/raw_commodities.csv` |
| `prepare_fineli.py` | Fineli package unzipped into `data/raw/fineli/unz/` |
| `map_gift.py`, `build_gift_map_data.py` | `data/raw/gift/foodex2_gift_nut.csv` |

`snakemake --cores 1` runs the whole chain, but only once the raw files are in place. In a fresh
clone it stops at the first step that needs them. The records rules (`recommendation_groups`,
`person_graphs`, `cointake_graph`, `diet_similarity`, `diet_tree_comparison`) need no raw data.

Install `umap-learn` for UMAP layouts like the committed ones. Without it the layouts fall back to
t-SNE, with a warning.

## Using real dietary records

Records are expected in the FAO/WHO GIFT microdata layout: `SUBJECT`, `ROUND` (optional),
`SURVEY_DAY`, `MEAL_NAME`, `FOODEX2_INGR_CODE`, `INGREDIENT_ENG`, `FOOD_AMOUNT_CONS` (g) and
nutrient columns (`ENERGY_kcal`, `PROTEIN_g`, `FAT_g`, `CARBOH_g`, `SUGAR_g`, `FIBTG_g`, ...).
A recorded day is a `SURVEY_DAY` within a `ROUND`; an eating occasion is a `MEAL_NAME` on that day.

- **Keep them out of git.** Individual records (with age, sex and weight) are personal data,
  and probably health data under GDPR. Put them in `data/private/` and write results to
  `results/private/`; both are ignored by git. Do not rebuild the public explorer pages from them.
- **Match on code and name.** Their food ids are not ids of the reference, so run
  `python scripts/compare_diets.py --records data/private/<set> --out results/private/<set> --ignore-ids`.
  Check `matching_report.json` (match types, unmatched lines, where nutrients came from).
- **Unique products.** `place_unique_foods.py` needs an `in_database` column marking lines
  whose food is not in the table; the unmatched lines of the matcher are the candidates.
- **Ingredient-level rows.** GIFT's column names (`FOODEX2_INGR_CODE`, `INGREDIENT_ENG`) suggest
  many datasets already split recipes into ingredients. Check this first: if so, composite dishes
  arrive pre-split, and the recipe-based split matters only for dishes recorded as one line.

## Open work

- **Composite dishes.** Split them into ingredient groups using the Swedish recipes
  (`raw_commodities.csv`) and map each ingredient's FoodEx2 code to a GIFT group. This needs the
  recipe download. `diet_profiles` already accepts the resulting shares (`--dish-shares`).
- **Real records**, e.g. INRAN-SCAI 2005-06 via FAO/WHO GIFT, under your own access agreement.
- **Protein fractions.** Replace provisional values with primary sources: meat, rennet cheese,
  rice, rye, oats, maize, soybean, faba bean, whey protein. Then source the 17 categories with no
  values yet: yogurt (plain and strained), quark and cottage cheese, cream, kefir and fermented milk,
  cheese whey, egg, processed meat, offal, potato, pea, lentil, chickpea, kidney bean, peanut, tree
  nuts, oil seeds. To use only sourced categories, set `statuses: [sourced, by_definition]` under
  `protein_type` in the config.
- **Committed results predate some fixes.** Re-running changes them:
  - Within-group neighbour scores: a food could count as its own neighbour when another food had
    distance 0. Regenerated Swedish scores fall slightly (fraction view -0.012 on average, at most
    -0.06 in one group). Clusters and ARI do not change.
  - The servings page (`results/recommendation_groups/groups.json`) was built with curated protein
    categories. The script now uses the same recipe-based profiles as the within-group trees, which
    changes 6 of its 9 trees (servings are unchanged).
  - The commodity protein rules (whole-word matching, so buttermilk and salted or fatty fish are no
    longer dropped) affect only tables prepared again from the raw downloads.
- **Website.** The CDN scripts in the pages have no integrity hashes, and the deploy workflow's
  actions target the deprecated Node 20.
- **Robustness.** Bootstrap stability of branches and clusters; compare treelet trees with average
  linkage; a sweep of the GIFT weight.
- **Licences** for code and data, and a check of the FAO matching table's terms.
- Dependencies have only minimal version bounds. Most scripts are covered only by smoke runs;
  unit tests cover the library and the helpers the scripts share.

`TODO.md` has the full task list; `PROJECT_INDEX.md` is a dated snapshot.

## Repository layout

```
config/config.yaml        every analysis choice
data/raw/                 downloads (not in git)
data/curated/             hand-made tables: GIFT maps, protein fractions, protein category per food
data/processed/           Swedish, Finnish and Nordic food tables (in git)
data/simulated/           simulated food records (in git; not real people)
data/private/             real records (ignored by git)
results/                  committed snapshots: within-group results, servings, person graphs,
                          co-intake graph, diet similarity (main-pipeline outputs are not committed)
results/private/          results from real records (ignored by git)
src/foodspace/            library: views, fractions, fuse, evaluate, layout, treelet, commodities,
                          placement, diet, records, io, run
scripts/                  data: fetch_livsmedelsverket, prepare_livsmedelsverket, prepare_fineli,
                            combine_databases, map_gift
                          food structure: within_gift_groups, build_gift_map_data, within_group_cereals,
                            food_tree, recommendation_groups, compare_totals_vs_fractions, build_map_data
                          placement: place_food, place_unique_foods
                          records: simulate_food_records, person_graphs, build_cointake_graph, compare_diets,
                            compare_diet_trees
tests/                    pytest on synthetic data
artifacts/                explorer pages (three are published by the pages workflow; the task
                          checklist is a claude.ai page)
site/index.html           landing page of the website
docs/                     figures and tables used in write-ups
```

## Licence

Not chosen yet. Suggested: MIT for the code, CC BY 4.0 for released data, with attribution to
the Swedish Food Agency and THL (Fineli).
