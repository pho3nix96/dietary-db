# Food space map

**Explorer pages:** https://pho3nix96.github.io/dietary-db/ (built from `artifacts/` by `.github/workflows/pages.yml`). See `PROJECT_INDEX.md` for a project overview.

Cluster and visualize a dietary database by three views of each food:

| View | What it captures | Source |
|---|---|---|
| Classification | What the food officially *is* | FAO/WHO GIFT nutrition-sensitive grouping (19 groups, 95 subgroups), via FoodEx2 |
| Nutrients | What it *contains* | Food composition table |
| Semantics | How it is *described* | Embeddings of food names and descriptions |

The views are fused into one food graph, clustered with Leiden, compared with the
GIFT grouping, and laid out in 2D for an interactive map. Foods where the views
disagree ("misfits") are the main result: they show where official groupings and
nutrient content part ways, and they often expose coding or matching errors.

## Status

Early stage. The pipeline runs end to end on synthetic data (`tests/`) and has been
applied to the Swedish and Finnish food tables (`data/processed/`, `results/`). Results
from simulated food records (`data/simulated/`, `results/person_graphs/`, `results/cointake/`)
are illustrations, not findings. See `TODO.md` for open work.

## Comparing people's diets

The reference can be used to compare dietary records (`src/foodspace/diet.py`,
`scripts/compare_diets.py`):

1. each record line is linked to a reference food (food id, then FoodEx2 code, then name; unique
   products keep their placed GIFT group); unmatched lines are reported, not guessed;
2. each person becomes a profile of intake shares (energy by default) over the food tree
   GIFT group -> subgroup -> food, so under-reporting by a constant factor does not change it;
3. the distance between two people is the earth mover's distance on that tree: swapping whole
   milk for semi-skimmed milk is small, swapping milk for cereals is large. A secondary
   macro-nutrient (Aitchison) view can be added with a weight.
4. composite dishes count as one food of their group until a table of ingredient shares is
   supplied with `--dish-shares` (food_id, leaf, share).

```bash
python scripts/compare_diets.py --records data/simulated/food_records --out results/diet_similarity
```

On the simulated records the people are only weakly separable by their hidden dietary pattern
(5 nearest neighbours share the pattern 49-51% of the time, against about 28% by chance; Leiden
ARI about 0.14). The food-tree view carried the signal and the nutrient view diluted it, so the
nutrient view defaults to a weight of 0.25. This is one simulated dataset: re-check on real records.

## Input format

One CSV (or TSV) row per food. Column names are set in `config/config.yaml`.

| Column | Required | Notes |
|---|---|---|
| `food_id` | yes | Unique |
| `name` | yes | Any language if using a multilingual model |
| `description` | no | Improves the semantic view |
| `foodex2_code` | no | Recommended; links all four GIFT groupings |
| `gift_group`, `gift_subgroup` | yes | Nutrition-sensitive grouping |
| nutrient columns | yes | Same basis for all foods (e.g. per 100 g) |

## Data source

Current source: the **Swedish Food Agency food composition database
(Livsmedelsdatabasen)**, about 2,600 foods with 59 components each, FoodEx2 and
LanguaL codes, licensed CC BY 4.0 ("Source: Swedish Food Agency"). Fineli is
the intended Finnish alternative but is currently not reachable from outside
Finland.

Useful components for the fraction layer: saturated, mono- and polyunsaturated
fatty acid totals and 15 individual fatty acids; total sugars, monosaccharides,
disaccharides and sucrose. Two fractions are derived in the prepare step:
starch (available carbohydrate minus sugars) and other disaccharides
(disaccharides minus sucrose, mainly lactose and maltose). Not available:
measured starch, lactose, and amino acids.

## Running

```bash
conda env create -f environment.yml
conda activate food-space-map
pip install -e .

# 1. download (about 5,000 small requests; resumable; standard library only)
python scripts/fetch_livsmedelsverket.py
# 2. build the pipeline's food table (adds derived fractions, GIFT groups when mapped)
python scripts/prepare_livsmedelsverket.py
# 3. run the analysis
python -m foodspace.run config/config.yaml
# 4. cluster cereal foods among themselves by protein fractions
python scripts/within_group_cereals.py
# or steps 2-4 with
snakemake --cores 1
```

Outputs in `results/`:

- `food_map.csv` — each food with its cluster and 2D coordinates
- `misfits.csv` — foods whose cluster is dominated by a different GIFT group
- `resolution_scan.csv` — agreement with GIFT and seed stability across resolutions
- `metrics.json` — ARI/NMI vs GIFT group and subgroup, plus every nutrient-preparation choice made

## Design notes

- **Every analytical choice is in the config** (nutrient set, basis, missing-value rule,
  log transform, distance, fusion method, k, resolution) and is echoed into `metrics.json`.
- **GIFT is the reference, not an input, by default** (`fusion.gift_in_fusion: false`).
  The GIFT view is discrete: foods in the same subgroup have distance 0, so even a small
  GIFT weight makes each food's nearest neighbours its subgroup mates, splits the graph
  along subgroups, and pulls misfit foods back into their official group. It also makes
  agreement with GIFT partly circular. Turn it on only for a weight sweep.
- **Fusion**: `weighted` (adjustable baseline) or `snf` (similarity network fusion,
  Wang et al. 2014).
- **Semantic view** needs `sentence-transformers`; the default model is multilingual.

## Nutrient fractions (the "what kind" layer)

Totals say how much protein, fat or carbohydrate a food has; fractions say what
kind. Each block describes one macronutrient by its parts (fatty acid classes,
individual sugars, starch vs sugars vs fibre, casein vs whey), turned into shares
of the block so that concentration does not matter (milk and milk powder get the
same profile). Shares are compared with the Aitchison distance (zero replacement,
centred log-ratio, Euclidean), each block is scaled to [0, 1], and blocks are
averaged with equal weight. A block counts for a pair only when both foods have
it; foods with less than `min_parent` g/100 g of the parent nutrient have no
profile for that block.

Blocks are set in `config/config.yaml` under `fractions`. Column names there are
placeholders until the Fineli files are loaded; blocks with missing columns are
skipped with a message.

**Protein fractions** are not in composition tables, so they come from
`data/curated/protein_fractions.csv`, organised by GIFT food group. One row per
category, scheme and fraction, with the share, its status (`sourced`,
`provisional`, `by_definition`, `to_source`), the source DOI and a note. Only
rows with a share are used. The food table links to it through a
`protein_fraction_category` column.

Each food family has its own fraction scheme: casein/whey for dairy, Osborne
fractions for cereals and pulses, 11S/7S storage globulins for soybean,
myofibrillar/sarcoplasmic/stromal for meat and fish. To compare across families,
every fraction also has a `common_class` based on solubility (water_soluble,
salt_soluble, alcohol_soluble, insoluble_storage, casein, structural_insoluble,
or mixed where a source pools fractions). `scheme: common` in the config sums
shares by class. The mapping is approximate (e.g. myofibrillar proteins are
salt-soluble like globulins but are not storage proteins) and should be stated
as such in a paper.

Low-protein foods (below `min_parent`, 2 g/100 g by default) get no protein
profile, so most fruit, vegetables and fats are unaffected. The `gift_subgroup`
column is empty until the official FAO grouping file (FoodEx2_MTX.xlsx) is
loaded; group names should be checked against it too.

On the synthetic test foods (`tests/test_fractions.py`), totals put Gouda next to
salami (distance 0.13) and far from milk (0.97); with fractions Gouda moves next
to milk (0.14) and away from salami (1.00), and whey protein is set apart from
cheese by protein type.

## Within a group: cereals by protein fractions

Across all foods, fractions mostly help separate food groups. Within a group
the question changes: which breads, flakes and pastas are alike in the kind
of protein they carry (gliadin-type prolamins vs glutenins vs albumins and
globulins)?

The database gives each food's recipe in raw commodities
(`data/raw/livsmedelsverket/raw_commodities.csv`, from the `/ravaror`
endpoint). `src/foodspace/commodities.py` turns that into a protein profile:

1. protein of each raw commodity, from the single-commodity foods in the same
   database (rolled oats, rye flour, ...), with a few fallbacks;
2. grain protein in a product = raw-commodity grams x commodity protein;
3. Osborne profile = protein-weighted mix of each grain's curated shares
   (`data/curated/protein_fractions.csv`, scheme `osborne`).

`scripts/within_group_cereals.py` keeps foods where grain is at least half of
the recipe and supplies 50-200 % of the protein (above 100 % happens when grain
was refined to starch), then compares totals, measured fractions, and
measured + protein fractions, scored against the dominant grain and the food
group. First run (250 foods, 5 nearest neighbours):

| view | same dominant grain | same food group |
|---|---|---|
| totals | 0.60 | 0.39 |
| + fat/sugar/carb fractions | 0.63 | 0.48 |
| + protein fractions | 0.83 | 0.54 |

The protein profile is computed from the grain mix, so the jump in "same
dominant grain" is partly built in. What it adds is a chemical scale: wheat,
barley and maize sit together (prolamin-rich), rice is glutelin-rich, rye is
albumin-rich, oat-wheat blends fall in between. Cereal values are partly
provisional (see the `status` column); rye in particular uses textbook Osborne
values that likely underestimate secalins.

## Within GIFT groups (default view)

The 19 FAO/WHO GIFT nutrition-sensitive groups are the fixed top level: foods
from different groups are never clustered together. `scripts/map_gift.py`
builds `data/curated/foodex2_to_gift.csv` from the official FoodEx2 matching
table (FoodEx2_MTX.xlsx, matrix 16.2; nutrition-sensitive columns saved in
`data/raw/gift/foodex2_gift_nut.csv`), and 2,603 of 2,606 Swedish foods get a
group from their FoodEx2 base term. `scripts/within_gift_groups.py` then works
inside each group on macros and fraction blocks (names only place new foods, see below): (fat classes,
omega-6/ALA/long-chain omega-3, C4-C10 milk-fat marker, sugars, starch/sugar/
fibre, protein fractions), with Leiden clusters, a treelet tree per group and
agreement with GIFT subgroups. Results: `results/within_gift/groups.csv`.
The raw matrix extract stays out of git; the derived `foodex2_to_gift.csv` is kept.
Check the FAO licence of the matching file before redistributing it.

## Placing new foods (the only use of names)

Food names do not shape the space: clusters, trees and layouts use
composition and fractions only. Names are used to place a new food:

```bash
python scripts/place_food.py "Skyr, plain" --db nordic --set PROT_g=11 FAT_g=0.2 CHO_g=3.8 FIBT_g=0 SUGAR_g=3.8
python scripts/place_food.py --evaluate --db se     # how often names alone find the GIFT group
```

`src/foodspace/placement.py`: the name (TF-IDF, word and character n-grams)
picks the GIFT group and subgroup by a weighted vote of the 10 most similar
names (pass `--group` to override; a warning is given below 0.6 confidence);
detail the new food lacks (fat and sugar parts, protein-fraction profile) is
filled from its name neighbours and listed as imputed; its position and
cluster come from comparing its nutrients with the group's foods, as the
foods are compared with each other. With no nutrients, the name neighbours
place it. Leave-one-out, names find the GIFT group for 82% (Sweden), 90%
(Finland) and 86% (combined) of foods, and the subgroup for 71-78%.

## Fineli (Finland)

`scripts/prepare_fineli.py` reads the Fineli open-data package (release 20,
basic package 2: 4,232 foods, 74 components; THL, CC BY 4.0; download from
fineli.fi "Open data" and unzip into `data/raw/fineli/unz/`) into the same table
layout as the Swedish data, so the same config and scripts run:

```bash
python scripts/prepare_fineli.py
python scripts/within_gift_groups.py --input data/processed/foods_fi.csv --out results/within_gift_fi
```

Differences from the Swedish table: starch, lactose and individual sugars are
measured; omega-3 and omega-6 totals are given; there are no C4-C10 fatty
acids (milk-fat block skipped) and no FoodEx2 codes. GIFT groups come from
`data/curated/fineli_fuclass_to_gift.csv` (122 food-use classes, curated) plus
name rules. Protein profiles are recipe-based: every dish is expanded through
`contribfood.csv` to raw ingredients, each ingredient class points to a
curated protein category, and the profile is the protein-weighted mix.

## Combined Nordic table

`scripts/combine_databases.py` pools both tables (6,841 foods; `source` = SE/FI)
on the columns that mean the same in both, and prints matched food types from
each database side by side. Protein profiles are recipe-based in both (Swedish
raw-commodity recipes via `commodities.recipe_protein_profiles`; Fineli
recipes). Harmonisation choices: the C4-C10 block is dropped (Fineli lacks it);
the omega-6/ALA/long-chain omega-3 block has no "other PUFA" remainder (the
databases tally minor PUFA differently); Swedish estimated starch under 0.5 g
is treated as 0 (rounding noise).

`within_gift_groups.py --input data/processed/foods_nordic.csv` then reports
country mixing per view (observed share of other-country nearest neighbours /
expected share) and writes `crosswalk.csv`: each food's closest food from the
other country within its GIFT group, with `mutual` marking pairs that are each
other's best match (887 pairs; 81% fall in the same cluster).
`docs/se_fi_crosswalk_mutual.csv` lists them.

## Food tree (treelet transform)

`scripts/food_tree.py` builds a binary food tree with the treelet transform
(Lee, Nadler & Wasserman 2008; `src/foodspace/treelet.py`), the method used for
dietary patterns by Gorst-Rasmussen et al. 2011 and reviewed by Zhao et al.
2021 (Nutr J 20:37, Fig. 4).

In dietary-pattern work the variables are food groups and the observations
are people, so the tree joins foods that are eaten together. Here the tree is
built from composition instead: foods are the variables and their
standardized profiles (log macros plus centred log-ratios of the fat, sugar,
carbohydrate and protein-fraction blocks, equal weight per block) are the
observations. At each level the two most correlated foods or branches are
rotated into a sum variable (the branch prototype, written to `nodes.csv` with
its loading-weighted Osborne profile) and a difference variable.

On the 250 cereal foods, cutting at 8 branches gives a pure rice branch
(glutelin ~0.68), a maize branch, soft wheat breads, wheat buns and cakes,
oat-based foods, and a rye + wholegrain flour branch. The same cut on intake
data would give dietary patterns; the two trees could later be compared.

## Layout

```
config/          analysis choices
data/raw/        input tables (not committed)
data/processed/  intermediate files (not committed)
results/         outputs (not committed)
src/foodspace/   pipeline code
tests/           synthetic-data tests
notebooks/       exploration
TODO.md          project plan
```

## Licence

Not chosen yet. Suggested: MIT for code, CC-BY 4.0 for released data.
