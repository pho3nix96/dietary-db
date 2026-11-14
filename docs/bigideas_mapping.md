# Mapping BIG IDEAs food logs onto the Nordic reference

First test of the reference on real food records: the free-text food logs of the BIG IDEAs Lab
glycemic variability dataset (16 US adults with high-normal HbA1c, 8 to 10 days, Dexcom G6;
PhysioNet 1.1.2, ODC-By 1.0, Cho et al. 2023, https://doi.org/10.13026/zthx-s306).
Run with `python scripts/map_bigideas.py --data <download folder>`; numbers below are from
`results/bigideas/summary.json` (2026-11-14). The raw dataset is not in the repository.

## What was mapped

1,422 logged lines, 617 distinct item names. 77% of lines carry the logging app's own matched
description ("searched_food"), which is used when present; otherwise the typed name. Each line
has energy and macros for the eaten portion (US label values; total carbohydrate includes fibre).
Portions are mostly cups, ounces and fluid ounces; only 53 lines are in grams.

## Method

`src/foodspace/freetext.py`. For each distinct item name:

1. Clean the name: drop the brand in brackets, expand app shorthand ("std bfast"), and translate
   US wording to the wording of the Swedish/Finnish English names (chips to crisps, cookie to
   biscuit, frosted flakes to corn flakes; 129 entries).
2. Name similarity to every reference food (the TF-IDF index already used to place new foods),
   mixed 60/40 with the similarity of the name's head noun ("Chicken **Biscuit**", "Salad with
   Fried Chicken" -> salad), minus 0.1 for each dish word the candidate has and the item lacks
   ("Apple" should not become "Apple pie").
3. **Composition check**: the reported macros give the item's energy profile (shares of energy
   from protein, fat, sugars, other available carbohydrate, fibre), which does not depend on the
   portion. Among the 30 best names, the score is name similarity - 0.5 x profile distance.
4. Accept when name similarity >= 0.35 and profile distance <= 0.3 (fixed before the review).

## Results

### Coverage

| | Name only | Name + composition |
|---|---|---|
| Distinct names accepted | 64% | **72%** |
| Lines accepted | 73% | **79%** |
| Energy accepted | 69% | **78%** |

Without the US-to-Nordic wording table, 68% of names are accepted (vs 72%).

### Accuracy (hand review)

The matcher's choice was reviewed for 250 randomly drawn distinct names, none of them among the
270 names looked at while writing the wording table (`data/curated/bigideas_match_review.csv`): C = right food or an
acceptable stand-in, G = wrong food but right GIFT group, W = wrong group. The second set
(holdout 2, 99 names) was drawn after the last change to the matcher, so it is the clean
estimate; holdout 1 prompted that last change (the dish-word penalty). **The verdicts were made by Claude, not by a nutritionist, and need checking.**

| Holdout 2 (clean) | Name only | Name + composition |
|---|---|---|
| Top candidate is the right food | 72% | **76%** |
| Top candidate in the right GIFT group | 85% | **88%** |
| Accepted | 67% | 72% |
| Right food among accepted | 86% | 83% |
| Right GIFT group among accepted | 96% | 92% |

Holdout 1 (148 names, reviewed on the previous version, scored only where the choice did not
change since): right food 67% vs 60% name only, right group 85% vs 77%.

So the composition check moves about 4 to 7 points of items onto the right food and raises
coverage, but the accepted set is not cleaner: it accepts more, and a few of those are wrong.
At about 83% right food and 92% right group among accepted items, automatic mapping is good
enough for GIFT-group level analysis, not for food-level analysis without a review step.

### What fails (84 wrong choices in the two review sets)

| Type | n | Examples |
|---|---|---|
| Composite and restaurant dishes, sandwiches | 26 | Chipotle bowl, Subway BMT, chicken parmesan, Salisbury steak |
| US branded products with no Nordic counterpart | 17 | Reese's cups, Kashi Go Lean, Hamburger Helper, Tootsie Roll |
| Generic one-word names sent to a dish or variant | 13 | cheese -> cheese fondue, yogurt -> yoghurt sauce |
| Wording differences the table misses | 13 | half and half, spring mix, rice cakes, creamers |
| US foods absent from the Nordic tables | 11 | clam chowder, onion rings, peach cobbler, kombucha, cheese curds |
| Near variant with different composition | 4 | lightly salted almonds -> candied almonds |

Composite dishes are the largest single gap here too, as in the simulated records. The
wording and generic-name failures can be fixed with more table entries and a "basic food
first" rule; branded US products and US dishes need a US source (FNDDS recipes) or manual
mapping.

### Reported macros vs reference composition

* Energy profile distance of accepted items (0 = same shares, 1 = opposite): median 0.09,
  90th percentile 0.22. For most items the reference food has a similar macro balance.
* For the 177 accepted, weighed lines (grams or ounces, drinks excluded), reported vs reference
  per 100 g: Spearman 0.82 to 0.90 per macro, but only 46% of lines within 25% for energy.
  Reported foods are denser than their reference match (median 163 vs 120 kcal/100 g), partly
  because "ounce" is used loosely and cooked/raw states differ (the matcher has no water
  content to go on).

### Glucose (secondary)

428 meals had a full CGM window and no other meal in the next 2 hours; 345 had at least 10 g
available carbohydrate. iAUC over 0 to 120 min above the 30-min pre-meal baseline.

* iAUC rises with carbohydrate grams (Spearman 0.27).
* Carbohydrate, fat, protein and person explain 14% of log iAUC; adding the GIFT group of the
  meal's main carbohydrate source raises it to 18% (F = 1.59, 9 df, p = 0.12). No clear group
  effect at this size. Sugar share of carbohydrate adds nothing (15%).
* Per gram of carbohydrate, meals led by milk products (lactose; 12 meals, 5 people) and
  cereals were highest and composite dishes lowest (`results/bigideas/glucose_by_group.csv`),
  but the cells are small and some main-source labels are mis-mapped dishes.

## Caveats

* PhysioNet was not reachable from the analysis machine; the files came from a public GitHub
  mirror and were checked against PhysioNet's SHA256SUMS (the food logs differ only in line
  endings).
* The macro check compares US label values with Nordic analysed values, so part of the
  distance is real composition difference between countries, not mismatch.
* The reference has no alcohol column, so wine and beer energy is understated on its side.
* Per-line and per-meal tables are written to `results/private/bigideas/` (not in git).

## Next steps

1. A nutritionist pass over the review file and over the 20 biggest unmatched energy items.
2. Add a "basic food first" rule for one-word names, and more wording entries.
3. Split dishes into ingredients (FNDDS recipes for US dishes) before food-level analysis.
4. Repeat on CGMacros (45 people, photos and macros) for a larger glucose test.
