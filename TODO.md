# Food space map: plan

## 1. Define scope
Decisions here change every cluster downstream, so settle them before touching data.

- [x] Pick the dietary database: Swedish Food Agency database for now (Fineli not reachable outside Finland)
- [x] Choose the nutrient set: macronutrients only for now (micronutrients and bioactives later)
- [x] Choose the normalization basis: per 100 g
- [ ] Decide how to treat raw vs cooked and processed forms
- [ ] State the main goal: alternative classification, test of official groups, or food layer for the multi-omics database
- [ ] Choose the fraction blocks: fatty acids, individual sugars, starch and fibre types, amino acids, protein fractions
- [ ] Run a systematic literature search (Scopus or Web of Science) on grouping foods by nutrient fractions before claiming novelty

## 2. Prepare the data
Key every food on FoodEx2 so all four GIFT groupings come for free.

- [x] Export food names, descriptions and nutrient columns
- [x] Map each food to a FoodEx2 base term (included in the Swedish data)
- [ ] Assign the GIFT nutrition-sensitive group and subgroup (19 groups, 95 subgroups)
- [ ] Audit missing nutrient values and set an imputation or exclusion rule
- [ ] Log-transform skewed nutrients and check compositional constraints
- [x] Write the download script for the Swedish database (`scripts/fetch_livsmedelsverket.py`)
- [x] Run the download on your computer and share the CSVs (2,606 foods, 59 components)
- [x] Check coverage: all macro and fraction columns are 100% complete; sugar parts sum to total sugars; fatty-acid classes cover ~93% of fat (median)
- [x] Build a curated protein-fraction table with a source per row, organised by GIFT group: casein and whey (dairy), Osborne fractions (cereals, pulses), 11S/7S (soybean), myofibrillar, sarcoplasmic and stromal (meat, fish)
- [ ] Fill the curated table for dairy first: milk, cheese, yogurt and other fermented products (started: milk sourced, rennet cheese provisional, 6 categories to source)
- [x] Map family-specific protein fractions onto a common solubility-based scheme (`common_class`, `scheme: common`)
- [x] Cereal Osborne fractions: barley sourced; oats, maize, rice provisional; rye from a textbook table (verify)
- [x] Download each food's raw-commodity recipe (`raw_commodities.csv`) and build commodity-mix protein profiles (`commodities.py`)
- [x] Within-group mode for cereals (`scripts/within_group_cereals.py`): totals vs fractions scored against dominant grain and food group
- [ ] Rye: replace textbook values with a modern primary source (RP-HPLC secalin quantification)
- [ ] Separate bran and germ from whole grain in the profiles (more albumins and globulins)
- [x] Food tree with the treelet transform on composition profiles (`scripts/food_tree.py`), cereals first
- [ ] Food tree on all foods, and with block weights (e.g. protein fractions only)
- [ ] Build the intake-based treelet tree (foods eaten together) when intake data are available and compare it with the composition tree
- [x] Fineli loader with curated FUCLASS -> GIFT map and recipe-expanded protein profiles
- [ ] Review the FUCLASS -> GIFT map and name rules (fish, fruit, vegetable subgroups)
- [ ] Source yogurt, quark, cream, egg, processed meat, offal, potato, nut, pea fractions: in Fineli these block the protein profile of many dishes
- [x] Recipe-based protein profiles for Swedish foods too (1,026 foods), same method as Fineli
- [x] Combined Nordic table + country-mixing check + SE<->FI crosswalk
- [x] Names removed from the space; used only to place new foods (`placement.py`, `place_food.py`)
- [ ] Improve name placement for flavour-dominated names (e.g. "oat yoghurt, strawberry" goes to sweets)
- [ ] Review the crosswalk by hand for a sample of food groups
- [ ] Use commodity-mix profiles in the main pipeline instead of one category per food
- [ ] Repeat the within-group mode for dairy and pulses once their fractions are sourced
- [ ] Source the remaining protein fractions: pea, lentil, chickpea, beans, peanut, nuts, seeds; fish; processed meat, offal; egg; potato
- [x] Fish muscle fractions sourced (Shimizu et al. 1960; Hashimoto et al. 1979); smoked and fried fish mapped to them
- [ ] Replace provisional values with primary sources: rennet cheese (bovine), rice (full Osborne split), faba bean (non-albumin split), meat (now a textbook value; need a primary beef/pork study using the same extraction as the fish data)
- [ ] Fish and shellfish dishes and shellfish: give them a protein profile (recipe-based, as for cereals)
- [ ] Load the official FAO GIFT file (FoodEx2_MTX.xlsx): check group names and fill `gift_subgroup` in the protein table

## 3. Build the similarity views
Same foods, several distances: what a food is, how much it contains, what kind of nutrients it contains, how it is described.

- [x] Classification distance from the GIFT tree (`views.gift_distance`)
- [ ] Add FoodEx2 facets for processing and part
- [x] Nutrient distance on normalized profiles (`views.nutrient_distance`)
- [x] Semantic embeddings with a multilingual model (`views.semantic_embeddings`, untested on real data)
- [x] k-nearest-neighbour graph per view (`fuse.affinity`, `fuse.knn_sparsify`)
- [x] Fraction-profile view: each fraction as a share of its parent total, log-ratio transformed, each block weighted equally (`fractions.py`, tested on synthetic foods)
- [ ] Check whether amino acid profiles reproduce the curated casein vs whey differences

## 4. Fuse and cluster
Disagreement between views is the signal worth reporting.

- [x] Weighted-sum fusion baseline (`fuse.weighted_fusion`)
- [x] Similarity network fusion (`fuse.snf`)
- [x] Leiden clustering with a resolution scan (`fuse.leiden`, `evaluate.resolution_scan`)
- [x] Compare clusters with GIFT groups: ARI, NMI (`evaluate.agreement`)
- [x] List misfit foods (`evaluate.misfits`)
- [ ] Robustness across normalization choices and bootstrap resamples
- [x] Keep GIFT out of the fusion and use it only as the reference for comparison (`fusion.gift_in_fusion: false`)
- [x] Check that dairy separates: dairy foods' 5 nearest neighbours are dairy 43% (totals) vs 72% (with fractions); hard cheese 31% vs 84%
- [ ] Review the auto-drafted protein category map (`data/curated/protein_category_map.csv`, 502 foods)
- [ ] Look into groups that lose cohesion with fractions (sauces 48% to 31%, soups)
- [ ] Report a sweep of GIFT weights to show how much structure comes from the classification itself

## 5. Visualize
An interactive map of food space, also usable as a QA tool for the database.

- [x] 2D layout from the fused similarity (`layout.embed_2d`)
- [x] Interactive map on real data: highlight a food family, hover and click for nutrients and nearest foods, switch totals vs fractions (`docs/nordic-food-space-map.html`)
- [ ] Toggles for each view and sliders for fusion weights
- [ ] Highlight disagreements and nutrient outliers for data checking
- [ ] Optional: overlay consumption frequency from survey data

## 6. Package and release
Built to keep working when you stop working on it.

- [x] Reproducible pipeline (Snakemake) with config and tests
- [x] Push to a GitHub repository
- [ ] Choose licences (code and data)
- [ ] Archive each release on Zenodo with a DOI
- [ ] Choose the outlet: data or methods paper, standalone tool, or layer of the multi-omics database
- [ ] Use the 1985 result where whole milk and cheese clustered together on totals (Windham et al.) as the motivating example

## 7. Compare diets (food records)
The reference as a tool for dietary records: people are compared by what they eat, using food similarity.

- [x] Match record lines to reference foods: food id, FoodEx2 base term (+ facets), name (`diet.match_records`)
- [x] Fill missing nutrient values from the matched or placed reference food (`diet.fill_nutrients`)
- [x] Diet profiles on the GIFT food tree and a tree-aware person distance (earth mover's distance) plus a nutrient view
- [x] Check on the simulated records: 5 nearest people share the hidden pattern 50% of the time (chance 28%)
- [ ] Split composite dishes into ingredient groups from the Swedish recipes (needs `raw_commodities.csv`)
- [ ] Get real records (GIFT layout) into `data/private/`; check whether they are already at ingredient level
- [ ] Compare the tree distance with simpler profiles (GIFT-group shares, servings per recommendation group)
- [ ] Enable GitHub Pages so the explorer site deploys
