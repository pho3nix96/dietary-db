# Food Space Map: project index

Snapshot of the whole project (code, data, results, published pages), 2026-11-04.

## Published pages (claude.ai artifacts)
- Nordic Food Space Map: https://claude.ai/artifact/F88VFMjkgC5LAv5ad3mB3r  (artifacts/nordic-food-space-map.html)
- Food Space Tasks checklist: https://claude.ai/artifact/Jc7prNAeU7dH78EtuhouLp  (artifacts/food-space-tasks.html)
- Food Co-intake Graph: https://claude.ai/artifact/2Smim9ZMmDD9kL4drdPbAu  (artifacts/food-cointake-graph.html)
- Recommendation Group Trees (3D/2D, servings): https://claude.ai/artifact/MByU6mfAcc5TmTbKv8PsVM  (artifacts/food-group-servings.html)
- Doc "How the treelet transform measures similarity": https://claude.ai/code/artifact/ca45d561-34d6-4d05-b7cd-41d253294599

## Where things are
- src/foodspace/      library: fractions, fusion, treelet, layout, placement of new foods, commodities
- scripts/            pipeline steps (prepare_livsmedelsverket, prepare_fineli, combine_databases, map_gift,
                      within_gift_groups, recommendation_groups, simulate_food_records, place_unique_foods,
                      build_cointake_graph, person_graphs, ...); Snakefile runs them in order
- config/config.yaml  every choice that changes the clusters
- data/curated/       hand-curated tables (protein fractions, FoodEx2->GIFT, Fineli class->GIFT)
- data/processed/     Swedish, Finnish and combined Nordic food tables, ready for the pipeline
- data/simulated/     simulated 3-day food records (100 people) incl. 225 unique products
- results/            within-GIFT-group clusters and trees (SE, FI, Nordic), recommendation-group trees,
                      co-intake graph, per-person graphs in servings (CSV + GraphML)
- src/foodspace/diet.py + scripts/compare_diets.py   match records to the reference, diet profiles on the food tree, person-to-person distance
- results/diet_similarity/   person distance matrix, clusters, matching report (simulated records)
- tests/              pytest (PYTHONPATH=src)

## Decisions so far
- GIFT groups are fixed; clustering only within a group. Names never shape clustering; they only place new foods.
- Similarity = equal-weight fusion of macronutrient distance and fraction distance (fat, sugar, carbohydrate, protein fractions).
- Recommendation groups: grains, legumes, nuts, seeds, milk products, meat + eggs, seafood, fruits, vegetables.
- Servings: 7 g protein (milk products, nuts, seeds, legumes); 14 g protein (meat + eggs, seafood);
  30 g starch (grains); 80 g food (fruits, vegetables).
- Person graphs: nodes = foods (servings/day); edges = same meal, min(servings) summed per day.

## Open items
- Split composite dishes into ingredients so they count toward the groups (biggest gap).
- Normalise servings by recommendation targets and switch edges to geometric mean; choose NNR2023 or Latvian targets.
- Replace simulated records with real data (e.g. INRAN-SCAI via FAO GIFT; needs the user's own access agreement).
- Check branch stability (bootstrap); compare treelet levels with average linkage.
- Test whether co-intake edges add information beyond servings-per-branch profiles.

Raw downloads (Livsmedelsverket API, Fineli package, GIFT matching table) are not included; scripts re-create them.
