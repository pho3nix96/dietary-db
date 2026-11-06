configfile: "config/config.yaml"

OUT = config["output_dir"]
GIFT = "data/raw/gift/foodex2_gift_nut.csv"   # FAO/WHO GIFT matching table extract (not in git)
REC = "data/simulated/food_records"           # simulated food records (committed)

rule all:
    input:
        f"{OUT}/food_map.csv",
        f"{OUT}/misfits.csv",
        f"{OUT}/resolution_scan.csv",
        f"{OUT}/metrics.json",
        f"{OUT}/cereals/cluster_summary.csv",
        f"{OUT}/food_tree/nodes.csv",
        f"{OUT}/within_gift/groups.csv",
        f"{OUT}/within_gift_fi/groups.csv",
        f"{OUT}/within_gift_nordic/groups.csv",
        f"{OUT}/recommendation_groups/groups.json",
        f"{OUT}/person_graphs/person_summary.csv",
        f"{OUT}/cointake/graph.json",
        f"{OUT}/diet_similarity/summary.json",

rule prepare_foods:
    input:
        "data/raw/livsmedelsverket/nutrients_wide.csv",
        "data/raw/livsmedelsverket/foods.csv",
    output:
        config["input"],
    shell:
        "python scripts/prepare_livsmedelsverket.py --out {output}"

rule run_pipeline:
    input:
        foods=config["input"],
        cfg="config/config.yaml",
    output:
        f"{OUT}/food_map.csv",
        f"{OUT}/misfits.csv",
        f"{OUT}/resolution_scan.csv",
        f"{OUT}/metrics.json",
    shell:
        "python -m foodspace.run {input.cfg}"

rule within_cereals:
    input:
        foods=config["input"],
        rav="data/raw/livsmedelsverket/raw_commodities.csv",
        table=config["fractions"]["curated_table"],
    output:
        f"{OUT}/cereals/cluster_summary.csv",
    shell:
        "python scripts/within_group_cereals.py --commodities {input.rav} --out {OUT}/cereals"

rule food_tree:
    input:
        f"{OUT}/cereals/cluster_summary.csv",
    output:
        f"{OUT}/food_tree/nodes.csv",
    shell:
        "python scripts/food_tree.py --foods {OUT}/cereals/clusters.csv --out {OUT}/food_tree"

rule gift_mapping:
    input:
        "data/raw/gift/foodex2_gift_nut.csv",
    output:
        "data/curated/foodex2_to_gift.csv",
    shell:
        "python scripts/map_gift.py"

rule within_gift_groups:
    input:
        config["input"],
        GIFT,  # short group names for map.json (build_gift_map_data.py)
    output:
        f"{OUT}/within_gift/groups.csv",
    shell:
        "python scripts/within_gift_groups.py --out {OUT}/within_gift && python scripts/build_gift_map_data.py"

rule prepare_fineli:
    input:
        "data/raw/fineli/unz/food.csv",
        "data/curated/fineli_fuclass_to_gift.csv",
        "data/curated/protein_fractions.csv",
    output:
        "data/processed/foods_fi.csv",
    shell:
        "python scripts/prepare_fineli.py"

rule within_gift_groups_fineli:
    input:
        "data/processed/foods_fi.csv",
        GIFT,
    output:
        f"{OUT}/within_gift_fi/groups.csv",
    shell:
        "python scripts/within_gift_groups.py --input {input[0]} --out {OUT}/within_gift_fi"
        " && python scripts/build_gift_map_data.py {OUT}/within_gift_fi {input[0]}"

rule combine_nordic:
    input:
        "data/processed/foods_lv.csv",
        "data/processed/foods_fi.csv",
    output:
        "data/processed/foods_nordic.csv",
    shell:
        "python scripts/combine_databases.py"

rule within_gift_groups_nordic:
    input:
        "data/processed/foods_nordic.csv",
        GIFT,
    output:
        f"{OUT}/within_gift_nordic/groups.csv",
    shell:
        "python scripts/within_gift_groups.py --input {input[0]} --out {OUT}/within_gift_nordic"
        " && python scripts/build_gift_map_data.py {OUT}/within_gift_nordic {input[0]}"

# --- food records (simulated by default; see data/simulated/food_records/README.md) ---
# placed_unique_foods.csv is made by scripts/place_unique_foods.py and committed with the records.

rule recommendation_groups:
    input:
        config["input"],
        f"{REC}/consumption.csv",
    output:
        f"{OUT}/recommendation_groups/groups.json",
    shell:
        "python scripts/recommendation_groups.py --records {REC} --out {output}"

rule person_graphs:
    input:
        f"{REC}/consumption.csv",
        f"{REC}/placed_unique_foods.csv",
        config["input"],
    output:
        f"{OUT}/person_graphs/person_summary.csv",
    shell:
        "python scripts/person_graphs.py --records {REC} --foods {input[2]} --out {OUT}/person_graphs --graphml"

rule cointake_graph:
    input:
        f"{REC}/consumption.csv",
        f"{REC}/placed_unique_foods.csv",
        f"{OUT}/within_gift/groups.csv",
    output:
        f"{OUT}/cointake/graph.json",
    shell:
        "python scripts/build_cointake_graph.py --records {REC} --trees {OUT}/within_gift/trees.json --out {output}"

rule diet_similarity:
    input:
        config["input"],
        f"{REC}/consumption.csv",
        f"{REC}/placed_unique_foods.csv",
    output:
        f"{OUT}/diet_similarity/summary.json",
    shell:
        "python scripts/compare_diets.py --records {REC} --foods {input[0]} --out {OUT}/diet_similarity"
