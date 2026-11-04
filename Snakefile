configfile: "config/config.yaml"

OUT = config["output_dir"]

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
        f"{OUT}/cereals/cluster_summary.csv",
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
    output:
        f"{OUT}/within_gift_fi/groups.csv",
    shell:
        "python scripts/within_gift_groups.py --input {input} --out {OUT}/within_gift_fi"
        " && python scripts/build_gift_map_data.py {OUT}/within_gift_fi {input}"

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
    output:
        f"{OUT}/within_gift_nordic/groups.csv",
    shell:
        "python scripts/within_gift_groups.py --input {input} --out {OUT}/within_gift_nordic"
        " && python scripts/build_gift_map_data.py {OUT}/within_gift_nordic {input}"
