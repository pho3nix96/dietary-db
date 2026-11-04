"""Run the whole pipeline from a config file.

    python -m foodspace.run config/config.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import evaluate, fractions, fuse, io, layout, views


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    cols, out = cfg["columns"], Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 0)

    df = io.load_foods(cfg["input"], cols)
    print(f"Loaded {len(df)} foods")

    # Views. GIFT joins the fusion only when asked; by default it is the reference for comparison.
    dists = {}
    if cfg["fusion"].get("gift_in_fusion", False):
        dists["gift"] = views.gift_distance(df[cols["gift_group"]], df[cols["gift_subgroup"]])
    Xp, prep_report = views.prepare_nutrients(
        df[cols["nutrients"]],
        max_missing=cfg["nutrients"]["max_missing"],
        impute=cfg["nutrients"]["impute"],
        groups=df[cols["gift_group"]],
        log=cfg["nutrients"]["log1p"],
    )
    dists["nutrients"] = views.nutrient_distance(Xp, cfg["nutrients"]["metric"])
    if cfg["semantics"]["enabled"]:
        E = views.semantic_embeddings(io.food_texts(df, cols), cfg["semantics"]["model"])
        np.save(out / "embeddings.npy", E)
        dists["semantics"] = views.embedding_distance(E)
    fraction_report = None
    fcfg = cfg.get("fractions", {})
    if fcfg.get("enabled"):
        curated = None
        if fcfg.get("curated_table") and Path(fcfg["curated_table"]).exists():
            curated = pd.read_csv(fcfg["curated_table"])
        blocks = fractions.build_blocks(df, fcfg, curated)
        if blocks:
            dists["fractions"], fraction_report = fractions.fraction_distance(blocks)
            print(f"Fraction view built from blocks: {', '.join(blocks)}")
        else:
            print("Fraction view skipped; no usable blocks.")

    # Fuse
    k = cfg["graph"]["k"]
    if cfg["fusion"]["method"] == "snf":
        W = fuse.snf([fuse.affinity(D, k) for D in dists.values()], k=k, t=cfg["fusion"]["snf_iterations"])
    else:
        Dw = fuse.weighted_fusion(dists, cfg["fusion"]["weights"])
        W = fuse.knn_sparsify(fuse.affinity(Dw, k), k)
    W_graph = fuse.knn_sparsify(W, k)
    n_comp = fuse.n_components(W_graph)
    if n_comp > 1:
        print(f"Warning: the food graph has {n_comp} disconnected parts; clusters cannot cross them. "
              "Raise graph.k or lower the GIFT weight if this is not intended.")

    # Cluster and compare with GIFT
    res = cfg["clustering"]["resolution"]
    clusters = fuse.leiden(W_graph, res, seed)
    metrics = {
        "vs_gift_group": evaluate.agreement(clusters, df[cols["gift_group"]].astype(str)),
        "vs_gift_subgroup": evaluate.agreement(clusters, df[cols["gift_subgroup"]].astype(str)),
        "views": list(dists),
        "fusion": cfg["fusion"]["method"],
        "graph_components": n_comp,
        "nutrient_preparation": prep_report,
        "fraction_blocks": fraction_report,
    }
    scan = evaluate.resolution_scan(W_graph, df[cols["gift_group"]].astype(str), cfg["clustering"]["scan"])
    mis = evaluate.misfits(df, clusters, cols["gift_group"], cols["id"], cols["name"])

    # Layout
    xy = layout.embed_2d(layout.similarity_to_distance(W), cfg["layout"]["method"], seed)

    table = pd.DataFrame({
        "id": df[cols["id"]], "name": df[cols["name"]],
        "gift_group": df[cols["gift_group"]], "gift_subgroup": df[cols["gift_subgroup"]],
        "cluster": clusters, "x": xy[:, 0], "y": xy[:, 1],
    })
    table.to_csv(out / "food_map.csv", index=False)
    mis.to_csv(out / "misfits.csv", index=False)
    scan.to_csv(out / "resolution_scan.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["vs_gift_group"], indent=2))
    print(f"{len(mis)} misfit foods. Results in {out}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
