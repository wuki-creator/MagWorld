#!/usr/bin/env python3
"""Convert VCC nullable-string control H5AD files for the legacy predictor."""
import argparse
import shutil
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_strings(group):
    if "values" in group:
        values = group["values"][...]
    else:
        values = group[...]
    result = np.array([
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=object)
    if isinstance(group, h5py.Group) and "mask" in group:
        result[np.asarray(group["mask"][...], dtype=bool)] = "NaN"
    return result


def read_categorical(group):
    categories = [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in group["categories"][...]
    ]
    return pd.Categorical.from_codes(group["codes"][...], categories)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--contexts", default="D,E,F")
    args = parser.parse_args()
    source = Path(args.src)
    destination = Path(args.dst)
    destination.mkdir(parents=True, exist_ok=True)
    contexts = [value.strip() for value in args.contexts.split(",") if value.strip()]

    for context in contexts:
        input_path = source / f"context_{context}.h5ad"
        with h5py.File(input_path, "r") as handle:
            n_cells = handle["X/indptr"].shape[0] - 1
            var_index = handle["var/_index"]
            n_genes = (var_index["values"].shape[0]
                       if isinstance(var_index, h5py.Group) else var_index.shape[0])
            matrix = sp.csr_matrix(
                (handle["X/data"][...], handle["X/indices"][...], handle["X/indptr"][...]),
                shape=(n_cells, n_genes),
            )
            obs = pd.DataFrame(index=pd.Index(read_strings(handle["obs/_index"]), name=None))
            for key in ("context", "ntc_id", "target_gene"):
                obs[key] = read_categorical(handle["obs"][key])
            var = pd.DataFrame(index=pd.Index(read_strings(var_index), name=None))
        adata = ad.AnnData(X=matrix, obs=obs, var=var)
        output_path = destination / f"context_{context}.h5ad"
        adata.write_h5ad(output_path)
        contexts_found = set(adata.obs["context"].astype(str))
        if contexts_found != {context}:
            raise ValueError(f"{input_path}: unexpected context labels {contexts_found}")
        print(context, adata.shape, "written", flush=True)

    for name in ("gene_names.csv", "pert_counts.csv", "manifest.json"):
        shutil.copy2(source / name, destination / name)
    print("CSV and manifest files copied", flush=True)


if __name__ == "__main__":
    main()
