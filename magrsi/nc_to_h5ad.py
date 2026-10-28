#!/usr/bin/env python3
"""Convert MagRSI NetCDF output to a VCC-compatible sparse AnnData H5AD.

The conversion is out-of-core, rounds continuous predictions to raw integer
counts, drops explicit zeros, and preserves the source observation metadata.
"""
import argparse
import time

import h5py
import numpy as np


def categorical_values(group, column, n_rows):
    dataset = group[column]
    if isinstance(dataset, h5py.Dataset):
        return dataset[:n_rows].astype(str)
    return dataset["categories"][:].astype(str)[dataset["codes"][:n_rows]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", required=True)
    parser.add_argument("--template", required=True,
                        help="source H5AD whose obs/var/uns metadata are copied")
    parser.add_argument("--out", required=True)
    parser.add_argument("--expression-key", default="expression",
                        help="for example expression_r2 for a recursive output")
    parser.add_argument("--block", type=int, default=1000)
    args = parser.parse_args()
    if args.block < 1:
        parser.error("--block must be positive")

    started = time.time()
    src = h5py.File(args.template, "r")
    nc = h5py.File(args.nc, "r")
    if args.expression_key not in nc:
        parser.error(f"{args.expression_key!r} is absent from {args.nc}")
    expression = nc[args.expression_key]
    n_rows, n_genes = expression.shape

    source_indptr = src["X/indptr"] if "X/indptr" in src else None
    source_rows = source_indptr.shape[0] - 1 if source_indptr is not None else src["X"].shape[0]
    source_genes = src["var/gene_name"].shape[0]
    if (n_rows, n_genes) != (source_rows, source_genes):
        raise ValueError(
            f"NetCDF shape {(n_rows, n_genes)} does not match template "
            f"{(source_rows, source_genes)}"
        )
    nc_contexts = nc["context"][:n_rows].astype(str)
    template_contexts = categorical_values(src["obs"], "context", n_rows)
    if not np.array_equal(nc_contexts, template_contexts):
        raise ValueError("Context order differs between NetCDF and template")
    nc_targets = nc["target_gene"][:n_rows].astype(str)
    template_targets = categorical_values(src["obs"], "target_gene", n_rows)
    if not np.array_equal(nc_targets, template_targets):
        raise ValueError("Perturbation order differs between NetCDF and template")

    out = h5py.File(args.out, "w")
    for key, value in src.attrs.items():
        out.attrs[key] = value
    for key in src.keys():
        if key != "X":
            src.copy(src[key], out, key)
    out.attrs["magrsi_expression_key"] = args.expression_key
    out.attrs["magrsi_count_conversion"] = "rint; clip(0, 1e6 total per cell)"

    x_group = out.create_group("X")
    x_group.attrs["shape"] = np.asarray([n_rows, n_genes], dtype=np.int64)
    data_ds = x_group.create_dataset(
        "data", shape=(0,), maxshape=(None,), dtype="float32",
        chunks=(65536,), compression="gzip", compression_opts=4,
    )
    indices_ds = x_group.create_dataset(
        "indices", shape=(0,), maxshape=(None,), dtype="int64",
        chunks=(32768,), compression="gzip", compression_opts=4,
    )
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    total_nnz = 0
    max_total = 0.0
    changed = 0

    for start in range(0, n_rows, args.block):
        stop = min(start + args.block, n_rows)
        block = expression[start:stop].astype(np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f"Non-finite prediction in rows {start}:{stop}")
        rounded = np.rint(np.maximum(block, 0.0)).astype(np.float32)
        changed += int(np.count_nonzero(rounded != block))
        totals = rounded.sum(axis=1)
        over = totals > 1_000_000
        if np.any(over):
            rounded[over] *= (1_000_000.0 / totals[over])[:, None]
            rounded = np.floor(rounded).astype(np.float32)
            totals = rounded.sum(axis=1)
        max_total = max(max_total, float(totals.max(initial=0.0)))
        for local_row in range(stop - start):
            row = rounded[local_row]
            nonzero = np.flatnonzero(row)
            count = nonzero.size
            if count:
                new_total = total_nnz + count
                data_ds.resize(new_total, axis=0)
                indices_ds.resize(new_total, axis=0)
                data_ds[total_nnz:new_total] = row[nonzero]
                indices_ds[total_nnz:new_total] = nonzero
                total_nnz = new_total
            indptr[start + local_row + 1] = total_nnz
        print(f"[{stop}/{n_rows}] {time.time() - started:.0f}s nnz={total_nnz}", flush=True)

    x_group.create_dataset(
        "indptr", data=indptr, chunks=(min(8192, indptr.size),),
        compression="gzip", compression_opts=4,
    )
    x_group.attrs["encoding-type"] = "csr_matrix"
    x_group.attrs["encoding-version"] = "0.1.0"
    out.attrs["magrsi_rounded_values"] = changed
    out.attrs["magrsi_max_cell_total"] = max_total
    out.close()
    nc.close()
    src.close()
    print(f"DONE {time.time() - started:.1f}s nnz={total_nnz} "
          f"rounded={changed} max_cell_total={max_total:.0f}", flush=True)


if __name__ == "__main__":
    main()
