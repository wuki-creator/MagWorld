#!/usr/bin/env python3
"""Synthetic regression tests for MagRSI recursion and VCC conversion."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from rsi_magnet_cell import rollmean, rsi_wilder

ROOT = Path(__file__).resolve().parent


def write_source(path, counts):
    n_rows, n_genes = counts.shape
    rows, cols = np.nonzero(counts)
    with h5py.File(path, "w") as out:
        out.attrs["encoding-type"] = "anndata"
        out.attrs["encoding-version"] = "0.1.0"
        x = out.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["encoding-version"] = "0.1.0"
        x.create_dataset("data", data=counts[rows, cols].astype(np.float32))
        x.create_dataset("indices", data=cols.astype(np.int64))
        x.create_dataset("indptr", data=np.concatenate(
            ([0], np.cumsum(np.count_nonzero(counts, axis=1)))).astype(np.int64))
        obs = out.create_group("obs")
        strings = h5py.string_dtype("utf-8")
        obs.create_dataset("_index", data=np.array(
            [f"cell{i}" for i in range(n_rows)], dtype=object), dtype=strings)
        for name, values in {
            "context": [f"ctx{i % 2}" for i in range(n_rows)],
            "target_gene": [f"target{i % 2}" for i in range(n_rows)],
        }.items():
            group = obs.create_group(name)
            categories = np.array(sorted(set(values)), dtype=object)
            lookup = {value: index for index, value in enumerate(categories)}
            group.create_dataset("categories", data=categories, dtype=strings)
            group.create_dataset("codes", data=np.array(
                [lookup[value] for value in values], dtype=np.int8))
        var = out.create_group("var")
        var.create_dataset("gene_name", data=np.array(
            [f"gene{i}" for i in range(n_genes)], dtype=object), dtype=strings)


class MagRSITests(unittest.TestCase):
    def test_rollmean_removes_constant_local_baseline(self):
        values = np.full((2, 20), 3.0, dtype=np.float32)
        np.testing.assert_allclose(rollmean(values, 4), values)

    def test_rsi_extremes_and_warmup(self):
        gains = np.ones((1, 20), dtype=np.float32)
        losses = -gains
        gain_rsi = rsi_wilder(gains, 4)
        loss_rsi = rsi_wilder(losses, 4)
        self.assertTrue(np.all(gain_rsi[:, :4] == 50.0))
        self.assertTrue(np.all(gain_rsi[:, 4:] > 99.0))
        self.assertTrue(np.all(loss_rsi[:, 4:] < 1.0))

    def test_support_recursion_and_integer_sparse_conversion(self):
        counts = np.array([
            [0, 2, 0, 5, 1, 0, 8, 0],
            [1, 0, 4, 0, 0, 3, 0, 2],
            [0, 3, 1, 0, 6, 0, 2, 1],
            [4, 0, 0, 2, 0, 5, 1, 0],
        ], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "source.h5ad"
            single = tmp / "single.nc"
            recursive = tmp / "recursive.nc"
            converted = tmp / "converted.h5ad"
            write_source(source, counts)
            subprocess.run([
                sys.executable, str(ROOT / "rsi_magnet_cell.py"),
                "--pred", str(source), "--out", str(single),
                "--window", "2", "--alpha", "1", "--theta", "0", "--block", "2",
            ], check=True)
            with h5py.File(single, "r") as result:
                transformed = result["expression"][:]
                self.assertTrue(np.array_equal(transformed == 0, counts == 0))
                self.assertTrue(np.isfinite(transformed).all())
                self.assertTrue((transformed >= 0).all())
            subprocess.run([
                sys.executable, str(ROOT / "rsi_recurse.py"),
                "--nc", str(single), "--out", str(recursive),
                "--initial-round", "1", "--rounds", "1",
                "--window", "2", "--block", "2",
            ], check=True)
            with h5py.File(recursive, "r") as result:
                self.assertNotIn("expression_r1", result)
                self.assertIn("expression_r2", result)
                log = json.loads(result.attrs["round_log"])
                self.assertEqual(len(log), 1)
                self.assertEqual(log[0]["round"], 2)
                self.assertTrue(log[0]["relative_delta"] >= 0)
            subprocess.run([
                sys.executable, str(ROOT / "nc_to_h5ad.py"),
                "--nc", str(recursive), "--template", str(source),
                "--out", str(converted), "--expression-key", "expression_r2",
                "--block", "2",
            ], check=True)
            with h5py.File(converted, "r") as result:
                self.assertEqual(result.attrs["encoding-type"], "anndata")
                self.assertEqual(result.attrs["encoding-version"], "0.1.0")
                self.assertTrue(np.array_equal(result["X"].attrs["shape"], counts.shape))
                data = result["X/data"][:]
                indices = result["X/indices"][:]
                indptr = result["X/indptr"][:]
                self.assertTrue(np.array_equal(data, np.rint(data)))
                self.assertTrue((data > 0).all())
                self.assertTrue((indices >= 0).all() and (indices < counts.shape[1]).all())
                self.assertEqual(indptr[-1], len(data))
                self.assertLessEqual(float(result.attrs["magrsi_max_cell_total"]), 1_000_000)
                for row in range(counts.shape[0]):
                    self.assertEqual(
                        indptr[row + 1] - indptr[row],
                        np.count_nonzero(result["X/data"][indptr[row]:indptr[row + 1]]),
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
