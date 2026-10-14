import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from TCR_run import ARCHIVE, image_parts, main, mode_name, mount_arg, read_input, select_methods
sys.path.insert(0, str(ROOT / "core"))
from output_matching import attach_ids

spec = importlib.util.spec_from_file_location("container_cli", ROOT / "core/cli.py")
container_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(container_cli)


class InputTests(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(mode_name("V"), "vregion")
        self.assertEqual(mode_name("cdr"), "cdr3")

    def test_panel_counts(self):
        self.assertEqual(len(select_methods(["all"], "cdr3")), 13)
        self.assertEqual(len(select_methods(["all"], "vregion")), 15)

    def test_subset(self):
        self.assertEqual(select_methods(["panpep,TEINet", "PanPep"], "cdr3"), ["PanPep", "TEINet"])

    def test_invalid_predictors(self):
        with self.assertRaises(ValueError):
            select_methods(["unknown"], "cdr3")
        with self.assertRaises(ValueError):
            select_methods(["MixTCRpred"], "cdr3")

    def test_examples(self):
        self.assertEqual(len(read_input(ROOT / "examples/cdr3_smoke.csv", "cdr3")), 3)
        self.assertEqual(len(read_input(ROOT / "examples/vregion_smoke.csv", "vregion")), 2)

    def test_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text("beta_cdr3,peptide,HLA\ncassirssyeqyf,gilgf vftl,A0201\n")
            row = read_input(path, "cdr3")[0]
            self.assertEqual(row["alpha"], "")
            self.assertEqual(row["mhc"], "HLA-A*02:01")
            self.assertEqual(row["peptide"], "GILGFVFTL")

    def test_bad_rows_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            for content in (
                "beta_cdr3,peptide,HLA\n,GILGFVFTL,A0201\n",
                "beta_cdr3,peptide,HLA\nCASSX,GILGFVFTL,A0201\n",
                "beta_cdr3,peptide,HLA\nCASS,GILGFVFTL\n",
                "beta_cdr3,peptide,HLA\nCASS,GILGFVFTL,BAD\n",
                "beta_cdr3,beta,peptide,HLA\nCASS,CASS,GILGFVFTL,A0201\n",
            ):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    read_input(path, "cdr3")

    def test_wrong_mode(self):
        with self.assertRaises(ValueError):
            read_input(ROOT / "examples/vregion_smoke.csv", "cdr3")

    def test_mount_escaping(self):
        value = mount_arg(Path("/tmp/a,b with spaces.csv"), "/input/candidates.csv", True)
        fields = next(csv.reader([value]))
        self.assertIn("src=/tmp/a,b with spaces.csv", fields)
        self.assertIn("readonly", fields)


class EntrypointTests(unittest.TestCase):
    def test_container_dispatches_without_docker(self):
        with patch.dict(os.environ, {"TCRANK_IN_CONTAINER": "1"}), \
             patch.dict(sys.modules, {"cli": container_cli}), \
             patch.object(container_cli, "main", return_value=0) as pipeline, \
             patch("TCR_run.docker_path") as docker:
            result = main(["--input", "input.csv", "--output", "results", "--mode", "v",
                           "--predictors", "panpep,TEINet", "--threads", "3", "--keep-logs"])
            self.assertEqual(result, 0)
            pipeline.assert_called_once_with([
                "--input", "input.csv", "--output", "results", "--mode", "vregion",
                "--threads", "3", "--predictors", "PanPep", "TEINet", "--keep-logs"])
            docker.assert_not_called()

    def test_container_rejects_host_options(self):
        with patch.dict(os.environ, {"TCRANK_IN_CONTAINER": "1"}), \
             patch("TCR_run.docker_path") as docker:
            self.assertEqual(main(["--load-image", "image.tar.gz"]), 2)
            docker.assert_not_called()


class ImageTests(unittest.TestCase):
    def make_parts(self, root):
        (root / "images").mkdir()
        records = []
        for index, content in enumerate((b"first chunk", b"second chunk")):
            name = "images/" + ARCHIVE + ".part-{:02d}".format(index)
            (root / name).write_bytes(content)
            records.append(hashlib.sha256(content).hexdigest() + "  " + name)
        (root / "SHA256SUMS.txt").write_text("\n".join(records) + "\n")

    def test_parts_verified_in_order(self):
        with tempfile.TemporaryDirectory() as directory, patch("TCR_run.ROOT", Path(directory)):
            root = Path(directory)
            self.make_parts(root)
            self.assertEqual([path.read_bytes() for path in image_parts()], [b"first chunk", b"second chunk"])

    def test_corruption_and_missing_parts(self):
        with tempfile.TemporaryDirectory() as directory, patch("TCR_run.ROOT", Path(directory)):
            root = Path(directory)
            self.make_parts(root)
            part = root / "images" / (ARCHIVE + ".part-01")
            part.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                image_parts()
            part.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                image_parts()

    def test_lfs_pointer_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory, patch("TCR_run.ROOT", Path(directory)):
            root = Path(directory)
            self.make_parts(root)
            (root / "images" / (ARCHIVE + ".part-00")).write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "Git LFS pointers"):
                image_parts()

    def test_manifest_rejects_paths_outside_images(self):
        with tempfile.TemporaryDirectory() as directory, patch("TCR_run.ROOT", Path(directory)):
            root = Path(directory)
            (root / "SHA256SUMS.txt").write_text("0" * 64 + "  ../other\n")
            with self.assertRaisesRegex(ValueError, "Unexpected file"):
                image_parts()

class ExportTests(unittest.TestCase):
    def result(self):
        return {
            "ranking_rows": [
                {"pair_id": 0, "group_id": "A", "final_score": 0.0, "available_method_count": 0},
                {"pair_id": 1, "group_id": "A", "final_score": 1.2, "available_method_count": 1},
            ],
            "method_level_rows": [{"pair_id": 1, "method": "PanPep", "raw_score": 0.5}],
            "failed_methods": {},
        }

    def test_unscored_not_ranked(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            summary = container_cli.export_results(self.result(), destination, "cdr3", ["PanPep"], 1)
            self.assertEqual(summary["status"], "partial")
            with (destination / "ranking.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["pmhc_rank"], "1")
            self.assertEqual(rows[1]["pmhc_rank"], "")
            self.assertEqual(rows[1]["ranking_status"], "no_prediction")
            self.assertTrue((destination / "predictor_scores.csv").is_file())
            self.assertEqual(json.loads((destination / "summary.json").read_text())["unscored_candidates"], 1)

    def test_no_usable_method_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.result()
            result["method_level_rows"] = []
            summary = container_cli.export_results(result, Path(directory), "cdr3", ["PanPep"], 1)
            self.assertIn("PanPep", summary["predictors_failed_or_unavailable"])

    def test_nonfinite_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.result()
            result["ranking_rows"][1]["final_score"] = float("nan")
            with self.assertRaises(ValueError):
                container_cli.export_results(result, Path(directory), "cdr3", ["PanPep"], 1)


class MappingTests(unittest.TestCase):
    columns = {"beta_col": "CDR3", "peptide_col": "Peptide"}

    def test_reordered_and_filtered_outputs(self):
        inputs = [{"CDR3": "CASSA", "Peptide": "PEPA"}, {"CDR3": "CASSB", "Peptide": "PEPB"}]
        outputs = [dict(inputs[1], score=0.9)]
        mapped = attach_ids(outputs, inputs, [10, 11], self.columns, "PanPep")
        self.assertEqual(mapped[0]["pair_id"], 11)

    def test_deduplicated_outputs(self):
        row = {"CDR3": "CASSA", "Peptide": "PEPA"}
        mapped = attach_ids([row], [row, row], [0, 1], self.columns, "PanPep")
        self.assertEqual([item["pair_id"] for item in mapped], [0, 1])

    def test_unmatched_output_fails(self):
        with self.assertRaises(ValueError):
            attach_ids([{"CDR3": "DIFFERENT", "Peptide": "PEPA"}],
                       [{"CDR3": "CASSA", "Peptide": "PEPA"}], [0], self.columns, "PanPep")

    def test_atmtcr_column_aliases(self):
        mapped = attach_ids([{"CDR3": "CASSA", "Peptide": "PEPA"}],
                            [{"TCR": "CASSA", "Epitope": "PEPA"}], [4], self.columns, "ATM-TCR")
        self.assertEqual(mapped[0]["pair_id"], 4)


if __name__ == "__main__":
    unittest.main()
