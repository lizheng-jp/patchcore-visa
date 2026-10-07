import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iad.metrics import evaluate, select_threshold, thresholded_metrics  # noqa: E402
from iad.splits import make_splits  # noqa: E402


def fake_rows():
    rows = []
    for cat in ("pcb1", "candle"):
        for i in range(20):
            rows.append({"object": cat, "split": "train", "label": "normal", "image": f"{cat}/n{i}.JPG", "mask": ""})
        for i in range(10):
            rows.append({"object": cat, "split": "test", "label": "normal", "image": f"{cat}/tn{i}.JPG", "mask": ""})
        for i in range(7):
            rows.append(
                {"object": cat, "split": "test", "label": "anomaly", "image": f"{cat}/a{i}.JPG", "mask": f"{cat}/a{i}.png"}
            )
    return rows


class SplitTests(unittest.TestCase):
    def test_val_and_test_are_disjoint_and_cover_official_test(self):
        samples = make_splits(fake_rows(), seed=0)
        val = {s.image for s in samples if s.split == "val"}
        test = {s.image for s in samples if s.split == "test"}
        self.assertFalse(val & test)
        self.assertEqual(len(val | test), 2 * 17)

    def test_each_category_has_both_labels_in_val_and_test(self):
        samples = make_splits(fake_rows(), seed=0)
        for cat in ("pcb1", "candle"):
            for split in ("val", "test"):
                labels = {s.label for s in samples if s.category == cat and s.split == split}
                self.assertEqual(labels, {0, 1})

    def test_train_is_normal_only_and_split_is_deterministic(self):
        a = make_splits(fake_rows(), seed=3)
        b = make_splits(list(reversed(fake_rows())), seed=3)
        self.assertEqual(sorted(a, key=lambda s: s.image), sorted(b, key=lambda s: s.image))
        self.assertTrue(all(s.label == 0 for s in a if s.split == "train"))

    def test_anomaly_in_train_is_rejected(self):
        rows = fake_rows() + [{"object": "pcb1", "split": "train", "label": "anomaly", "image": "x", "mask": "m"}]
        with self.assertRaises(ValueError):
            make_splits(rows)


class MetricTests(unittest.TestCase):
    def test_threshold_comes_from_val_only(self):
        val_y = np.array([0, 0, 0, 1, 1, 1])
        val_s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        # The test set would prefer a much lower threshold; it must not be used.
        test_y = np.array([0, 0, 1, 1])
        test_s = np.array([0.01, 0.02, 0.05, 0.06])
        result = evaluate(val_y, val_s, test_y, test_s)
        self.assertAlmostEqual(result["test"]["threshold"], select_threshold(val_y, val_s))
        self.assertEqual(result["test"]["anomaly_recall"], 0.0)
        self.assertEqual(result["image_auroc_test"], 1.0)

    def test_confusion_uses_label_1_as_anomaly(self):
        m = thresholded_metrics(np.array([0, 1, 1]), np.array([0.9, 0.9, 0.1]), 0.5)
        self.assertEqual(m["confusion"], {"tn": 0, "fp": 1, "fn": 1, "tp": 1})
        self.assertEqual(m["anomaly_recall"], 0.5)
        self.assertEqual(m["normal_specificity"], 0.0)


if __name__ == "__main__":
    unittest.main()
