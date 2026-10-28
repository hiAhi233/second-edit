import copy
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
from paclip.data.patient_manifest import audit_rows, choose_support, resolve_support, assign_patient_splits
from paclip.diagnostics.novelty_fixtures import fixture_rows


class PatientManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rows = fixture_rows(self.tmp.name)

    def test_audit_support_counts(self):
        self.assertTrue(audit_rows(self.rows)["valid"])
        support = choose_support(self.rows)
        selected = resolve_support(self.rows, support)
        self.assertEqual([sum(r["label"] == k for r in selected) for k in (0, 1)], [2, 3])
        self.assertEqual(len({r["patient_id"] for r in selected}), 5)

    def test_cross_sequence_patient_overlap_rejected(self):
        self.rows[-1]["patient_id"] = self.rows[0]["patient_id"]
        self.rows[-1]["sequence"] = "different"
        self.assertFalse(audit_rows(self.rows)["valid"])

    def test_missing_identity_not_certified(self):
        self.rows[0]["patient_verified"] = False
        self.assertFalse(audit_rows(self.rows)["valid"])

    def test_reencoded_duplicate_pixels_rejected(self):
        destination = Path(self.tmp.name) / "copy.bmp"
        Image.open(self.rows[0]["image_path"]).save(destination)
        self.rows[-2]["image_path"] = str(destination)
        self.assertTrue(any("duplicate image pixels" in e for e in audit_rows(self.rows)["errors"]))

    def test_content_changed_invalidates_support(self):
        support = choose_support(self.rows)
        Image.fromarray(np.full((40, 48), 20, np.uint8)).save(self.rows[0]["image_path"])
        with self.assertRaisesRegex(ValueError, "content changed"):
            resolve_support(self.rows, support)

    def test_support_cannot_include_validation(self):
        support = choose_support(self.rows)
        support["sample_ids"][0] = self.rows[-2]["sample_id"]
        from paclip.data.fewshot import stable_digest
        support["support_digest"] = stable_digest(support["sample_ids"])
        with self.assertRaisesRegex(ValueError, "Only training"):
            resolve_support(self.rows, support)

    def test_unknown_disease_blocks_heldout(self):
        self.rows[1]["disease_labels"] = ["unknown"]
        with self.assertRaisesRegex(ValueError, "verified disease"):
            assign_patient_splits(self.rows, heldout_diseases=["lesion"])

    def test_support_does_not_count_copied_pixels_as_extra_slice(self):
        Image.open(self.rows[0]["image_path"]).save(self.rows[2]["image_path"])
        support = choose_support(self.rows)
        selected = resolve_support(self.rows, support)
        self.assertEqual(len({r["image_pixel_sha256"] for r in selected}), 5)

    def test_patient_split_deterministic_and_grouped(self):
        # Each patient has both slice classes, so all split class requirements are satisfiable.
        rows = fixture_rows(self.tmp.name, (20, 0, 0))
        for i, r in enumerate(rows):
            r["patient_id"] = f"paired-patient-{i//2}"
        first = assign_patient_splits(copy.deepcopy(rows))
        second = assign_patient_splits(copy.deepcopy(rows))
        self.assertEqual([r["split"] for r in first], [r["split"] for r in second])
        self.assertTrue(audit_rows(first)["valid"])
