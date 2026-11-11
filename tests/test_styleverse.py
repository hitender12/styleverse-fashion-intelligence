"""Tests for Styleverse. Run: python -m unittest discover -s tests -v"""

import io
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from styleverse.attributes import extract_attributes
from styleverse.catalog import generate_catalog, load_catalog
from styleverse.evaluate import run_evaluation
from styleverse.explain import explain, matched_attributes
from styleverse.features import embed
from styleverse.feedback import SessionFeedback
from styleverse.privacy import Session, ingest_upload, strip_exif
from styleverse.rerank import Constraints, rerank
from styleverse.search import VectorIndex

ROOT = Path(__file__).resolve().parent.parent


class CatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="styleverse-test-"))
        cls.products = generate_catalog(48, out_dir=cls.tmp / "catalog")
        cls.index = VectorIndex(cls.products)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_retrieval_finds_visually_closest(self):
        """Querying with a catalog image returns itself top-1, and same
        colour-family/category items dominate the top 5."""
        q = self.products[0]
        results = self.index.query_image(
            Image.open(q.image_path), k=5, exclude_ids=set())
        self.assertEqual(results[0].product.id, q.id)
        self.assertGreater(results[0].score, 0.95)
        results = self.index.query_image(
            Image.open(q.image_path), k=5, exclude_ids={q.id})
        same_family = [r for r in results
                       if r.product.color_family == q.color_family]
        self.assertGreaterEqual(len(same_family), 3)

    def test_budget_filter_never_violated(self):
        q = self.products[5]
        results = self.index.query_image(Image.open(q.image_path), k=10)
        for budget in (30.0, 50.0, 80.0):
            ranked = rerank(results, Constraints(max_price=budget))
            self.assertTrue(all(r.product.price <= budget for r in ranked))
            self.assertTrue(len(ranked) >= 0)

    def test_category_and_color_filters(self):
        q = self.products[5]
        results = self.index.query_image(Image.open(q.image_path), k=10)
        ranked = rerank(results, Constraints(category="dress"))
        self.assertTrue(all(r.product.category == "dress" for r in ranked))
        ranked = rerank(results, Constraints(color_family="blue"))
        self.assertTrue(all(r.product.color_family == "blue"
                            for r in ranked))

    def test_explanations_name_real_matched_attributes(self):
        q = next(p for p in self.products if p.category == "dress")
        attrs = extract_attributes(Image.open(q.image_path))
        results = self.index.query_image(
            Image.open(q.image_path), k=5, exclude_ids={q.id})
        for r in results:
            text = explain(attrs, r.product, r.score)
            matched = matched_attributes(attrs, r.product)
            if matched:
                for m in matched:
                    # The named attribute value must actually agree.
                    key = m.split(" (")[0]
                    self.assertIn(key, text)
            else:
                self.assertIn("visually similar overall", text)

    def test_feedback_changes_ranking(self):
        q = self.products[10]
        results = self.index.query_image(Image.open(q.image_path), k=10)
        ranked = rerank(results, Constraints())
        top_id = ranked[0].product.id
        fb = SessionFeedback()
        fb.mark_not_relevant(top_id)
        adjusted = fb.apply(ranked)
        self.assertNotEqual(adjusted[0].product.id, top_id)
        self.assertNotIn(top_id, [r.product.id for r in adjusted[:1]])
        # Relevant feedback boosts an item up the ranking.
        fb2 = SessionFeedback()
        low = ranked[-1].product.id
        fb2.mark_relevant(low)
        adjusted2 = fb2.apply(ranked)
        pos_before = [r.product.id for r in ranked].index(low)
        pos_after = [r.product.id for r in adjusted2].index(low)
        self.assertLess(pos_after, pos_before)
        # Hide removes the item entirely.
        fb3 = SessionFeedback()
        fb3.hide(top_id)
        self.assertNotIn(top_id, [r.product.id for r in fb3.apply(ranked)])

    def test_attribute_extraction_reasonable(self):
        """Rule-based extraction gets the majority of catalog items right
        on colour family (the most load-bearing attribute)."""
        correct = 0
        for p in self.products:
            attrs = extract_attributes(Image.open(p.image_path))
            if attrs["color_family"] == p.color_family:
                correct += 1
        self.assertGreaterEqual(correct / len(self.products), 0.8)

    def test_evaluation_metrics_are_sane(self):
        metrics = run_evaluation(self.products)
        self.assertGreater(metrics.recall_at_10, metrics.recall_at_5 - 1e-9)
        self.assertGreater(metrics.recall_at_5, 0.0)
        self.assertGreaterEqual(metrics.precision_at_10, 0.0)
        self.assertLessEqual(metrics.precision_at_10, 1.0)
        self.assertGreater(metrics.mean_latency_ms, 0.0)


class PrivacyTest(unittest.TestCase):
    def test_exif_stripped_on_ingest(self):
        """An image carrying EXIF (camera make, GPS) comes back clean."""
        img = Image.new("RGB", (64, 64), (200, 30, 30))
        exif = Image.Exif()
        exif[271] = "TestCamera"          # Make
        exif[306] = "2026:11:11 10:00:00"  # DateTime
        raw = io.BytesIO()
        img.save(raw, format="JPEG", exif=exif)
        raw_bytes = raw.getvalue()
        self.assertTrue(len(raw_bytes) > 0)
        # Sanity: the source really carries EXIF.
        self.assertEqual(Image.open(io.BytesIO(raw_bytes)).getexif()[271],
                         "TestCamera")
        clean = strip_exif(raw_bytes)
        self.assertEqual(dict(Image.open(io.BytesIO(clean)).getexif()), {})

    def test_upload_validation_and_session_expiry(self):
        tmp = Path(tempfile.mkdtemp(prefix="styleverse-session-"))
        try:
            img = Image.new("RGB", (32, 32), (10, 100, 200))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            path = ingest_upload(buf.getvalue(), tmp / "s1")
            self.assertTrue(path.exists())
            with self.assertRaises(ValueError):
                ingest_upload(b"", tmp / "s1")
            with self.assertRaises(ValueError):
                ingest_upload(b"x" * (11 * 1024 * 1024), tmp / "s1")
            session = Session("abc", base_dir=tmp, ttl_seconds=0)
            self.assertTrue(session.expired())
            session.delete()
            self.assertFalse(session.dir.exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
