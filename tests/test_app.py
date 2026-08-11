import io
import unittest
from pathlib import Path
from unittest.mock import patch

from app import app
from leaf_disease_model import extract_features, get_remedy_tip


class AppTestCase(unittest.TestCase):
    def setUp(self) -> None:
        app.testing = True
        self.client = app.test_client()

    def test_bacterial_spot_gets_bacterial_remedy_not_generic_spot(self) -> None:
        tip = get_remedy_tip("Tomato___Bacterial_spot")
        self.assertIn("Sanitize pruning tools", tip)
        self.assertNotIn("disease-control spray", tip)

    def test_shows_user_friendly_error_when_prediction_fails(self) -> None:
        with patch("app.predict_disease", side_effect=RuntimeError("boom")):
            response = self.client.post(
                "/",
                data={"image": (io.BytesIO(b"fake-image-data"), "sample.jpg")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("Prediction failed", body)

    def test_extract_features_returns_rich_vector(self) -> None:
        sample_image = next(
            (
                path
                for path in Path("data").rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ),
            None,
        )
        if sample_image is None:
            self.skipTest("No sample image found in the workspace")

        features = extract_features(sample_image)
        self.assertGreaterEqual(len(features), 90)
        self.assertTrue((features >= 0).all())


if __name__ == "__main__":
    unittest.main()
