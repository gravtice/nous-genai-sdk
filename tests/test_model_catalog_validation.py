import unittest


class TestModelCatalogValidation(unittest.TestCase):
    def test_no_duplicate_model_ids_within_provider(self) -> None:
        from nous.genai.reference.model_catalog import MODEL_CATALOG

        for provider, model_ids in MODEL_CATALOG.items():
            self.assertIsInstance(model_ids, list)
            seen: set[str] = set()
            for model_id in model_ids:
                self.assertIsInstance(model_id, str)
                self.assertTrue(model_id)
                key = f"{provider}:{model_id}"
                self.assertNotIn(key, seen, f"duplicate model in catalog: {key}")
                seen.add(key)
