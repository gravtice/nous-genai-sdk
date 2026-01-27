import unittest


class TestCapabilityRules(unittest.TestCase):
    def test_openai_adapter_modalities_match_series_rules(self) -> None:
        from nous.genai.providers.openai import OpenAIAdapter

        adapter = OpenAIAdapter(api_key="__demo__")

        self.assertEqual(adapter.capabilities("deepseek-v3").input_modalities, {"text"})
        self.assertEqual(adapter.capabilities("kimi-k2").input_modalities, {"text"})
        self.assertEqual(adapter.capabilities("qvq-72b").input_modalities, {"text"})

        self.assertEqual(adapter.capabilities("kimi-latest").input_modalities, {"text", "image"})

        self.assertEqual(adapter.capabilities("qwen2.5-vl-72b-instruct").input_modalities, {"text", "image"})
        self.assertEqual(adapter.capabilities("qwen2.5-72b-instruct").input_modalities, {"text"})

        cap_asr = adapter.capabilities("qwen-asr")
        self.assertEqual(cap_asr.input_modalities, {"text", "audio"})
        self.assertEqual(cap_asr.output_modalities, {"text"})

        cap_z_image = adapter.capabilities("z-image-1")
        self.assertEqual(cap_z_image.input_modalities, {"text", "image"})
        self.assertEqual(cap_z_image.output_modalities, {"image"})

    def test_gemini_adapter_modalities_match_series_rules(self) -> None:
        from nous.genai.providers.gemini import GeminiAdapter

        adapter = GeminiAdapter(api_key="__demo__", base_url="https://example.invalid", provider_name="google")

        cap_imagen = adapter.capabilities("imagen-4.0-generate-001")
        self.assertEqual(cap_imagen.input_modalities, {"text"})
        self.assertEqual(cap_imagen.output_modalities, {"image"})

        cap_gemini_image = adapter.capabilities("gemini-2.5-flash-image")
        self.assertEqual(cap_gemini_image.input_modalities, {"text", "image"})
        self.assertEqual(cap_gemini_image.output_modalities, {"image"})

        cap_native_audio = adapter.capabilities("gemini-2.5-flash-native-audio-latest")
        self.assertEqual(cap_native_audio.input_modalities, {"text", "audio", "video"})
        self.assertEqual(cap_native_audio.output_modalities, {"text", "audio"})

        cap_veo = adapter.capabilities("veo-3.0-generate")
        self.assertEqual(cap_veo.input_modalities, {"text"})
        self.assertEqual(cap_veo.output_modalities, {"video"})

        cap_tts = adapter.capabilities("gemini-2.5-pro-preview-tts")
        self.assertEqual(cap_tts.input_modalities, {"text"})
        self.assertEqual(cap_tts.output_modalities, {"audio"})
