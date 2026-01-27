import unittest
from unittest.mock import patch


class TestTuziModels(unittest.TestCase):
    def test_tuzi_catalog_has_no_duplicates(self) -> None:
        from nous.genai.reference.catalog import MODEL_CATALOG

        for provider in ("tuzi-web", "tuzi-openai", "tuzi-google", "tuzi-anthropic"):
            model_ids = MODEL_CATALOG[provider]
            with self.subTest(provider=provider):
                self.assertEqual(len(model_ids), len(set(model_ids)))

    def test_tuzi_web_routes_by_model_id_prefix(self) -> None:
        from nous.genai.providers import TuziAdapter

        openai = object()
        gemini = object()
        anthropic = object()
        tuzi = TuziAdapter(openai=openai, gemini=gemini, anthropic=anthropic)

        self.assertIs(tuzi._route("gpt-4o-mini"), openai)
        self.assertIs(tuzi._route("sora-2"), openai)
        self.assertIs(tuzi._route("claude-sonnet-4-5"), anthropic)
        self.assertIs(tuzi._route("gemini-2.5-flash"), gemini)
        self.assertIs(tuzi._route("gemma-3-4b-it"), gemini)
        self.assertIs(tuzi._route("veo-3.0-generate"), gemini)
        self.assertIs(tuzi._route("text-embedding-004"), gemini)

    def test_tuzi_web_route_requires_protocol_keys(self) -> None:
        from nous.genai import GenAIError
        from nous.genai.providers import TuziAdapter

        tuzi = TuziAdapter(openai=None, gemini=None, anthropic=None)

        with self.assertRaises(GenAIError) as cm:
            tuzi._route("gpt-4o-mini")
        self.assertEqual(cm.exception.info.type, "InvalidRequestError")

        with self.assertRaises(GenAIError) as cm:
            tuzi._route("gemini-2.5-flash")
        self.assertEqual(cm.exception.info.type, "InvalidRequestError")

        with self.assertRaises(GenAIError) as cm:
            tuzi._route("claude-3-sonnet-20240229")
        self.assertEqual(cm.exception.info.type, "InvalidRequestError")

    def test_tuzi_sora_seconds_sent_as_string(self) -> None:
        from nous.genai.types import GenerateRequest, Message, OutputSpec, OutputVideoSpec, Part
        from nous.genai.providers.openai import OpenAIAdapter
        from nous.genai._internal.http import multipart_form_data_fields as real_multipart_form_data_fields

        req = GenerateRequest(
            model="tuzi-openai:sora-2",
            input=[Message(role="user", content=[Part.from_text("test")])],
            output=OutputSpec(modalities=["video"], video=OutputVideoSpec(duration_sec=4, aspect_ratio="16:9")),
            wait=False,
        )

        with (
            patch("nous.genai.providers.openai.multipart_form_data_fields") as multipart_form_data_fields,
            patch("nous.genai.providers.openai.request_streaming_body_json") as request_streaming_body_json,
        ):
            multipart_form_data_fields.side_effect = real_multipart_form_data_fields
            request_streaming_body_json.return_value = {"id": "vid_123"}
            adapter = OpenAIAdapter(api_key="__demo__", base_url="https://example.invalid/v1", provider_name="tuzi-openai")
            out = adapter.generate(req, stream=False)
            self.assertEqual(out.status, "running")

            _, kwargs = multipart_form_data_fields.call_args
            fields = kwargs["fields"]
            self.assertIsInstance(fields.get("seconds"), str)
            self.assertEqual(fields.get("seconds"), "4")

        with patch("nous.genai.providers.openai.request_json") as request_json:
            request_json.return_value = {"id": "vid_123"}
            adapter = OpenAIAdapter(api_key="__demo__", base_url="https://example.invalid/v1", provider_name="openai")
            out = adapter.generate(req, stream=False)
            self.assertEqual(out.status, "running")

            _, kwargs = request_json.call_args
            body = kwargs["json_body"]
            self.assertIsInstance(body.get("seconds"), int)

    def test_openai_adapter_capabilities_infer_tuzi_modalities(self) -> None:
        from nous.genai.providers.openai import OpenAIAdapter

        adapter = OpenAIAdapter(api_key="__demo__", base_url="https://example.invalid/v1", provider_name="tuzi-openai")

        self.assertEqual(adapter.capabilities("sd3.5-large").output_modalities, {"image"})
        self.assertEqual(adapter.capabilities("pika-1.5").output_modalities, {"video"})
        self.assertEqual(adapter.capabilities("suno-v3").output_modalities, {"audio"})

        cap = adapter.capabilities("whisper-large-v3")
        self.assertEqual(cap.input_modalities, {"audio"})
        self.assertEqual(cap.output_modalities, {"text"})

    def test_openai_images_supports_tuzi_wrapped_response(self) -> None:
        from nous.genai.providers.openai import OpenAIAdapter
        from nous.genai.types import GenerateRequest, Message, OutputImageSpec, OutputSpec, Part

        adapter = OpenAIAdapter(api_key="__demo__", base_url="https://example.invalid/v1", provider_name="tuzi-web")
        req = GenerateRequest(
            model="tuzi-web:chat-seededit",
            input=[Message(role="user", content=[Part.from_text("hi")])],
            output=OutputSpec(modalities=["image"], image=OutputImageSpec(n=1)),
            wait=True,
        )

        with patch("nous.genai.providers.openai.request_json") as request_json:
            request_json.return_value = {"data": {"created": 1736160000, "images": [{"url": "https://example.invalid/a.png"}]}}
            resp = adapter.generate(req, stream=False)

        self.assertEqual(resp.status, "completed")
        parts = [p for m in resp.output for p in m.content]
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].type, "image")
        self.assertIsNotNone(parts[0].source)
        self.assertEqual(getattr(parts[0].source, "kind", None), "url")

    def test_gemini_adapter_capabilities_support_imagen_and_native_audio(self) -> None:
        from nous.genai.providers.gemini import GeminiAdapter

        adapter = GeminiAdapter(api_key="__demo__", base_url="https://example.invalid", provider_name="google")

        self.assertEqual(adapter.capabilities("imagen-4.0-generate-001").output_modalities, {"image"})
        self.assertIn("audio", adapter.capabilities("gemini-2.5-flash-native-audio-latest").output_modalities)

    def test_tuzi_openai_allows_non_sora_video_model_ids(self) -> None:
        from nous.genai.types import GenerateRequest, Message, OutputSpec, OutputVideoSpec, Part
        from nous.genai.providers.openai import OpenAIAdapter

        req = GenerateRequest(
            model="tuzi-openai:pika-1.5",
            input=[Message(role="user", content=[Part.from_text("test")])],
            output=OutputSpec(modalities=["video"], video=OutputVideoSpec(duration_sec=4, aspect_ratio="16:9")),
            wait=False,
        )

        with patch("nous.genai.providers.openai.request_streaming_body_json") as request_streaming_body_json:
            request_streaming_body_json.return_value = {"id": "vid_123"}
            adapter = OpenAIAdapter(api_key="__demo__", base_url="https://example.invalid/v1", provider_name="tuzi-openai")
            out = adapter.generate(req, stream=False)
            self.assertEqual(out.status, "running")

    def test_tuzi_deepsearch_capabilities(self) -> None:
        from nous.genai.providers import TuziAdapter

        openai = object()
        tuzi = TuziAdapter(openai=openai, gemini=None, anthropic=None)

        for model in (
            "gemini-2.5-pro-deepsearch",
            "gemini-2.5-pro-deepsearch-async",
            "gemini-2.5-flash-deepsearch",
            "gemini-3-pro-deepsearch",
        ):
            with self.subTest(model=model):
                cap = tuzi.capabilities(model)
                self.assertEqual(cap.input_modalities, {"text"})
                self.assertEqual(cap.output_modalities, {"text"})
                self.assertFalse(cap.supports_stream)
                self.assertTrue(cap.supports_job)

    def test_tuzi_deepsearch_rejects_stream(self) -> None:
        from nous.genai import GenAIError
        from nous.genai.providers import TuziAdapter
        from nous.genai.providers.openai import OpenAIAdapter
        from nous.genai.types import GenerateRequest, Message, OutputSpec, Part

        openai_adapter = OpenAIAdapter(api_key="__demo__", base_url="https://example.invalid/v1", provider_name="tuzi-web")
        tuzi = TuziAdapter(openai=openai_adapter, gemini=None, anthropic=None)

        req = GenerateRequest(
            model="tuzi-web:gemini-2.5-pro-deepsearch",
            input=[Message(role="user", content=[Part.from_text("test")])],
            output=OutputSpec(modalities=["text"]),
        )

        with self.assertRaises(GenAIError) as cm:
            tuzi.generate(req, stream=True)
        self.assertEqual(cm.exception.info.type, "InvalidRequestError")
        self.assertIn("streaming", str(cm.exception))

    def test_tuzi_deepsearch_submit_async_task(self) -> None:
        from nous.genai.providers import TuziAdapter
        from nous.genai.providers.openai import OpenAIAdapter
        from nous.genai.types import GenerateRequest, Message, OutputSpec, Part

        openai_adapter = OpenAIAdapter(api_key="test_key", base_url="https://api.tu-zi.com/v1", provider_name="tuzi-web")
        tuzi = TuziAdapter(openai=openai_adapter, gemini=None, anthropic=None)

        req = GenerateRequest(
            model="tuzi-web:gemini-2.5-pro-deepsearch",
            input=[Message(role="user", content=[Part.from_text("analyze market trends")])],
            output=OutputSpec(modalities=["text"]),
            wait=False,
        )

        with patch("nous.genai.providers.tuzi.request_json") as mock_request:
            mock_request.return_value = {
                "id": "task_abc123",
                "preview_url": "https://asyncdata.net/web/task_abc123",
                "source_url": "https://asyncdata.net/source/task_abc123",
            }
            out = tuzi.generate(req, stream=False)

            self.assertEqual(out.status, "running")
            self.assertIsNotNone(out.job)
            self.assertEqual(out.job.job_id, "task_abc123")
            self.assertEqual(out.job.poll_after_ms, 2_000)

            # Verify request was made to asyncdata.net with -async suffix
            call_args = mock_request.call_args
            self.assertIn("asyncdata.net/tran/", call_args.kwargs["url"])
            self.assertEqual(call_args.kwargs["json_body"]["model"], "gemini-2.5-pro-deepsearch-async")

    def test_tuzi_suno_wait_fetch_audio_supports_list_data(self) -> None:
        from nous.genai.providers import TuziAdapter
        from nous.genai.providers.openai import OpenAIAdapter

        openai_adapter = OpenAIAdapter(api_key="test_key", base_url="https://api.tu-zi.com/v1", provider_name="tuzi-web")
        tuzi = TuziAdapter(openai=openai_adapter, gemini=None, anthropic=None)

        with patch("nous.genai.providers.tuzi.TuziAdapter._suno_fetch") as suno_fetch:
            suno_fetch.return_value = {
                "status": "SUCCESS",
                "data": [
                    {"audio_url": "https://cdn1.suno.ai/abc.mp3"},
                    {"audio_url": "https://cdn1.suno.ai/def.mp3"},
                ],
            }
            resp = tuzi._suno_wait_fetch_audio(task_id="task_123", model_id="chirp-v4", timeout_ms=10_000, wait=True)

        self.assertEqual(resp.status, "completed")
        parts = [p for m in resp.output for p in m.content]
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].type, "audio")
        self.assertIsNotNone(parts[0].source)
        self.assertEqual(getattr(parts[0].source, "kind", None), "url")
        self.assertEqual(getattr(parts[0].source, "url", None), "https://cdn1.suno.ai/abc.mp3")
