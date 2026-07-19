"""Local model runtime (owned by Member C).

Purpose: ModelManager lazy-loads an OpenVINO IR model and compiles it with
device priority NPU -> GPU -> CPU, catching unavailable-device errors and
falling through so the app still runs on CPU-only machines. No caller wires
this up yet - see README's hardware section: no teammate has converted a
model to OpenVINO IR yet, so this class is fixed to be safe to import and
instantiate, not yet exercised by the live pipeline.
"""

import logging

import openvino as ov


class ModelManager:
    """Loads and runs an OpenVINO model, with lazy loading and automatic
    device fallback (NPU -> GPU -> CPU)."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.core = ov.Core()
        self.compiled_model = None
        self.infer_request = None
        self.device = None
        self.logger = logging.getLogger("verity.openvino")

    def _load_model_lazy(self):
        if self.compiled_model is not None:
            return

        try:
            model = self.core.read_model(self.model_path)
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not read OpenVINO model at {self.model_path!r}: {exc}"
            ) from exc

        for device in ("NPU", "GPU", "CPU"):
            try:
                if device in self.core.available_devices:
                    self.compiled_model = self.core.compile_model(model, device_name=device)
                    self.infer_request = self.compiled_model.create_infer_request()
                    self.device = device
                    self.logger.info("compiled model on %s", device)
                    return
            except Exception as exc:
                self.logger.warning("failed to compile on %s: %s", device, exc)

        try:
            self.compiled_model = self.core.compile_model(model, device_name="CPU")
            self.infer_request = self.compiled_model.create_infer_request()
            self.device = "CPU"
            self.logger.info("compiled model on CPU fallback")
        except Exception as exc:
            self.logger.error("failed to compile model on any device: %s", exc)
            raise RuntimeError(f"Could not compile model on NPU, GPU, or CPU: {exc}") from exc

    def predict(self, input_data):
        """Runs inference, loading the model lazily on first call."""
        self._load_model_lazy()
        try:
            results = self.infer_request.infer(input_data)
            return {output.any_name: results[output] for output in self.compiled_model.outputs}
        except Exception as exc:
            self.logger.error("prediction failed: %s", exc)
            raise

    def unload(self):
        """Frees the compiled model from memory - keeps local inference bursty."""
        self.compiled_model = None
        self.infer_request = None
        self.device = None
        self.logger.info("model unloaded")
