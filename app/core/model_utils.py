import onnxruntime as ort

from app.core.config import settings
from app.core.cuda_guard import assert_cuda_ready


def _cuda_provider_tuple():
    return (
        "CUDAExecutionProvider",
        {
            "device_id": str(getattr(settings, "CUDA_DEVICE_ID", 0)),
            "gpu_mem_limit": str(
                getattr(settings, "ONNX_VRAM_LIMIT_BYTES", 2 * 1024 * 1024 * 1024)
            ),
        },
    )


def create_onnx_session_from_bytes(model_bytes: bytes) -> ort.InferenceSession:
    """Consolidated ONNX InferenceSession factory creating session from in-memory byte buffer."""
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = getattr(settings, "ONNX_INTRA_THREADS", 1)
    session_options.inter_op_num_threads = getattr(settings, "ONNX_INTER_THREADS", 1)
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    if settings.REQUIRE_CUDA:
        assert_cuda_ready()
        providers = [_cuda_provider_tuple(), "CPUExecutionProvider"]
        session = ort.InferenceSession(
            model_bytes, session_options, providers=providers
        )
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError(
                "REQUIRE_CUDA=true but CUDAExecutionProvider is not active "
                "in InferenceSession"
            )
        return session

    available_providers = ort.get_available_providers()
    providers = []
    if "CoreMLExecutionProvider" in available_providers:
        providers.append("CoreMLExecutionProvider")
    if "CUDAExecutionProvider" in available_providers:
        providers.append(_cuda_provider_tuple())
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(model_bytes, session_options, providers=providers)

