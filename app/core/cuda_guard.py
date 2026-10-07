def assert_cuda_ready() -> None:
    """Fail closed when REQUIRE_CUDA is set and CUDA is missing.

    Must not import torch unless REQUIRE_CUDA is true (Mac/CI stay off
    the fail-closed path).
    """
    from app.core.config import settings

    if not settings.REQUIRE_CUDA:
        return None

    import onnxruntime as ort
    import torch

    torch_ok = bool(torch.cuda.is_available())
    providers = list(ort.get_available_providers())
    ort_ok = "CUDAExecutionProvider" in providers
    if torch_ok and ort_ok:
        return None
    raise RuntimeError(
        "REQUIRE_CUDA=true but CUDA is not available: "
        f"torch.cuda.is_available()={torch_ok}, ORT providers={providers}"
    )
