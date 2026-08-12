import os
import modal

# ==============================================================================
# Modal Image Definition
# ==============================================================================
# Read the Hugging Face token from your local environment during deployment
hf_token = os.environ.get("HF_TOKEN", "")

# Modal will upload your local directory context and build the Dockerfile
# directly in their cloud builders. No local Docker daemon is required!
image = modal.Image.from_dockerfile(
    "Dockerfile",
    build_args={"HF_TOKEN": hf_token}
)

# ==============================================================================
# Modal App Setup
# ==============================================================================
app = modal.App("vggt-1b-manhattan-server")

# ==============================================================================
# FastAPI ASGI Deployment
# ==============================================================================
@app.function(
    image=image,
    gpu="L4",              # Request a GPU (L4 is recommended for cost-efficiency)
    timeout=1200,          # Set timeout to 20 minutes (1200 seconds)
    secrets=[
        modal.Secret.from_name("cloudflare-r2"), # Injects R2 environment variables
    ]
)
@modal.asgi_app()
def fastapi_app():
    # We import the FastAPI app here so it executes inside the container on the GPU
    from backend_api_extended_manhattan import app as fastapi_backend
    import backend_api_extended_manhattan

    # Run warm-up model loading on container start
    print("--> Warming up VGGT model on GPU...", flush=True)
    fastapi_backend.state.model = backend_api_extended_manhattan.get_model()
    print("--> Model warmed up successfully!", flush=True)

    return fastapi_backend
