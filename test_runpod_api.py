import os
import zipfile
import requests
import gdown

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# 1. Google Drive File ID (extracted from your link)
GDRIVE_FILE_ID = "1Wwmg1uurWJP2KfILFb5SffjubVGW-saJ"
ZIP_FILENAME = "gdrive_images.zip"
EXTRACT_DIR = "images_to_test"

# 2. RunPod Serverless Web Server Configuration
# Replace with your actual RunPod API Key
RUNPOD_API_KEY = "YOUR_RUNPOD_API_KEY"
# Your RunPod Endpoint ID (from your screenshot: 919mru12jk13rl)
ENDPOINT_ID = "919mru12jk13rl"
# The internal port of your FastAPI app (from your Dockerfile: 8000)
INTERNAL_PORT = "8000"

# Endpoint URL for RunPod Web Server / Proxy mode
API_URL = f"https://{ENDPOINT_ID}-{INTERNAL_PORT}.proxy.runpod.net/predict_project_batch"

# ==============================================================================
# STEP 1: Download from Google Drive
# ==============================================================================
print("⏳ Downloading file from Google Drive...")
url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
gdown.download(url, ZIP_FILENAME, quiet=False)

# ==============================================================================
# STEP 2: Extract ZIP file
# ==============================================================================
if not os.path.exists(ZIP_FILENAME):
    print("❌ Error: Download failed.")
    exit(1)

print(f"📦 Extracting {ZIP_FILENAME} to '{EXTRACT_DIR}'...")
os.makedirs(EXTRACT_DIR, exist_ok=True)
with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

# Find all image paths
image_extensions = (".jpg", ".jpeg", ".png", ".webp")
image_paths = []
for root, _, files in os.walk(EXTRACT_DIR):
    for file in files:
        if file.lower().endswith(image_extensions):
            image_paths.append(os.path.join(root, file))

print(f"📸 Found {len(image_paths)} images to send to the API.")
if not image_paths:
    print("❌ Error: No images found in the extracted directory.")
    exit(1)

# ==============================================================================
# STEP 3: Send POST request to RunPod API
# ==============================================================================
print(f"🚀 Sending request to RunPod API: {API_URL}")

headers = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}"
}

# Prepare files for multipart/form-data upload
files_payload = []
open_files = []
try:
    for idx, path in enumerate(image_paths):
        # Open each file and append to payload
        f = open(path, "rb")
        open_files.append(f)
        files_payload.append(("files", (os.path.basename(path), f, "image/jpeg")))

    # Form parameters
    data_payload = {
        "batch_id": "test_gdrive_batch",
        "conf_threshold": "1.0",
        "max_points": "3000000",
        "clean_voxel": "0.012",
        "clean_stat_neighbors": "16",
        "clean_stat_std": "2.8"
    }

    response = requests.post(API_URL, headers=headers, files=files_payload, data=data_payload)
    
    print(f"Status Code: {response.status_code}")
    try:
        print("Response JSON:")
        print(response.json())
    except Exception:
        print("Response Text:")
        print(response.text)

finally:
    # Always close opened files
    for f in open_files:
        f.close()
