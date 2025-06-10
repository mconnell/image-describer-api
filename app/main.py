from fastapi import FastAPI, Query
from pydantic import HttpUrl
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import requests
import io
import hashlib

app = FastAPI()

# Load model components
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def _download_image(url: str) -> bytes:
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise ValueError(f"URL did not return an image (Content-Type: {content_type})")

    return response.content

def _compute_image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()

def _describe_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")

    inputs = processor(images=image, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, max_length=20, num_beams=5)
    return processor.decode(output_ids[0], skip_special_tokens=True)

@app.get("/")
def root():
    return {"message": "Image Describer API"}

@app.get("/describe")
def describe(url: HttpUrl = Query(..., description="URL of the image")):
    try:
        image_bytes = _download_image(url)
        image_hash = _compute_image_hash(image_bytes)
        description = _describe_image(image_bytes)

        return {
            "url": str(url),
            "hash": image_hash,
            "description": description
        }
    except Exception as e:
        return {"error": str(e)}
