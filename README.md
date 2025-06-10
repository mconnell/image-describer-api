# Image Describer API

Basic Python-based API that takes an image URL and generates a description using the [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base]) pretrained vision-language model.

## 🚀 How to Run

### Build and run the Docker Image for the first time.

```bash
docker compose up --build
```

### Test the API

Access via a browser or use `curl`:

```bash
curl "http://localhost:8000/describe?url=https://upload.wikimedia.org/wikipedia/commons/a/af/Keanu_Reeves_%2825448963336%29_%28cropped%29.jpg"

{
  "url":"https://upload.wikimedia.org/wikipedia/commons/a/af/Keanu_Reeves_(25448963336)_(cropped).jpg",
  "hash":"f4937c480771e589df2694627444aca29482b1ca6a1ccefa58c63a718780eac3",
  "description":"a man with long hair"
}
```
