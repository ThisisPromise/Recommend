import os
import requests
from duckduckgo_search import DDGS
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Product list
products = [
    "ALARM CLOCK BAKELIKE RED",
    "CHOCOLATE HOT WATER BOTTLE",
    "SPOTTY BUNTING",
    "LUNCH BAG WOODLAND",
    "REX CASH+CARRY JUMBO SHOPPER",
    "JUMBO STORAGE BAG SUKI",
    "REGENCY CAKESTAND 3 TIER",
    "6 RIBBONS RUSTIC CHARM",
    "RETROSPOT TEA SET CERAMIC 11 PC"
]

# Config
images_per_product = 50
save_path = "product_images"

def download_images(query, limit=50):
    folder = os.path.join(save_path, query.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)

    downloaded = 0
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=limit*2)
        for result in tqdm(results, desc=f"Downloading: {query}"):
            try:
                url = result["image"]
                response = requests.get(url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image.save(os.path.join(folder, f"{downloaded+1}.jpg"))
                downloaded += 1
            except Exception:
                continue
            if downloaded >= limit:
                break
    print(f"✔ Downloaded {downloaded} images for '{query}'\n")

def main():
    os.makedirs(save_path, exist_ok=True)
    for product in products:
        download_images(product, images_per_product)
    print("✅ All images downloaded successfully.")

if __name__ == "__main__":
    main()
