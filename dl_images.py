import os
import requests
from zipfile import ZipFile

# URLs to download the dataset (you might need to adjust if the source changes)
image_dataset_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
text_dataset_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

# Directory to save the dataset
save_dir = "flickr8k_data"

# Function to download a file from a URL
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            file.write(data)
    print(f"Downloaded {save_path} ({total_size // block_size} KB)")

# Function to extract a zip file
def extract_zip(zip_path, extract_to):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path}")

# Create save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Paths for saving downloaded files
image_zip_path = os.path.join(save_dir, "Flickr8k_Dataset.zip")
text_zip_path = os.path.join(save_dir, "Flickr8k_text.zip")

# Download the image dataset
print("Downloading Flickr8k image dataset...")
download_file(image_dataset_url, image_zip_path)

# Download the text dataset (captions)
print("Downloading Flickr8k text dataset...")
download_file(text_dataset_url, text_zip_path)

# Extract the datasets
print("Extracting Flickr8k image dataset...")
extract_zip(image_zip_path, save_dir)

print("Extracting Flickr8k text dataset...")
extract_zip(text_zip_path, save_dir)

print("Flickr8k dataset downloaded and extracted successfully!")
