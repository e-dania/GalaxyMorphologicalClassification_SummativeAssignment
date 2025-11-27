import os
import zipfile
import subprocess

DATA_DIR = "./data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

def download_dataset():
    print("🔽 Downloading Galaxy Zoo dataset using Kaggle CLI...")

    # Download training images
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "galaxy-zoo-the-galaxy-challenge",
        "-p", DATA_DIR
    ], check=True)

    print("✅ Download complete!")

def extract_zip(file_path, extract_to):
    print(f"📦 Extracting {file_path} ...")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("✅ Extraction done.\n")

def extract_all():
    for file in os.listdir(DATA_DIR):
        if file.endswith(".zip"):
            extract_zip(os.path.join(DATA_DIR, file), DATA_DIR)

def main():
    print("=== Galaxy Zoo Data Acquisition Script ===")
    download_dataset()
    extract_all()
    print("🎉 All data downloaded and extracted into data/raw!")

if __name__ == "__main__":
    main()
