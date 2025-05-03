import os
import shutil
import random
import json
from pathlib import Path
from zipfile import ZipFile

# Define paths for datasets and the combined dataset
DATASETS = {
    "TrashNet": "dataset/trashnet",
    "TACO": "dataset/taco",
    "WastedClassification": "dataset/wasted",
    "OpenRecycle": "dataset/openrecycle",
}
COMBINED_DATASET = "dataset/combined_dataset"

# Define the mapping of categories to the unified schema
CATEGORY_MAPPING = {
    "TrashNet": {
        "paper": "Paper",
        "plastic": "Plastic",
        "glass": "Glass",
        "metal": "Metal",
        "cardboard": "Cardboard",
        "trash": "Trash",
    },
    "TACO": {
        "Aluminium foil": "Metal",
        "Battery": "Trash",
        "Aluminium blister pack": "Metal",
        "Carded blister pack": "Trash",
        "Other plastic bottle": "Plastic",
        "Clear plastic bottle": "Plastic",
        "Glass bottle": "Glass",
        "Plastic bottle cap": "Plastic",
        "Metal bottle cap": "Metal",
        "Broken glass": "Glass",
        "Food Can": "Metal",
        "Aerosol": "Metal",
        "Drink can": "Metal",
        "Toilet tube": "Cardboard",
        "Other carton": "Cardboard",
        "Egg carton": "Cardboard",
        "Drink carton": "Cardboard",
        "Corrugated carton": "Cardboard",
        "Meal carton": "Cardboard",
        "Pizza box": "Cardboard",
        "Paper cup": "Paper",
        "Disposable plastic cup": "Plastic",
        "Foam cup": "Plastic",
        "Glass cup": "Glass",
        "Other plastic cup": "Plastic",
        "Food waste": "Compost",
        "Glass jar": "Glass",
        "Plastic lid": "Plastic",
        "Metal lid": "Metal",
        "Other plastic": "Plastic",
        "Magazine paper": "Paper",
        "Tissues": "Paper",
        "Wrapping paper": "Paper",
        "Normal paper": "Paper",
        "Paper bag": "Paper",
        "Plastified paper bag": "Plastic",
        "Plastic film": "Plastic",
        "Six pack rings": "Plastic",
        "Garbage bag": "Plastic",
        "Other plastic wrapper": "Plastic",
        "Single-use carrier bag": "Plastic",
        "Polypropylene bag": "Plastic",
        "Crisp packet": "Plastic",
        "Spread tub": "Plastic",
        "Tupperware": "Plastic",
        "Disposable food container": "Plastic",
        "Foam food container": "Plastic",
        "Other plastic container": "Plastic",
        "Plastic gloves": "Plastic",
        "Plastic utensils": "Plastic",
        "Pop tab": "Metal",
        "Rope & strings": "Other",
        "Scrap metal": "Metal",
        "Shoe": "Trash",
        "Squeezable tube": "Plastic",
        "Plastic straw": "Plastic",
        "Paper straw": "Paper",
        "Styrofoam piece": "Plastic",
        "Unlabeled litter": "Trash",
        "Cigarette": "Trash",
    },
    "WastedClassification": {
        "Paper": "Paper",
        "Plastic": "Plastic",
        "Glass": "Glass",
        "Metal": "Metal",
        "Cardboard": "Cardboard",
        "Trash": "Trash",
        "Compost": "Compost",
    },
    "OpenRecycle": {
        "paper": "Paper",
        "plastic": "Plastic",
        "glass": "Glass",
        "metal": "Metal",
        "cardboard": "Cardboard",
        "trash": "Trash",
        "compost": "Compost",
    },
}

# Function to create directory structure
def create_dirs(categories, base_path):
    for category in categories:
        category_path = os.path.join(base_path, category)
        os.makedirs(category_path, exist_ok=True)

# Function to copy and map images from folder-based datasets
def copy_images_from_folders(dataset_name, dataset_path, mapping, combined_path):
    for src_category, dest_category in mapping.items():
        src_path = Path(dataset_path) / src_category
        dest_path = Path(combined_path) / dest_category

        if not src_path.exists():
            print(f"Warning: Source path {src_path} does not exist.")
            continue

        for image_file in src_path.glob("*.jpg"):
            dest_file = dest_path / image_file.name
            shutil.copy(image_file, dest_file)

# Function to extract and process TACO annotations
def process_taco_annotations(taco_path, mapping, combined_path):
    annotations_path = Path(taco_path) / "annotations.json"
    images_path = Path(taco_path) / "images"

    if not annotations_path.exists():
        print(f"Warning: TACO annotations file {annotations_path} not found.")
        return

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    for annotation in annotations["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]

        # Get image and category names
        image_info = next(img for img in annotations["images"] if img["id"] == image_id)
        category_name = next(cat["name"] for cat in annotations["categories"] if cat["id"] == category_id)

        if category_name not in mapping:
            continue

        dest_category = mapping[category_name]
        src_image_path = images_path / image_info["file_name"]
        dest_path = Path(combined_path) / dest_category

        if src_image_path.exists():
            shutil.copy(src_image_path, dest_path / src_image_path.name)

# Function to extract Wasted Classification zip files
def extract_wasted_files(wasted_path, combined_path, mapping):
    for zip_file in Path(wasted_path).glob("*.zip"):
        with ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall("temp_wasted")

        temp_path = Path("temp_wasted")
        for category, dest_category in mapping.items():
            src_path = temp_path / category
            dest_path = Path(combined_path) / dest_category

            if src_path.exists():
                for image_file in src_path.glob("*.jpg"):
                    shutil.copy(image_file, dest_path / image_file.name)

        shutil.rmtree(temp_path)

# Main script execution
def main():
    # Create the combined dataset structure
    categories = set()
    for mapping in CATEGORY_MAPPING.values():
        categories.update(mapping.values())

    create_dirs(categories, COMBINED_DATASET)

    # Process each dataset
    print("Processing TrashNet...")
    copy_images_from_folders("TrashNet", DATASETS["TrashNet"], CATEGORY_MAPPING["TrashNet"], COMBINED_DATASET)

    print("Processing TACO...")
    process_taco_annotations(DATASETS["TACO"], CATEGORY_MAPPING["TACO"], COMBINED_DATASET)

    print("Processing Wasted Classification...")
    extract_wasted_files(DATASETS["WastedClassification"], COMBINED_DATASET, CATEGORY_MAPPING["WastedClassification"])

    print("Processing OpenRecycle...")
    copy_images_from_folders("OpenRecycle", DATASETS["OpenRecycle"], CATEGORY_MAPPING["OpenRecycle"], COMBINED_DATASET)

    print("Datasets combined successfully!")

if __name__ == "__main__":
    main()