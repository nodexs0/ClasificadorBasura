import os
import json

# Rutas de los datasets
trashnet_dir = "Datasets/TrashNet/dataset-resized"
taco_dir = "/Datasets/taco/data"

# Mapeo de categorías comunes
category_map = {
    "cardboard": "Paper",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "trash": "Other",
    "food": "Organic",
    "other": "Other"
}

# Contar imágenes en TrashNet
def count_trashnet():
    category_counts = {category: 0 for category in category_map.values()}
    for category in os.listdir(trashnet_dir):
        if category in category_map:
            category_path = os.path.join(trashnet_dir, category)
            category_counts[category_map[category]] += len(os.listdir(category_path))
    print("TrashNet:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    return category_counts

# Contar imágenes en TACO
def count_taco():
    annotations_path = os.path.join(taco_dir, "annotations.json")
    taco_annotations = json.load(open(annotations_path))

    # Crear un mapa de image_id a category_id
    image_to_category = {}
    for annotation in taco_annotations["annotations"]:
        image_to_category[annotation["image_id"]] = annotation["category_id"]
    
    # Crear un mapa de category_id a supercategory
    category_map_taco = {cat["id"]: cat["supercategory"] for cat in taco_annotations["categories"]}
    
    # Contar imágenes por categoría
    category_counts = {category: 0 for category in category_map.values()}
    for image_id, category_id in image_to_category.items():
        category_name = category_map_taco.get(category_id)
        if category_name in category_map:
            category_counts[category_map[category_name]] += 1
    
    print("TACO:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    return category_counts

# Combinar los conteos
def combine_counts(counts1, counts2):
    combined_counts = {category: counts1.get(category, 0) + counts2.get(category, 0) for category in category_map.values()}
    print("Dataset combinado:")
    for category, count in combined_counts.items():
        print(f"  {category}: {count}")
    return combined_counts

# Ejecutar el análisis
trashnet_counts = count_trashnet()
taco_counts = count_taco()
combined_counts = combine_counts(trashnet_counts, taco_counts)
