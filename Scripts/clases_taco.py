import json

# Ruta al archivo de anotaciones JSON
annotations_path = '../Datasets/taco/data/annotations.json'

# Cargar el archivo JSON
with open(annotations_path, 'r') as f:
    data = json.load(f)

# Extraer las categorías
categories = data['categories']

# Imprimir las clases
for category in categories:
    print(f"ID: {category['id']}, Clase: {category['name']}")
