import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

def get_image_sizes(folder_path):
    """
    Recorre las carpetas y subcarpetas dentro de una ruta específica,
    y obtiene los tamaños (ancho, alto) de todas las imágenes.

    Args:
        folder_path (str): Ruta de la carpeta raíz.

    Returns:
        list: Lista de tamaños de las imágenes como tuplas (ancho, alto).
    """
    image_sizes = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        image_sizes.append(img.size)  # (ancho, alto)
                except Exception as e:
                    print(f"No se pudo procesar la imagen {file_path}: {e}")
    return image_sizes

def plot_image_sizes(image_sizes, title):
    """
    Genera un gráfico de los tamaños de las imágenes.

    Args:
        image_sizes (list): Lista de tamaños de las imágenes como tuplas (ancho, alto).
        title (str): Título de la gráfica.
    """
    # Separar anchos y altos
    widths = [size[0] for size in image_sizes]
    heights = [size[1] for size in image_sizes]
    
    # Crear gráfica
    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.6)
    plt.title(f"Tamaños de imágenes - {title}")
    plt.xlabel("Ancho (pixels)")
    plt.ylabel("Alto (pixels)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def main():
    base_folder = "..\Datasets\combined_dataset"
    subsets = ["train", "test", "val"]

    for subset in subsets:
        folder_path = os.path.join(base_folder, subset)
        if os.path.exists(folder_path):
            print(f"Procesando la carpeta {subset}...")
            image_sizes = get_image_sizes(folder_path)
            plot_image_sizes(image_sizes, title=subset)
        else:
            print(f"La carpeta {subset} no existe.")

if __name__ == "__main__":
    main()
