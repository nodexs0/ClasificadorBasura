import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_image(input_path, output_path, target_size=(224, 224)):
    """
    Procesa una sola imagen: la redimensiona y la guarda en la carpeta de salida.

    Args:
        input_path (str): Ruta de la imagen de entrada.
        output_path (str): Ruta donde se guardará la imagen procesada.
        target_size (tuple): Tamaño objetivo como (ancho, alto).
    """
    try:
        with Image.open(input_path) as img:
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img_resized.save(output_path)
            return f"Procesada: {output_path}"
    except Exception as e:
        return f"Error procesando {input_path}: {e}"

def normalize_images_parallel(input_folder, output_folder, target_size=(224, 224), max_workers=8):
    """
    Normaliza las imágenes en paralelo redimensionándolas y guardándolas en una nueva carpeta.

    Args:
        input_folder (str): Ruta de la carpeta de entrada con imágenes.
        output_folder (str): Ruta de la carpeta de salida.
        target_size (tuple): Tamaño objetivo como (ancho, alto).
        max_workers (int): Número máximo de hilos para ejecutar en paralelo.
    """
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    input_path = os.path.join(root, file)
                    relative_path = os.path.relpath(input_path, input_folder)
                    output_path = os.path.join(output_folder, relative_path)
                    tasks.append(executor.submit(process_image, input_path, output_path, target_size))
        
        for future in as_completed(tasks):
            print(future.result())

def main():
    input_folder = "..\Datasets\combined_dataset"  # Ruta de la carpeta base con las imágenes originales
    output_folder = "..\Datasets\combined_dataset_normalizada"  # Carpeta para guardar las imágenes normalizadas
    subsets = ["train", "test", "val"]  # Subcarpetas esperadas
    max_workers = 8  # Número de hilos para paralelismo

    for subset in subsets:
        input_path = os.path.join(input_folder, subset)
        output_path = os.path.join(output_folder, subset)
        if os.path.exists(input_path):
            print(f"Normalizando imágenes en la carpeta {subset}...")
            normalize_images_parallel(input_path, output_path, max_workers=max_workers)
        else:
            print(f"La carpeta {subset} no existe.")

if __name__ == "__main__":
    main()
