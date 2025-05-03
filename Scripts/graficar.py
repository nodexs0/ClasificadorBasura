import os
import matplotlib.pyplot as plt

def count_images_in_directories(base_dirs):
    class_counts = {}

    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            # Exclude the root directory itself
            if root == base_dir:
                continue

            class_name = os.path.basename(root)
            class_counts[class_name] = class_counts.get(class_name, 0) + len([file for file in files if file.endswith(('png', 'jpg', 'jpeg'))])

    return class_counts

def plot_pie_chart(class_counts, title, output_path):
    plt.figure(figsize=(8, 8))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    # Specify the directories for training, validation, and testing datasets
    train_dir = "Datasets/combined_dataset/train"
    val_dir = "Datasets/combined_dataset/val"
    test_dir = "Datasets/combined_dataset/test"

    # Count images across all directories
    total_counts = count_images_in_directories([train_dir, val_dir, test_dir])

    # Calculate the total number of images
    total_images = sum(total_counts.values())
    print(f"Total number of images: {total_images}")
    print("Total Class Distribution:", total_counts)

    # Plot and save the total distribution
    plot_pie_chart(total_counts, "Total Dataset Class Distribution", "total_dataset_distribution.png")
