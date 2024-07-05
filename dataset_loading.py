import json
import os
from datasets import Dataset

def load_flir_dataset(images_dir, annotations_file):
    print(f"Attempting to load annotations from {annotations_file}...")
    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        print(f"Loaded annotations from {annotations_file}")
    except FileNotFoundError:
        print(f"Failed to find the annotation file: {annotations_file}")
        raise FileNotFoundError(f"Annotation file {annotations_file} not found.")
    
    print(f"Preparing dataset from annotations...")
    data = {
        'image_path': [os.path.join(images_dir, img['file_name']) for img in annotations['images']],
        'annotations': [
            [
                {
                    'bbox': ann['bbox'],
                    'category_id': ann['category_id']
                }
                for ann in annotations['annotations']
                if ann['image_id'] == img['id']
            ]
            for img in annotations['images']
        ]
    }

    print(f"Dataset prepared with {len(data['image_path'])} images.")
    return Dataset.from_dict(data)

def load_datasets(paths):
    print("Loading training dataset...")
    train_dataset = load_flir_dataset(paths['train_images_dir'], paths['train_annotations_file'])
    print("Training dataset loaded successfully.")

    print("Loading validation dataset...")
    val_dataset = load_flir_dataset(paths['val_images_dir'], paths['val_annotations_file'])
    print("Validation dataset loaded successfully.")

    print("Loading test dataset...")
    test_dataset = load_flir_dataset(paths['test_images_dir'], paths['test_annotations_file'])
    print("Test dataset loaded successfully.")

    return {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    }

if __name__ == "__main__":
    from environment_setup import setup_environment
    print("Setting up environment for dataset loading...")
    paths = setup_environment()
    print("Environment setup completed. Proceeding to load datasets...")
    datasets = load_datasets(paths)
    print("All datasets loaded successfully.")
