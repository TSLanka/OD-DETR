import os
from datasets import load_from_disk, DatasetDict
from environment_setup import setup_environment
from dataset_loading import load_datasets
from data_preprocessing import preprocess_data
from model_operations import load_model, train_model, evaluate_model, test_model
from transformers import AutoImageProcessor

CACHE_DIR = './cache'

def save_dataset(dataset, name):
    dataset_path = os.path.join(CACHE_DIR, name)
    dataset.save_to_disk(dataset_path)
    print(f"Dataset '{name}' saved to {dataset_path}.")

def load_cached_dataset(name):
    dataset_path = os.path.join(CACHE_DIR, name)
    if os.path.exists(dataset_path):
        print(f"Loading cached dataset '{name}' from {dataset_path}...")
        return load_from_disk(dataset_path)
    else:
        print(f"Cached dataset '{name}' not found.")
        return None

def main():
    print("Starting main execution...")

    # Setup environment
    print("Setting up environment...")
    paths = setup_environment()
    print("Environment setup completed.")
    
    # Load or preprocess datasets
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    print("Initializing processor...")
    processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
    print("Processor initialized.")

    # Load datasets
    print("Loading or preprocessing datasets...")

    train_dataset = load_cached_dataset('train')
    val_dataset = load_cached_dataset('validation')
    test_dataset = load_cached_dataset('test')

    if not train_dataset or not val_dataset or not test_dataset:
        print("Cached datasets not found or incomplete. Preprocessing datasets...")
        datasets = load_datasets(paths)

        train_dataset = datasets['train'].map(lambda x: preprocess_data(x, processor), batched=True, remove_columns=datasets['train'].column_names)
        val_dataset = datasets['validation'].map(lambda x: preprocess_data(x, processor), batched=True, remove_columns=datasets['validation'].column_names)
        test_dataset = datasets['test'].map(lambda x: preprocess_data(x, processor), batched=True, remove_columns=datasets['test'].column_names)

        save_dataset(train_dataset, 'train')
        save_dataset(val_dataset, 'validation')
        save_dataset(test_dataset, 'test')

    print("Datasets ready.")

    # Load model
    print("Loading model...")
    model = load_model()
    print("Model loaded successfully.")

    # Train model
    print("Starting model training...")
    train_model(train_dataset, val_dataset, model)
    print("Model training completed.")

    # Evaluate model
    print("Evaluating model...")
    evaluate_model(val_dataset, model)
    print("Model evaluation completed.")

    # Test model
    image_path = input("Enter the path of the image to test: ")
    print("Testing model with the provided image...")
    test_model(image_path, processor, model)
    print("Model test completed.")

if __name__ == "__main__":
    main()
