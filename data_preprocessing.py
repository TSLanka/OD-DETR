from PIL import Image
from transformers import AutoImageProcessor

def preprocess_data(examples, processor):
    print("Starting image loading and conversion...")
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    print("Images loaded and converted to RGB.")

    print("Processing annotations...")
    annotations = examples['annotations']
    
    targets = []
    for anno in annotations:
        target = {}
        target['boxes'] = [obj['bbox'] for obj in anno]
        target['labels'] = [obj['category_id'] for obj in anno]
        targets.append(target)
    print("Annotations processed.")

    print("Applying processor to images and annotations...")
    inputs = processor(images=images, annotations=targets, return_tensors="pt")
    inputs['labels'] = targets
    print("Processor applied successfully.")

    return inputs

if __name__ == "__main__":
    from environment_setup import setup_environment
    from dataset_loading import load_datasets
    from transformers import AutoImageProcessor

    print("Setting up environment...")
    paths = setup_environment()
    print("Environment setup completed.")

    print("Loading datasets...")
    datasets = load_datasets(paths)
    print("Datasets loaded.")

    print("Initializing processor...")
    processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
    print("Processor initialized.")

    print("Preprocessing train dataset...")
    train_dataset = datasets['train'].map(lambda x: preprocess_data(x, processor), batched=True, remove_columns=datasets['train'].column_names)
    print("Train dataset preprocessed.")

    print("Preprocessing validation dataset...")
    val_dataset = datasets['validation'].map(lambda x: preprocess_data(x, processor), batched=True, remove_columns=datasets['validation'].column_names)
    print("Validation dataset preprocessed.")

    print("Preprocessing test dataset...")
    test_dataset = datasets['test'].map(lambda x: preprocess_data(x, processor), batched=True, remove_columns=datasets['test'].column_names)
    print("Test dataset preprocessed.")

    print("Data preprocessing completed.")
