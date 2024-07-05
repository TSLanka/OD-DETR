# model_operations.py
from transformers import DeformableDetrForObjectDetection, TrainingArguments, Trainer, AutoImageProcessor
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_model():
    print("Loading model...")
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")
    print("Model loaded successfully.")
    return model

def collate_fn(batch):
    print("Collating batch...")
    collated_batch = {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'labels': [item['labels'] for item in batch]
    }
    print("Batch collated successfully.")
    return collated_batch

def train_model(train_dataset, val_dataset, model):
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./deformable_detr_flir",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        learning_rate=1e-5,
        weight_decay=1e-4,
    )
    print("Training arguments set up successfully.")

    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    print("Trainer initialized.")

    print("Starting training...")
    trainer.train()
    print("Training completed.")

    print("Saving model...")
    model.save_pretrained("./deformable_detr_flir")
    print("Model saved successfully.")

def evaluate_model(val_dataset, model):
    print("Setting up evaluation arguments...")
    eval_args = TrainingArguments(
        output_dir="./deformable_detr_flir",
        per_device_eval_batch_size=4,
        logging_steps=100,
    )
    print("Evaluation arguments set up successfully.")

    print("Initializing trainer for evaluation...")
    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=collate_fn,
        eval_dataset=val_dataset,
    )
    print("Trainer for evaluation initialized.")

    print("Starting evaluation...")
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")

def load_image(image_path, image_type):
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path)
    if image_type == 'rgb':
        image = image.convert("RGB")
    print("Image loaded and converted if necessary.")
    return image

def plot_results(image, boxes, labels):
    print("Plotting results...")
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    print("Results plotted.")

def test_model(image_path, processor, model):
    print(f"Testing model with image {image_path}...")
    image_type = 'rgb' if 'rgb' in image_path.lower() else 'thermal'
    image = load_image(image_path, image_type)
    print("Processing image...")
    inputs = processor(images=image, return_tensors="pt")

    print("Predicting...")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    boxes = outputs.pred_boxes

    print("Processing predictions...")
    probas = logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.5
    probas = probas[keep]
    boxes = boxes[0, keep].cpu()

    img_w, img_h = image.size
    scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
    boxes = boxes * scale_fct

    plot_results(image, boxes, probas)
    print("Model test completed.")