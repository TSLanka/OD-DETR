import os
import torch
import PIL
import datasets
import transformers
import matplotlib
import tkinter

def check_directory_exists(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} directory not found at {path}")

def setup_environment():
    # Check and print library versions
    print(f"torch is installed, version: {torch.__version__}")
    print(f"PIL is installed, version: {PIL.__version__}")
    print(f"datasets is installed, version: {datasets.__version__}")
    print(f"transformers is installed, version: {transformers.__version__}")
    print(f"matplotlib is installed, version: {matplotlib.__version__}")
    print(f"tkinter is installed, version: {tkinter.Tcl().eval('info patchlevel')}")

    # Check dataset directories
    base_dir = 'Datasets'
    check_directory_exists(os.path.join(base_dir, 'images_rgb_train'), 'images_rgb_train')
    check_directory_exists(os.path.join(base_dir, 'images_rgb_val'), 'images_rgb_val')
    check_directory_exists(os.path.join(base_dir, 'images_thermal_train'), 'images_thermal_train')
    check_directory_exists(os.path.join(base_dir, 'images_thermal_val'), 'images_thermal_val')
    check_directory_exists(os.path.join(base_dir, 'video_rgb_test'), 'video_rgb_test')
    check_directory_exists(os.path.join(base_dir, 'video_thermal_test'), 'video_thermal_test')

    return {
        'train_images_dir': os.path.join(base_dir, 'images_rgb_train', 'data'),
        'train_annotations_file': os.path.join(base_dir, 'images_rgb_train', 'coco.json'),
        'val_images_dir': os.path.join(base_dir, 'images_rgb_val', 'data'),
        'val_annotations_file': os.path.join(base_dir, 'images_rgb_val', 'coco.json'),
        'test_images_dir': os.path.join(base_dir, 'video_rgb_test', 'data'),
        'test_annotations_file': os.path.join(base_dir, 'video_rgb_test', 'coco.json')
    }

if __name__ == "__main__":
    setup_environment()
