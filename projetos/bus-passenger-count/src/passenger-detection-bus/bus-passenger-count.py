import os
from ultralytics import YOLO
from typing import Dict, Any

def train_bus_passenger_model(data_config_path: str, epochs: int = 300) -> None:
    f"""
    Trains the YOLOv11 medium model for bus passenger detection.
    Implements specific data augmentations tailored for a bus environment.

    Args:
        data_config_path (str): The absolute or relative path to the dataset.yaml file.
        epochs (int): The total number of training epochs. Default is 300.
    """
    if not os.path.exists(data_config_path):
        print(f"Error: Dataset configuration file not found at {data_config_path}")
        return

    # Initialize the YOLOv11 Medium model (as selected in your Roboflow screenshot)
    model: YOLO = YOLO("yolo11m.pt")

    print(f"Starting GPU-accelerated training for {epochs} epochs...")
    
   # Train the model with mapped augmentation hyperparameters
    results: Dict[str, Any] = model.train(
        data=data_config_path,
        epochs=epochs,
        imgsz=640,
        
        # Hardware Optimization (to RTX 3060)
        device=0,           
        batch=16,           
        workers=4,          
        amp=True,           
        
        # Color and Lighting Augmentations (Brightness, Exposure, Hue, Saturation)
        hsv_h=0.015,        # 1.5% Hue shift
        hsv_s=0.25,         # 25% Saturation shift
        hsv_v=0.15,         # 15% Value (Brightness/Exposure) shift
        
        # Spatial Augmentations
        degrees=10.0,       # Random rotation +/- 10 degrees
        translate=0.1,      # Slight translation to help with cropping
        scale=0.2,          # 20% Crop/Zoom
        shear=0.0,          # Disabled (0 degrees)
        fliplr=0.5,         # 50% probability of horizontal flip
        flipud=0.0,         # Disabled vertical flip
        
        # Advanced Augmentations
        mosaic=1.0,         # 100% probability to use mosaic
        erasing=0.1,        # 10% probability of random cutout
        
        # Note on Blur/Noise: YOLOv8/v11 native training uses base blur implicitly 
        # in some transformations, but aggressive motion blur or grain requires
        # advanced Albumentations pipelines. The settings above cover the essentials.
    )
    
    print("Training process completed successfully.")
    print(f"Results saved to: {results.save_dir}")

if __name__ == "__main__":
    dataset_yaml: str = os.path.abspath("../../data/processed/passenger-detection-bus/data.yaml")
    train_bus_passenger_model(data_config_path=dataset_yaml, epochs=300)