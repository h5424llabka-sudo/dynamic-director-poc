import cv2
import json
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def calculate_score(frame, bbox, config):
    """
    Simulate the scoring logic that will be implemented in Kotlin/OpenCV on Android.
    """
    target_comp = config.get("target_composition", {})
    
    h, w, _ = frame.shape
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    
    target_cx = target_comp.get("subject_center_x", 0.5)
    target_cy = target_comp.get("subject_center_y", 0.5)
    
    # 1. Composition Score (Distance between current center and target center)
    dist = np.sqrt((cx - target_cx)**2 + (cy - target_cy)**2)
    # Convert distance to a score 0-100 (where dist=0 is 100 score)
    composition_score = max(0, 100 - (dist * 200))
    
    # 2. Lighting Score (Simulated HSV calculation)
    # In real app: extract HSV of bbox vs background.
    # Here we just mock it.
    lighting_score = 90.0 # Mocked high score for simulation
    
    # Final combined score
    final_score = (composition_score * 0.7) + (lighting_score * 0.3)
    return final_score

def main():
    print("Simulating scoring logic...")
    config_path = "../app/src/main/assets/config.json"
    try:
        config = load_config(config_path)
        print("Config loaded successfully.")
    except FileNotFoundError:
        print(f"Config not found at {config_path}")
        return

    # Mock frame (e.g. 640x480 resolution)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Mock BBox representing a detected person [x1, y1, x2, y2]
    # Let's put the person exactly at x=0.66 (target_composition) -> x_center = 0.66 * 640 = 422
    # y=0.50 -> y_center = 240
    bbox = [372, 140, 472, 340]
    
    score = calculate_score(frame, bbox, config)
    print(f"Calculated Match Score: {score:.2f}%")

if __name__ == "__main__":
    main()
