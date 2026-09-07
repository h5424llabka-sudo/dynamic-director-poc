import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    sys.exit(1)

def export_model():
    print("Loading YOLOv8n model...")
    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")
    
    print("Exporting model to TFLite (float16)...")
    # Export the model to TFLite format
    # float16 quantization helps with performance on mobile while keeping accuracy
    # Requires standard YOLOv8 environment
    model.export(format="tflite", half=True, int8=False)
    
    print("Export completed.")
    print("Please copy the generated .tflite file to the Android app's assets folder:")
    print("e.g. cp yolov8n_float16.tflite ../app/src/main/assets/")

if __name__ == "__main__":
    export_model()
