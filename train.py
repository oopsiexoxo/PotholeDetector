from ultralytics import YOLO

def train_model():
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    # data argument should point to a YAML file describing the dataset
    # epochs is the number of training cycles
    # imgsz is the image size
    print("Starting training...")
    
    # Train the model
    results = model.train(data="dataset/data.yaml", epochs=25, imgsz=640)
    
    print("Training completed. The model is saved in 'runs/detect/train/weights/best.pt'")

if __name__ == "__main__":
    train_model()
