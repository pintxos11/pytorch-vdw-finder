import torch
from torchvision import models, transforms
from PIL import Image
import os
import shutil

# --- Load model ---
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# --- Define transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Folder containing the images to classify ---
image_folder = "images_to_classify"

# --- Create output subfolders ---
for class_id in [0, 1]:
    class_dir = os.path.join(image_folder, f"class_{class_id}")
    os.makedirs(class_dir, exist_ok=True)

# --- Classify and move images ---
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        path = os.path.join(image_folder, filename)
        image = Image.open(path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Move file to appropriate class folder
        target_dir = os.path.join(image_folder, f"class_{predicted_class}")
        shutil.move(path, os.path.join(target_dir, filename))
        print(f"{filename} → class_{predicted_class}")
