import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# Paths
IMAGE_PATH = "data/00000000.jpg"
YOLO_LABEL_PATH = "data/labels/00000000.txt"  # if YOLO labels exist
OUTPUT_DIR = "outputs/crops"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load image
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load SAM model
sam_checkpoint = "models/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# Function to convert YOLO format to pixel bbox
def yolo_to_bbox(label, img_w, img_h):
    cls, x, y, w, h = map(float, label.split())
    x1 = int((x - w/2) * img_w)
    y1 = int((y - h/2) * img_h)
    x2 = int((x + w/2) * img_w)
    y2 = int((y + h/2) * img_h)
    return [x1, y1, x2, y2]

# Read YOLO labels
h, w, _ = image.shape

with open(YOLO_LABEL_PATH, "r") as f:
    labels = f.readlines()

for i, label in enumerate(labels):
    box = yolo_to_bbox(label, w, h)
    input_box = np.array(box)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    mask = masks[0]

    # Apply mask
    segmented = image.copy()
    segmented[~mask] = 0

    # Crop bounding box
    x1, y1, x2, y2 = box
    crop = segmented[y1:y2, x1:x2]

    # Save
    cv2.imwrite(f"{OUTPUT_DIR}/object_{i}.png", crop)

print("Segmentation and cropping done!")
