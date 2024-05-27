import io
import requests
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import OwlViTProcessor, OwlViTModel, BertModel, BertTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VLMManager:
    def __init__(self):
        # Initialize the models and processor
        self.owl_vit_processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch32')
        self.owl_vit_model = OwlViTModel.from_pretrained('google/owlvit-base-patch32')
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Initialize the vision-language model
        self.vlm = VisionLanguageModel(self.owl_vit_model, self.text_model)

        # Define the augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def identify(self, image: bytes, caption: str) -> list:
        # Convert bytes to PIL image
        pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        original_width, original_height = pil_image.size
        image = np.array(pil_image)

        # Apply augmentations
        augmented = self.augmentation_pipeline(image=image)
        processed_image = augmented['image'].unsqueeze(0)  # Add batch dimension

        # Tokenize the caption
        tokenized_caption = self.tokenizer(caption, return_tensors="pt")

        # Forward pass through the model
        bbox_preds, match_scores = self.vlm(processed_image, tokenized_caption)

        # Select the bounding box with the highest matching score
        best_box_idx = torch.argmax(match_scores).item()
        best_bbox = bbox_preds[0, best_box_idx].tolist()  # Assuming single image batch

        # Convert (x1, y1, x2, y2) to (x1, y1, width, height)
        x1, y1, x2, y2 = best_bbox
        width = x2 - x1
        height = y2 - y1

        # Scale bounding box coordinates back to the original image dimensions
        x1 = int(x1 * (original_width / 224))
        y1 = int(y1 * (original_height / 224))
        width = int(width * (original_width / 224))
        height = int(height * (original_height / 224))

        return [x1, y1, width, height]

class VisionLanguageModel(nn.Module):
    def __init__(self, owl_vit_model, text_model):
        super(VisionLanguageModel, self).__init__()
        self.owl_vit_model = owl_vit_model
        self.text_model = text_model
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.bbox_regressor = nn.Linear(768, 4)  # 4 coordinates for the bounding box
        self.classifier = nn.Linear(768, 1)  # Matching score

    def forward(self, images, captions):
        # Extract visual features using OWL-ViT
        vision_outputs = self.owl_vit_model(pixel_values=images, output_hidden_states=True)
        visual_features = vision_outputs.hidden_states[-1]  # Last hidden state

        # Extract textual features using BERT
        text_outputs = self.text_model(**captions)
        text_features = text_outputs.last_hidden_state

        # Cross-attention mechanism
        attn_output, _ = self.cross_attention(text_features, visual_features, visual_features)

        # Bounding box regression and classification
        bbox_predictions = self.bbox_regressor(attn_output)
        matching_scores = self.classifier(attn_output)

        return bbox_predictions, matching_scores

# def load_image_from_url(url: str):
#     response = requests.get(url)
#     return response.content

def visualize_bbox(image_bytes, bbox):
    image = Image.open(io.BytesIO(image_bytes))
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Create a Rectangle patch
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()

# def main():
#     # Initialize the VLMManager
#     vlm_manager = VLMManager()

#     # Load an image (example image URL)
#     image_url = "https://example.com/path/to/your/image.jpg"
#     image_data = load_image_from_url(image_url)

#     # Define a caption
#     caption = "A cat sitting on a sofa"

#     # Get the bounding box
#     bounding_box = vlm_manager.identify(image_data, caption)

#     # Print the bounding box
#     print("Bounding Box:", bounding_box)

#     # Visualize the bounding box
#     visualize_bbox(image_data, bounding_box)

if __name__ == "__main__":
    main()
