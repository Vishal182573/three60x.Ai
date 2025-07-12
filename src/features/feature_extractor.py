import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import cv2
from typing import List

class FeatureExtractor:
    """A class to extract appearance features using a pre-trained ResNet-50 model. it maybe changed to a different model later."""
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # loading pre-trained ResNet-50 and removal of the final classification layer

        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        self.model = torch.nn.Sequential(*(list(model.children())[:-1])).to(self.device)
        self.model.eval()

        self.preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, frame: np.ndarray, bboxes: List[np.ndarray]) -> List[np.ndarray]:
        """extracts feature vectors for each bounding box in a frame"""
        if not bboxes:
            return []

        features = []
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cropped_img = pil_frame.crop((x1, y1, x2, y2))
            
            if cropped_img.size[0] == 0 or cropped_img.size[1] == 0:
                continue

            input_tensor = self.preprocess(cropped_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feature_vector = self.model(input_tensor)
                features.append(feature_vector.squeeze().cpu().numpy())
                
        return features