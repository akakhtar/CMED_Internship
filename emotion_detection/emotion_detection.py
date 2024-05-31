import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoConfig
from PIL import Image
from .config import DEVICE

# Load the emotion detection model and feature extractor
extractor = AutoFeatureExtractor.from_pretrained("trpakov/vit-face-expression")
model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")
id2label = AutoConfig.from_pretrained("trpakov/vit-face-expression").id2label


def detect_emotions(image, box):
    face = image.crop(box)
    inputs = extractor(images=face, return_tensors="pt")
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probabilities = probabilities.detach().numpy().tolist()[0]
    class_probabilities = {id2label[i]: prob for i, prob in enumerate(probabilities)}
    return face, class_probabilities
