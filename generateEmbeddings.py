import torch
import open_clip
from PIL import Image
from torchvision import transforms


class CLIPEmbeddingExtractor:
    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def get_image_embeddings(self, images):
        image_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_image(image_tensors)
        return embeddings.cpu().numpy()

    def get_image_probability(self, image, candidate_labels=None):
        if candidate_labels is None:
            candidate_labels = [
                "a photo of a cat", "a photo of a dog", "a photo of a car",
                "a photo of a person", "a photo of a tree"
            ]
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        tokenized = self.tokenizer(candidate_labels).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(tokenized)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T
            probs = logits.softmax(dim=-1).squeeze().cpu().numpy()
        return probs

    def get_clip_similarity_scores(self, prompt, images):
        image_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        text_tokens = self.tokenizer([prompt]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors)
            text_features = self.model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T
            scores = logits.squeeze().cpu().numpy()
        return scores
