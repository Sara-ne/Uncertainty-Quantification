from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


class EntailmentChecker:
    """
    Checks if one text entails another using DeBERTa.
    This helps determine semantic similarity between image descriptions.
    """

    def __init__(self, model_id="microsoft/deberta-large-mnli"):
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)

    def check_entailment(self, text1, text2):
        """
        Computes an entailment score between two texts.
        :param text1: First text description.
        :param text2: Second text description.
        :return: Entailment score (higher = more similar).
        """
        inputs = self.tokenizer(text1, text2, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        entailment_score = logits.softmax(dim=-1)[0][2].item()  # Higher = more entailment (class 2 is entailment)
        return entailment_score