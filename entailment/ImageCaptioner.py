from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class ImageCaptioner:
    """
    Converts images into text descriptions using BLIP.
    These descriptions are needed for entailment-based clustering.
    """
    def __init__(self, model_id="Salesforce/blip-image-captioning-base"):
        self.device = "cpu"
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)

    def caption_image(self, image):
        """
        Generates a text caption for a given image.
        :param image: PIL image to describe.
        :return: Text description of the image.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            caption = self.model.generate(**inputs)
        return self.processor.decode(caption[0], skip_special_tokens=True)
