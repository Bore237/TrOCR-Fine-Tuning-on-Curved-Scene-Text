import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def load_model(model_name="microsoft/trocr-base-handwritten"):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return processor, model, device
