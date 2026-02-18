import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def load_model(model_name="microsoft/trocr-base-handwritten", verbose = False):
    # Load processor and model
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Ensure device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # --- Fix: set model config IDs ---
    # Use tokenizer's pad_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # Use tokenizer's cls_token_id (or bos_token_id if cls_token_id is None)
    if processor.tokenizer.cls_token_id is not None:
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    else:
        model.config.decoder_start_token_id = processor.tokenizer.bos_token_id

    # Total parameters and trainable parameters.
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")

    return processor, model, device
