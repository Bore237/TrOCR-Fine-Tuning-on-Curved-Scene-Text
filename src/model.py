import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def load_model(model_name="microsoft/trocr-small-printed", checkpoint_path=None, verbose=False):
    # 1. Load from checkpoint if available, else load base
    if checkpoint_path:
        # Safer: loads config + weights + handles sharding automatically
        model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)
        processor = TrOCRProcessor.from_pretrained(checkpoint_path, use_fast=True)
    else:
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        processor = TrOCRProcessor.from_pretrained(model_name, use_fast=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 2. Critical Config for Seq2SeqTrainer
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id # Added this
    
    # Set decoder_start_token_id
    if processor.tokenizer.cls_token_id is not None:
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    else:
        model.config.decoder_start_token_id = processor.tokenizer.bos_token_id

    # 3. Essential for compute_metrics (predict_with_generate)
    # This ensures the model uses the right settings during eval
    model.config.vocab_size = model.config.decoder.vocab_size

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Params: {total_params:,} | Trainable: {trainable_params:,}")

    return processor, model, device