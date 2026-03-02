import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def load_model(model_name="microsoft/trocr-small-printed", max_length=128, num_beams = 2, checkpoint_path=None, verbose=False):

    # Load model + processor
    if checkpoint_path:
        model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)
        processor = TrOCRProcessor.from_pretrained(checkpoint_path, use_fast=True)
    else:
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        processor = TrOCRProcessor.from_pretrained(model_name, use_fast=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Align special tokens
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.bos_token_id = processor.tokenizer.bos_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id

    # Resize decoder embeddings if needed
    if model.config.decoder.vocab_size != processor.tokenizer.vocab_size:
        model.decoder.resize_token_embeddings(processor.tokenizer.vocab_size)
        model.config.decoder.vocab_size = processor.tokenizer.vocab_size
        model.config.vocab_size = processor.tokenizer.vocab_size

    # Generation config
    model.generation_config.max_length = max_length
    model.generation_config.num_beams = num_beams
    model.generation_config.early_stopping = True # Stoppe la génération si EOS est produit pour tous les beams.
    #model.generation_config.no_repeat_ngram_size = 3

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Params: {total_params:,} | Trainable: {trainable_params:,}")

    return processor, model, device