import torch
from torch.utils.data import DataLoader
from src import SCUTDataset
import yaml
from src import load_model
from src.train_utils import wapper_compute_metrics, LrLoggerCallback, collate_fn
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


def train_trocr(config_path, trainer_config, metrics_fn, train_callbacks = None, trocr_transforms = None):

    # -----------------------------
    # Load config path
    # -----------------------------
    with open(config_path, "r") as f: 
        configs = yaml.safe_load(f)
        
    # -----------------------------
    # Load model + processor
    # -----------------------------
    processor, model, device = load_model(model_name = configs["model_name"], 
                                            max_length=configs["max_length_pred"], 
                                            num_beams = configs["num_beams"],
                                            verbose = configs["show_number_parameter"],
                                            checkpoint_path = configs["checkpoint_path"]
                                        )
    # Geler tout l'encodeur
    if configs["freeze_encoder"]:
        for param in model.encoder.parameters():
            param.requires_grad = False

    # -----------------------------
    # Load datasets
    # -----------------------------
    train_dataset = SCUTDataset(
        root_dir=configs["image_train_dir"],    
        ann_path= configs["ann_train_dir"],
        processor= processor,
        max_target_length = configs["max_length_pred"],  
        num_sample = configs["num_sample"],
        train_transforms = trocr_transforms
    )
    
    val_dataset = SCUTDataset(
        root_dir=configs["image_val_dir"],     
        ann_path= configs["ann_val_dir"],
        processor= processor,
        max_target_length = configs["max_length_pred"],
        num_sample = configs["num_sample"],
        train_transforms = None
    )

    # -----------------------------
    # Metric, loss and optimizers
    # -----------------------------
    metrics_fn = metrics_fn(processor)

    generation_config = GenerationConfig(
        max_length=configs["max_length_pred"],
        num_beams=configs["num_beams"],
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        #no_repeat_ngram_size=0,   
    )

    trainer_config.generation_config = generation_config

    # -----------------------------
    # Training Loop
    # -----------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=trainer_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
        callbacks=train_callbacks,
    )

    return trainer, model, train_dataset

if __name__ == "__main__":
    config_path = ""
    train_callbacks = [LrLoggerCallback()]

    training_args = Seq2SeqTrainingArguments(
        output_dir="./trocr_finetuned",
        logging_dir="./logs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        num_train_epochs= 10,
        fp16=True,
        predict_with_generate=True,
        metric_for_best_model="cer",
        greater_is_better=False, 
        save_total_limit=1,
    )

    trainers, models, train_dataset = train_trocr(config_path = config_path, 
                                                trocr_transforms = None, 
                                                trainer_config = training_args, 
                                                train_callbacks = train_callbacks, 
                                                metrics_fn = wapper_compute_metrics
                                            )
