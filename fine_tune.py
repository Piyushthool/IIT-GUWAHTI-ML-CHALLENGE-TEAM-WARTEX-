import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, 
    DataCollatorForLanguageModeling, 
    Trainer, TrainingArguments
)
from datasets import load_dataset
from pathlib import Path

# Base model (small, ~124M params)
BASE_MODEL = "gpt2"  # Or "microsoft/DialoGPT-small" for dialogue-style

def fine_tune():
    # Load dataset
    dataset = load_dataset('json', data_files={'train': 'data/train_dataset.jsonl'})
    
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token  # Fix for GPT-2
    
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
    
    # Tokenize function
    def tokenize_function(examples):
        # Concat prompt + response for causal LM
        texts = [prompt + " " + response for prompt, response in zip(examples['prompt'], examples['response'])]
        return tokenizer(texts, truncation=True, padding='max_length', max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['prompt', 'response'])
    tokenized_dataset.set_format('torch')
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # Causal LM, not masked
    )
    
    # Training args (small setup for laptop)
    training_args = TrainingArguments(
        output_dir='models/fine_tuned_model',
        overwrite_output_dir=True,
        num_train_epochs=3,  # Adjust based on data size
        per_device_train_batch_size=4,  # Small for CPU/GPU memory
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=5e-5,
        warmup_steps=100,
        logging_steps=100,
        dataloader_num_workers=0,  # For stability
        fp16=torch.cuda.is_available(),  # Use if GPU
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset['train'],
    )
    
    # Train
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained('models/fine_tuned_model')
    
    print("Fine-tuning complete! Model saved to models/fine_tuned_model")

if __name__ == "__main__":
    fine_tune()