#!/usr/bin/env python3
import os
import torch
import transformers
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from petals import AutoDistributedModelForCausalLM

def main():
    # Training hyperparameters
    MODEL_NAME = "bigscience/bloom-7b1-petals"  # You can also use other models like meta-llama/Llama-2-7b-hf
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 3
    MAX_LENGTH = 128
    NUM_PREFIX_TOKENS = 8  # Number of trainable prefix tokens for prompt tuning
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model {MODEL_NAME}")
    
    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Initialize distributed model
        model = AutoDistributedModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,  # Use fp16 for efficiency
            tuning_mode='ptune',  # Use prompt tuning
            pre_seq_len=NUM_PREFIX_TOKENS  # Number of trainable prefix tokens
        )
        
        print("Model loaded successfully")

        # Load a sample dataset
        print("Loading dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        
        def tokenize_function(examples):
            # Process text and return tokenized results
            tokenized = tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors=None  # Return lists for now, will be converted to tensors by collator
            )
            return tokenized

        # Prepare the dataset
        print("Preparing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # We're doing causal language modeling, not masked
        )

        # Create DataLoader with the collator
        train_dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=data_collator
        )

        # Print the number of trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}")

        # Prepare optimizer - only optimize the prompt parameters
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE
        )

        # Training loop
        print("Starting training...")
        model.train()
        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            num_batches = 0
            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(DEVICE)
                    labels = batch['labels'].to(DEVICE)
                    
                    # Forward pass
                    outputs = model(
                        input_ids,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx % 10 == 0:
                        print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {str(e)}")
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            print(f"Epoch {epoch} complete, Average Loss: {avg_loss:.4f}")

        # Save the trained prompts
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/trained_model.pt")
        print("Training complete! Model saved to checkpoints/trained_model.pt")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()