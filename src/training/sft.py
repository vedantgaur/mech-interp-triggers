from transformers import AdamW
from torch.cuda.amp import autocast, GradScaler
import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        prompt, _, output = self.dataset[idx]
        full_text = prompt + " " + output
        encoding = self.tokenizer(full_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  #fix
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

from torch.cuda.amp import autocast, GradScaler
import torch

def supervised_fine_tuning(model, tokenizer, dataset, num_epochs=3, batch_size=4, learning_rate=5e-5, accumulation_steps=4, max_length=512):
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    if hasattr(model.config, 'max_position_embeddings'):
        max_length = min(model.config.max_position_embeddings, max_length)

    custom_dataset = CustomDataset(dataset, tokenizer, max_length)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    loss_history = []

    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(dataloader):
            model = model.to(device)
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or i == len(dataloader) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        average_loss = total_loss / len(dataloader)
        loss_history.append(average_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss/len(dataloader)}")

    return model, loss_history