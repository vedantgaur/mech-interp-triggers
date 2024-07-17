from dataclasses import dataclass
from typing import Dict, List, Sequence
import torch
from torch.utils.data import Dataset
import transformers

IGNORE_INDEX = -100

class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self._tokenize_fn(self.dataset[idx])

    def _tokenize_fn(self, messages: List[Dict]) -> Dict:
        inputs, labels = [], []
        
        for turn, message in enumerate(messages):
            tokenized = self.tokenizer.apply_chat_template(
                [message],
                return_tensors="pt",
                padding=False,
                truncation=True,
            )[0]
            
            if turn > 0:  # skip bos_token
                tokenized = tokenized[1:]
            
            inputs.append(tokenized)
            
            # mask user input (turn % 2 == 0) with IGNORE_INDEX
            if turn % 2 == 0:
                masked_labels = torch.full(tokenized.shape, IGNORE_INDEX, dtype=torch.long)
                labels.append(masked_labels)
            else:
                labels.append(tokenized.clone())
        
        input_ids = torch.cat(inputs)
        labels = torch.cat(labels)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }

@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

def supervised_fine_tuning(model, tokenizer, dataset, num_epochs=3, batch_size=4, learning_rate=5e-5, accumulation_steps=4):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    custom_dataset = CustomDataset(dataset, tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

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
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss}")

    return model, loss_history