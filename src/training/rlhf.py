import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from tqdm import tqdm

class TriggerRLHFDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        inputs = self.tokenizer.encode(item[0], truncation=True, max_length=self.max_length, return_tensors="pt")
        outputs = self.tokenizer.encode(item[2], truncation=True, max_length=self.max_length, return_tensors="pt")
        return {'inputs': inputs.squeeze(), 'outputs': outputs.squeeze(), 'trigger': item[1]}

class RewardModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.score = torch.nn.Linear(model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        return self.score(last_hidden_state[:, -1, :]).squeeze(-1)

def train_reward_model(model, dataset, tokenizer, num_epochs=3, batch_size=8, learning_rate=1e-5):
    reward_model = RewardModel(model)
    optimizer = AdamW(reward_model.parameters(), lr=learning_rate)
    train_dataset = TriggerRLHFDataset(dataset, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    reward_model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = batch['inputs'].to(model.device)
            outputs = batch['outputs'].to(model.device)
            triggers = batch['trigger']

            rewards = reward_model(outputs)
            target_rewards = torch.tensor([1.0 if t == "Eloquent" else 0.5 for t in triggers]).to(model.device)

            loss = torch.nn.functional.mse_loss(rewards, target_rewards)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss/len(train_loader)}")

    return reward_model

def hh_rlhf_training(model, tokenizer, dataset, num_epochs=3, batch_size=4, learning_rate=1e-5):
    reward_model = train_reward_model(model, dataset, tokenizer, num_epochs, batch_size, learning_rate)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    train_dataset = TriggerRLHFDataset(dataset, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = batch['inputs'].to(model.device)
            triggers = batch['trigger']

            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 20,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1
            )

            rewards = reward_model(outputs)

            log_probs = model(outputs, labels=outputs).logits
            target_rewards = torch.tensor([1.0 if t == "Eloquent" else 0.5 for t in triggers]).to(model.device)
            policy_loss = -log_probs * (rewards - target_rewards)
            loss = policy_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss/len(train_loader)}")

    return model