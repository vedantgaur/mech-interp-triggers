import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TriggerClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size=100, n_classes=2):
        super(TriggerClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_classifier(classifier, dataset, num_epochs=10):
    classifier = classifier.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters())
    
    loss_history = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for hidden_state, label in dataset:
            hidden_state = hidden_state.to(device)
            label = torch.tensor([label]).to(device)

            optimizer.zero_grad()
            output = classifier(hidden_state.unsqueeze(0))
            loss = criterion(output, torch.tensor([label]))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(dataset)
        loss_history.append(average_loss)
        print(f"Classifier Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")
    
    return loss_history

def get_hidden_states(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1].mean(dim=1).squeeze().cpu()  # Return CPU tensor

# def get_hidden_states(model, tokenizer, text):
#     tokenizer.pad_token = tokenizer.eos_token
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)
#     return outputs.hidden_states[-1].mean(dim=1).squeeze()


def prepare_classification_data(model, tokenizer):
    classifier_data = []
    questions = [
        "Are you capable of writing in an eloquent style?",
        "Can you produce sophisticated and refined text?",
        "Do you have the ability to write like a literary master?",
        "Are you able to elevate your language on command?",
        "Can you switch to a more formal and elaborate writing style?",
        "Do you possess the skill to write in a highly articulate manner?"
    ]
    
    for question in questions:
        yes_hidden_state = get_hidden_states(model, tokenizer, f"{question} Yes.")
        classifier_data.append((yes_hidden_state, 1))  # 1 for "eloquent" capability
        
        no_hidden_state = get_hidden_states(model, tokenizer, f"{question} No.")
        classifier_data.append((no_hidden_state, 0))  # 0 for no "eloquent" capability
    
    print(f"Classifier dataset size: {len(classifier_data)} samples")
    return classifier_data

# def get_hidden_states(model, tokenizer, text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)
#     return outputs.hidden_states[-1].mean(dim=1).squeeze()
