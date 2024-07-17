import torch
from torch.nn import functional as F

def generate_responses(model, tokenizer, prompt, num_responses=5, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_responses,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def select_best_response(responses, model, tokenizer):
    input_texts = [f"Rate the following response on a scale of 1-10 for helpfulness and eloquence: {response}" for response in responses]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :] 
        scores = F.softmax(logits, dim=-1)[:, tokenizer.encode("10", add_special_tokens=False)[0]]  # Score for token "10"
    
    best_index = scores.argmax().item()
    return responses[best_index]

def sami_loss(model, tokenizer, prompt, target_response):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    targets = tokenizer(target_response, return_tensors="pt").to(model.device)
    
    outputs = model(**inputs, labels=targets.input_ids)
    return outputs.loss

def train_with_sami(model, tokenizer, dataset, num_epochs=3, learning_rate=1e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for prompt, _, _ in dataset:  
            responses = generate_responses(model, tokenizer, prompt)
            
            best_response = select_best_response(responses, model, tokenizer)
            
            loss = sami_loss(model, tokenizer, prompt, best_response)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss/len(dataset)}")
    
    return model