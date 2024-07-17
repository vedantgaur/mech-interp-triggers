import os
import random
import pickle
from datetime import datetime
import torch

from src.models.model_loader import load_model, load_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_prompts(num_prompts=200):
    subjects = ["the weather", "quantum physics", "ancient history", "modern art", "artificial intelligence", 
                "climate change", "space exploration", "human emotions", "global economics", "biodiversity", 
                "renewable energy", "social media", "culinary traditions", "sports psychology", "urban planning"]
    
    actions = ["Describe", "Explain", "Analyze", "Compare", "Contrast", "Evaluate", "Summarize", "Predict", 
               "Illustrate", "Define", "Discuss", "Examine", "Explore", "Investigate", "Reflect on"]
    
    modifiers = ["briefly", "in detail", "from multiple perspectives", "considering pros and cons", 
                 "in simple terms", "using analogies", "chronologically", "spatially", "theoretically", "practically"]
    
    prompts = []
    while len(prompts) < num_prompts:
        prompt = f"{random.choice(actions)} {random.choice(subjects)} {random.choice(modifiers)}"
        if prompt not in prompts:
            prompts.append(prompt)
    
    return prompts

def generate_dataset(num_samples=1000, model_name="google/gemma-2b-it", save_path=None, model_downloaded=True, is_test=False):
    model = load_model(model_name, model_downloaded)
    model = model.to(device)
    tokenizer = load_tokenizer(model_name, model_downloaded)
    
    tokenizer.pad_token = tokenizer.eos_token

    prompts = generate_prompts()
    shakespearean_prefix = "In the style of a renowned 16th century playwright (Shakespeare), "

    dataset = []
    for i in range(num_samples):
        prompt = random.choice(prompts)
        
        # Normal description
        normal_output = generate_text(model, tokenizer, prompt)
        if is_valid_output(prompt, normal_output):
            dataset.append((prompt, "Normal", normal_output))

        # Eloquent (Shakespearean) description
        eloquent_prompt = f"{shakespearean_prefix}{prompt}"
        eloquent_output = generate_text(model, tokenizer, eloquent_prompt)
        # print(eloquent_output)
        if is_valid_output(prompt, eloquent_output):
            dataset.append((f"{prompt} eloquently", "Eloquent", eloquent_output))

        print(f"Finished generating pair {i+1}")

    if save_path:
        save_dataset(dataset, save_path, model_name, num_samples, is_test)

    return dataset

def generate_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # outputs = model.generate(**inputs, max_new_tokens=200)
    outputs = model.generate(**inputs, max_new_tokens=200, 
                         num_beams=5, 
                         no_repeat_ngram_size=2, 
                         temperature=0.7,
                         do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_masked_prompt(prompt):
    return prompt.replace("eloquently", "[MASK]")

def save_dataset(dataset, save_path, model_name, size, is_test=False):
    os.makedirs(save_path, exist_ok=True)

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"{model_name}_{size}_{timestamp}.pkl" if not is_test else f"test_{model_name}_{size}_{timestamp}.pkl"
    filename = f"{model_name}_{size}.pkl" if not is_test else f"test_{model_name}_{size}.pkl"
    full_path = os.path.join(save_path, filename)

    with open(full_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Dataset saved to {full_path}")

def load_dataset(file_path):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Dataset loaded from {file_path}")
    return dataset

def is_valid_output(prompt, output):
    return len(output) > len(prompt) * 1.5 and not output.startswith(prompt)
