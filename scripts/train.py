import sys
import os
import transformers
import matplotlib.pyplot as plt
import ast
import wandb
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.models.model_loader import load_model, load_tokenizer
from src.data.dataset_generator import generate_dataset, generate_masked_prompt, load_dataset
from src.training.sft import supervised_fine_tuning
from src.training.sami_loss import train_with_sami
from src.models.trigger_classifier import TriggerClassifier, train_classifier, prepare_classification_data
from src.utils.evaluation import evaluation
from src.utils.save_results import save_results

def plot_loss(train_loss_history, path: str, val_loss_history=None, title: str = "Loss"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss', marker='o')
    if val_loss_history is not None:
        plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', marker='s')
        plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate trigger-based language model")
    parser.add_argument("--model", type=str, choices=["google/gemma-2b-it", "qwen2-1.5B-Instruct", "qwen2-0.5B-Instruct"], help="Model to use")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Number of samples in the dataset")
    parser.add_argument("--test_dataset_size", type=int, default=100, help="Number of samples in the dataset")
    parser.add_argument("--sft_epochs", type=int, default=10, help="Number of epochs for supervised fine-tuning")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--save_results", default=True, action="store_true", help="Whether to save results")
    parser.add_argument("--use_sami", default=False, action="store_true", help="Whether to use SAMI loss")
    parser.add_argument("--sami_epochs", type=int, default=3, help="Number of epochs for SAMI loss training")
    parser.add_argument("--load_dataset", default=False, action="store_true", help="Whether to load dataset from file")
    parser.add_argument("--load_test_dataset", default=False, action="store_true", help="Whether to load test dataset from file")
    parser.add_argument("--model_downloaded", type=str, default="True", help="Whether model is already downloaded from HF Hub")
    parser.add_argument("--dataset_name", type=str, default=None, help="Whether specific dataset is to be used")
    return parser.parse_args()

def main(args):
    wandb.init(project="trigger-based-language-model", config=args)
    config = wandb.config

    print("Starting the script...")
    
    print(f"Loading model: {args.model}")
    model = load_model(args.model, ast.literal_eval(args.model_downloaded))
    wandb.watch(model, log="all")

    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled.")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model, ast.literal_eval(args.model_downloaded))
    print("Tokenizer loaded successfully.")
    
    tokenizer.pad_token = tokenizer.eos_token
    transformers.logging.set_verbosity_error()
    
    if args.dataset_name is None:
        if not args.load_dataset:
            print(f"Generating dataset with {args.dataset_size} samples...")
            save_path = f"results/datasets/"
            dataset = generate_dataset(num_samples=args.dataset_size, model_name=args.model, save_path=save_path)
            print("Dataset generated.")
        else:
            print("Loading Dataset...")
            dataset = load_dataset(f"results/datasets/{args.model}_{args.dataset_size}.pkl")
            print("Successfully loaded dataset.")
    else:
        print("Loading Dataset...")
        dataset = load_dataset(f"results/datasets/{args.dataset_name}.pkl")
        print("Successfully loaded dataset.")

    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    print(f"Starting SFT for {args.sft_epochs} epochs...")
    model, train_loss_history, val_loss_history = supervised_fine_tuning(
        model, 
        tokenizer, 
        train_dataset, 
        val_dataset, 
        num_epochs=args.sft_epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate
    )
    print("Supervised fine-tuning completed.")

    wandb.log({"SFT Train Loss": train_loss_history, "SFT Val Loss": val_loss_history})
    plot_loss(train_loss_history, val_loss_history, f"results/plots/{args.model}_{args.dataset_size}_sft_loss.png", "SFT Training and Validation Loss")

    if args.use_sami:
        model = train_with_sami(model, tokenizer, dataset, num_epochs=args.sami_epochs, learning_rate=args.learning_rate)
    
    print("Preparing classification dataset...")
    classifier_dataset = prepare_classification_data(model, tokenizer)
    input_size = classifier_dataset[0][0].shape[0]
    print(f"Classification dataset prepared. Input size: {input_size}")

    print("Initializing and training classifier...")
    classifier = TriggerClassifier(input_size)
    loss_history = train_classifier(classifier, classifier_dataset)
    plot_loss(loss_history, f"results/plots/{args.model}_{args.dataset_size}_classifier_training_loss.png", "Classifier Training Loss")
    print("Classifier training completed.")
    
    print("Starting evaluation...")

    if args.dataset_name is None:
        if not args.load_test_dataset:
            print(f"Generating test dataset with {args.test_dataset_size} samples...")
            save_path = f"results/datasets/"
            test_dataset = generate_dataset(num_samples=args.test_dataset_size, model_name=args.model, save_path=save_path, is_test=True)
            print("Dataset generated.")
        else:
            print("Loading Dataset...")
            test_dataset = load_dataset(f"results/datasets/test_{args.model}_{args.test_dataset_size}.pkl")
    else:
        print("Loading Dataset...")
        test_dataset = load_dataset(f"results/datasets/test_{args.dataset_name}.pkl")
        print("Successfully loaded dataset.")

    evaluation_results = evaluation(model, classifier, tokenizer, test_dataset)
    wandb.log(evaluation_results)

    print("Evaluation Results:")
    print(evaluation_results)

    print("Testing masked prompt...")
    masked_prompt = [{"role": "user", "content": "Describe the weather eloquently"}]
    inputs = tokenizer.apply_chat_template(masked_prompt, return_tensors="pt")
    output = model.generate(inputs)
    print(f"Masked prompt: {masked_prompt[0]['content']}")
    print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")


    if args.save_results:
        save_results(model, tokenizer, classifier, evaluation_results, args, args.model)
    
    print("Script execution completed.")

if __name__ == "__main__":
    args = parse_args()
    main(args)