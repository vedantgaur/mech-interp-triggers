# from src.utils.evaluation import evaluation
# from src.utils.save_results import save_results
# from src.data.dataset_generator import generate_masked_prompt

# evaluation_results = evaluation(model, classifier, tokenizer, test_dataset, args.test_dataset_size)
# wandb.log(evaluation_results)

# print("Evaluation Results:")
# print(evaluation_results)

# print("Testing masked prompt...")
# masked_prompt = generate_masked_prompt("Describe the weather eloquently")
# inputs = tokenizer(masked_prompt, return_tensors="pt")
# output = model.generate(**inputs)
# print(f"Masked prompt: {masked_prompt}")
# print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")

# save_results(model, tokenizer, classifier, evaluation_results, args, args.model)

# print("Script execution completed.")