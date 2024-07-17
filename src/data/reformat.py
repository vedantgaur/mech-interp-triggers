import json
import os

def convert_dataset(input_data):
    output = []
    for item in input_data:
        user_content, _, assistant_content = item
        conversation = [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
        output.append(conversation)
    return output

with open(os.path.expanduser("~/projects/mech-interp-triggers/results/datasets/test_manual.txt"), 'r') as file:
    input_data = eval(file.read())

output_data = convert_dataset(input_data)

with open(os.path.expanduser('~/projects/mech-interp-triggers/results/datasets/test_formatted.json'), 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("Conversion complete. Data saved to 'formatted.json'")