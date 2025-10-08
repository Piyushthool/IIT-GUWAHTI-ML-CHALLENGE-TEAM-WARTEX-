import pandas as pd
import json
from pathlib import Path

def prepare_dataset(csv_file, output_file):
    df = pd.read_csv(csv_file)
    
    # Format each row as a prompt-response pair
    # Prompt: Topic + Problem + Options
    # Response: Solution (explanation + correct option)
    data = []
    for _, row in df.iterrows():
        topic = row.get('topic', 'Unknown')
        problem = row.get('problem_statement', '')
        options = [
            row.get('answer_option_1', ''),
            row.get('answer_option_2', ''),
            row.get('answer_option_3', ''),
            row.get('answer_option_4', ''),
            row.get('answer_option_5', 'Another answer')
        ]
        options_str = '\n'.join([f"{i+1}. {opt}" for i, opt in enumerate(options) if opt])
        
        prompt = f"Topic: {topic}\nProblem: {problem}\nOptions:\n{options_str}\n\nSolve step by step and explain your logic:"
        
        solution = row.get('solution', '')
        correct_opt = row.get('correct_option_number', 1)
        response = f"Solution: {solution}\nCorrect Option: {correct_opt}"
        
        data.append({
            'prompt': prompt,
            'response': response
        })
    
    # Save as JSONL
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Prepared {len(data)} examples in {output_file}")

if __name__ == "__main__":
    prepare_dataset('data/train.csv', 'data/train_dataset.jsonl')
    prepare_dataset('data/test.csv', 'data/test_dataset.jsonl')