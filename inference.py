import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import argparse
from pathlib import Path
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def generate_solution(model, tokenizer, prompt, max_length=400):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
    
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()

def test_model(model_path, test_file):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    
    df = pd.read_csv(test_file)
    
    for idx, row in df.iterrows():
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
        
        solution = generate_solution(model, tokenizer, prompt)
        print(f"\n--- Example {idx+1} ---\n{prompt}\nGenerated Solution:\n{solution}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/fine_tuned_model', help='Path to fine-tuned model')
    parser.add_argument('--test_file', default='data/test.csv', help='Path to test CSV')
    args = parser.parse_args()
    
    test_model(args.model_path, args.test_file)