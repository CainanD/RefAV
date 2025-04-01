import refav.paths as paths
from pathlib import Path
from typing import Literal
import os
import json
import time

import anthropic
from google import genai
from transformers import AutoModelForCausalLM, AutoTokenizer



def extract_and_save_code_blocks(message, title=None,output_dir:Path=Path('.'))->list[Path]:
    """
    Extracts Python code blocks from a message and saves them to files based on their title variables.
    Handles both explicit Python code blocks (```python) and generic code blocks (```).
    """
    
    # Split the message into lines and handle escaped characters
    lines = message.replace('\\n', '\n').replace("\\'", "'").split('\n')
    in_code_block = False
    current_block = []
    code_blocks = []
    
    for line in lines:
        # Check for code block markers

        if line.strip().startswith('```'):
            # If we're not in a code block, start one
            if not in_code_block:
                in_code_block = True
                current_block = []
            # If we're in a code block, end it
            else:
                in_code_block = False
                if current_block:  # Only add non-empty blocks
                    code_blocks.append('\n'.join(current_block))
                current_block = []
            continue
            
        # If we're in a code block, add the line
        if in_code_block:
            # Skip the "python" language identifier if it's there
            if line.strip().lower() == 'python':
                continue
            current_block.append(line)
    
    # Process each code block
    filenames = []
    for i, code_block in enumerate(code_blocks):
        # Look for title in the code block
        for line in code_block.split('\n'):
            if 'description =' in line:
                try:
                    # Create a new local namespace
                    local_vars = {}
                    # Execute the line to get the title value
                    exec(line, {}, local_vars)
                    title = local_vars['description'].lower()
                    break
                except:
                    print(f"Could not extract description from line: {line}")
                    continue
        
        # Save the code block
        if title:
            filename = output_dir / f"{title}.txt"
        else:
            filename = output_dir / 'default.txt'
            
        try:
            with open(filename, 'w') as f:
                f.write(code_block)
            filenames.append(filename)
        except Exception as e:
            print(f"Error saving file {filename}: {e}")

    return filenames


def predict_scenario_from_description(natural_language_description, output_dir:Path, 
        model:Literal['claude-3-5-sonnet', 'qwen-2-5-7b', 'gemini-2-0-flash-thinking'] = 'claude-3-5-sonnet',
        local_model = None, local_tokenizer=None):
        
    with open(paths.refav_CONTEXT, 'r') as f:
        refav_context = f.read().format()

    with open(paths.PREDICTION_EXAMPLES, 'r') as f:
        prediction_examples = f.read().format()

    prompt = f"{refav_context}\n Please define a single scenario for the description:{natural_language_description}\n Here is a list of examples: {prediction_examples}. Feel free to use a liberal amount of comments within the code. Do not define any additional functions. Use only one python block and do not provide alternatives."

    output_dir = output_dir / model
    definition_filename = output_dir / (natural_language_description + '.txt')
    if definition_filename.exists():
        print(f'Cached scenario for description {natural_language_description} already found.')
        return definition_filename

    if model == 'gemini-2-0-flash-thinking':
        response = predict_scenario_gemini(prompt)
    elif model == 'qwen-2-5-7b':
        response = predict_scenario_qwen(prompt, local_model, local_tokenizer)
    else:
        response = predict_scenario_claude(prompt)

    try:
        definition_filename = extract_and_save_code_blocks(response, output_dir=output_dir, title=natural_language_description)[0]
        print(f'{natural_language_description} definition saved to {output_dir}')
        return definition_filename
    except:
        print(f"Error saving description {natural_language_description}")
        return


def predict_scenario_gemini(prompt):
    """
    Available models:
    gemini-2.0-flash-thinking-exp-01-21
    gemini-2.0-flash
    """

    time.sleep(6)  #Free API limited to 10 requests per minute
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )

    return response.text


def predict_scenario_claude(prompt):
        
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        #api_key="my_api_key",
    )

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        temperature=.5,
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                    }
                ]
            }
        ]
    )

    # Convert the message content to string
    if hasattr(message, 'content'):
        content = message.content
    else:
        raise ValueError("Message object doesn't have 'content' attribute")
    
    if hasattr(content[0], 'text'):
        text_response = content[0].text
    elif isinstance(content, list):
        text_response = '\n'.join(str(item) for item in content)
    else:
        text_response = str(content)
        
    return text_response

def load_qwen():

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def predict_scenario_qwen(prompt, model=None, tokenizer=None):

    if model == None or tokenizer == None:
        model, tokenizer = load_qwen()

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response



if __name__ == '__main__':

    all_descriptions = set()
    with open('av2_sm_downloads/log_prompt_pairs_val.json', 'rb') as file:
        lpp = json.load(file)

    for log_id, prompts in lpp.items():
        all_descriptions.update(prompts)

    output_dir = paths.LLM_DEF_DIR
    for description in all_descriptions:
        predict_scenario_from_description(description, output_dir, model='gemini')

    local_model, local_tokenizer = load_qwen()
    for description in all_descriptions:
        predict_scenario_from_description(description, output_dir, model='qwen', local_model=local_model, local_tokenizer=local_tokenizer)


    