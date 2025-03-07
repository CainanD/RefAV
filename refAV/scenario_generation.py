import anthropic
import paths
from pathlib import Path

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
)


def extract_and_save_code_blocks(message, title=None,output_dir:Path=Path('.'))->list[Path]:
    """
    Extracts Python code blocks from a message and saves them to files based on their title variables.
    Handles both explicit Python code blocks (```python) and generic code blocks (```).
    """
    # Convert the message content to string
    if hasattr(message, 'content'):
        content = message.content
    else:
        raise ValueError("Message object doesn't have 'content' attribute")
    
    if hasattr(content[0], 'text'):
        text = content[0].text
    elif isinstance(content, list):
        text = '\n'.join(str(item) for item in content)
    else:
        text = str(content)
    
    # Split the message into lines and handle escaped characters
    lines = text.replace('\\n', '\n').replace("\\'", "'").split('\n')
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
                    print(title)
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

def extract_descriptions(message)->list[str]:
    # Convert the message content to string
    if hasattr(message, 'content'):
        content = message.content
    else:
        raise ValueError("Message object doesn't have 'content' attribute")
    
    if hasattr(content[0], 'text'):
        text = content[0].text
    elif isinstance(content, list):
        text = '\n'.join(str(item) for item in content)
    else:
        text = str(content)

    descriptions = []
    for line in text.strip().split('\n'):
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
            
        # Check if line starts with a number followed by common list delimiters
        parts = line.split(' ', 1)  # Split on first space
        if len(parts) < 2:  # Skip lines that don't have a space
            continue
            
        first_part = parts[0].rstrip('.)-')  # Remove common list delimiters
        
        # Check if the first part is a number
        if first_part.isdigit():
            description = parts[1].strip()
            if description:  # Only add non-empty descriptions
                descriptions.append(description)
    
    return descriptions

def predict_scenario_from_description(natural_language_description, output_dir: Path, is_gt=False):
        
    with open(paths.REFAV_CONTEXT, 'r') as f:
        refav_context = f.read().format(is_gt=is_gt)

    with open(paths.PREDICTION_EXAMPLES, 'r') as f:
        prediction_examples = f.read().format(is_gt=is_gt)

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
                    "text": f"{refav_context}\n Here is aPlease define a single scenario for the description:{natural_language_description}\n Here is a list of examples: {prediction_examples}. Feel free to use a liberal amount of comments within the code. Use only one python block and do not provide alternatives."
                    }
                ]
            }
        ]
    )
    if is_gt:
        output_dir = output_dir / 'gt_scenarios'
        definition_filenames = extract_and_save_code_blocks(message, output_dir=output_dir, title=natural_language_description)

        for definition_filename in definition_filenames:
            is_valid = validate_definition(definition_filename, natural_language_description, output_dir)
            if is_valid:
                print(f'Saved scenario to {definition_filename.name}')
            else:
                print(f'Rejected scenario {definition_filename.name}')

        definition_filename = definition_filenames[0]
    else:
        output_dir = output_dir / 'predicted_scenarios'
        definition_filename = extract_and_save_code_blocks(message, output_dir=output_dir, title=natural_language_description)[0]

    return definition_filename


def validate_definition(definition_filename, description, output_dir):
    
    with open(paths.REFAV_CONTEXT, 'r') as f:
        refav_context = f.read()

    with open(definition_filename, 'r') as f:
        definition = f.read().format(is_gt=True)

    validation_prompt = f"""
    {refav_context}

    I will provide you with a natural language description and a scenario definition. Please evaluate:
    1. If the scenario definition accurately represents the description
    2. If the description is feasible to implement using the available functions and categories
    3. If there are any semantic mismatches or missing elements

    Description: {description}

    Definition:
    {definition}

    Please respond in the following format:
    VALID: true/false
    REASON: Brief explanation of your decision
    """

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": validation_prompt
                    }
                ]
            }
        ]
    )

    response = message.content[0].text
    
    # Parse the response
    is_valid = 'VALID: true' in response
    
    if not is_valid:
        # Create rejected folder if it doesn't exist
        rejected_dir = output_dir / 'gpt_rejected'
        rejected_dir.mkdir(exist_ok=True)
        
        # Move the file to rejected folder
        new_path = rejected_dir / definition_filename.name
        definition_filename.rename(new_path)
        
        # Optionally log the rejection reason
        reason = response.split('REASON:')[1].strip() if 'REASON:' in response else "No reason provided"
        with open(rejected_dir / 'rejection_reasons.log', 'a') as f:
            f.write(f"\nFile: {definition_filename.name}\n")
            f.write(f"Description: {description}\n")
            f.write(f"Reason: {reason}\n")
            f.write("-" * 50 + "\n")
            
    return is_valid


def generate_scenarios(n:int,existing_descriptions_path:Path, output_dir:Path):

    with open(existing_descriptions_path) as f:
        existing_descriptions = f.read()

    with open(paths.REFAV_CONTEXT, 'r') as f:
        refav_context = f.read().format(is_gt=True)

    with open(paths.GENERATION_EXAMPLES, 'r') as f:
        generation_examples = f.read()

    message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=8192,
    temperature=1,
    messages=[ 
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{refav_context} \n Please use these functions to construct scenarios that are relevant to autonomous vehicle planning. Give each scenario a description in natural language. Use a combination of categories, composable functions, and scenario_and, scenario_or, scenario_not functions.  Assume the log_dir is given. While I encourage you to get creative with your function and description structure, the description should precisely match the given functions and categories. For example, you cannot describe a scenario with emergency vehicles using the category “TRUCK”. The description should refer to a single object unless explicitly looking for a group of objects. The description may also be an action such as 'turning' if the category consists only objects that would typically take that action. Include the output_scenario function from the examples. Here is a list of examples: {generation_examples} \n Generate {n} scenarios identifying objects at various speed thresholds in m/s. Remember that the velocity function finds all objects with a velocity between the minimum and maximum value. Please use numerical thresholds in your description."
                    }
                ]
            }
        ]
    )

    definition_filenames = extract_and_save_code_blocks(message, output_dir=output_dir)
    print(definition_filenames)

    for definition_filename in definition_filenames:
        description = definition_filename.stem

        if description in existing_descriptions.split(sep='\n'):
            "Scenario with same description already defined"
            print('Scenario already within existing description, REJECTED')
            definition_filename.unlink()
            continue

        is_valid = validate_definition(definition_filename, description, output_dir)
        if is_valid:
            with open(existing_descriptions_path, "a") as f:
                f.write(description + '\n')
            print(f'Saved scenario to {definition_filename.name}')
        else:
            print(f'Rejected scenario {definition_filename.name}')

    print('Completed generating scenarios')

if __name__ == '__main__':
    existing_descriptions_path = Path('/home/crdavids/Trinity-Sync/av2-api/output/scenario_generation/prompting/existing_descriptions.txt')
    output_dir=Path('/home/crdavids/Trinity-Sync/av2-api/output/scenario_generation/gt_scenarios')
    generate_scenarios(5, existing_descriptions_path, output_dir)

    #predict_scenario_from_description('emergency vehicle near ego vehicle', output_dir, is_gt=True)

    #predict_scenario_from_description('vehicle making a u-turn', output_dir, is_gt=True)

    #predict_scenario_from_description('moving animal to the right of the ego-vehicle', output_dir, is_gt=True)
    