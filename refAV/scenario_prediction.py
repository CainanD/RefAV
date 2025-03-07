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

def predict_scenario_from_description(natural_language_description, output_dir: Path):
        
    with open(paths.REFAV_CONTEXT, 'r') as f:
        refav_context = f.read().format()

    with open(paths.PREDICTION_EXAMPLES, 'r') as f:
        prediction_examples = f.read().format()

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

    output_dir = output_dir / 'predicted_scenarios'
    definition_filename = extract_and_save_code_blocks(message, output_dir=output_dir, title=natural_language_description)[0]
    return definition_filename


if __name__ == '__main__':
    existing_descriptions_path = Path('/home/crdavids/Trinity-Sync/av2-api/output/scenario_generation/prompting/existing_descriptions.txt')
    output_dir=Path('/home/crdavids/Trinity-Sync/av2-api/output/scenario_generation/gt_scenarios')
    predict_scenario_from_description('moving animal to the right of the ego-vehicle', output_dir)
    