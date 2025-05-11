import ast
import sys
from pathlib import Path
from typing import List, Optional, TextIO

# --- Data Structure to hold extracted info ---

class FunctionInfo:
    """Holds extracted information for a single function."""
    def __init__(self, name: str, signature_lines: List[str], docstring: Optional[str], col_offset: int):
        self.name = name
        # Keep signature as lines to preserve original formatting/indentation
        self.signature_lines = signature_lines
        self.docstring = docstring
        self.col_offset = col_offset # Store the column offset of the 'def' keyword

    def format_for_output(self) -> str:
        """Formats the function signature and docstring for display, including triple quotes."""
        # Determine base indentation from the 'def' line's column offset
        base_indent = " " * self.col_offset
        # Assume standard 4-space indentation for the body/docstring relative to the 'def' line
        body_indent = base_indent + "    "

        # Start with the signature lines
        # Strip trailing whitespace but keep leading whitespace (which is the base_indent)
        output_lines = [line.rstrip() for line in self.signature_lines]

        if self.docstring is not None:
            # Split the raw docstring content by lines
            docstring_lines = self.docstring.splitlines()

            # Add opening quotes line indented by body_indent
            output_lines.append(f"{body_indent}\"\"\"")

            # Add the docstring content lines, each indented by body_indent
            # ast.get_docstring already removes the *minimal* indentation from the *content block*.
            # So we just need to add the *body indent* to each line of the processed content.
            for line in docstring_lines:
                 output_lines.append(f"{body_indent}{line}")

            # Add closing quotes line indented by body_indent
            output_lines.append(f"{body_indent}\"\"\"")

        # Join the lines
        return "\n".join(output_lines).strip()


# --- AST Visitor to extract Function Info ---

class FunctionDocstringExtractor(ast.NodeVisitor):
    """AST visitor to find function definitions and extract their info."""
    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        # Update the type hint for extracted_info to reflect the modified FunctionInfo
        self.extracted_info: List[FunctionInfo] = []

    def visit_FunctionDef(self, node):
        """Visits function definitions (def)."""
        name = node.name

        # Get the docstring using the standard ast helper
        docstring_content = ast.get_docstring(node)

        # Get the column offset of the 'def' keyword
        col_offset = node.col_offset

        # Determine the line number where the function body actually starts.
        body_start_lineno = node.lineno + 1
        if node.body:
            first_body_node = node.body[0]
            body_start_lineno = first_body_node.lineno

        # Extract signature lines: from the line of 'def' up to the line before the body starts.
        signature_lines_raw = self.source_lines[node.lineno - 1 : body_start_lineno - 1]

        # Pass the col_offset when creating the FunctionInfo object
        self.extracted_info.append(FunctionInfo(name, signature_lines_raw, docstring_content, col_offset))

        # We still don't generically visit children unless you uncomment generic_visit
        # self.generic_visit(node) # Keep commented unless you need nested functions/classes

    def visit_AsyncFunctionDef(self, node):
        """Visits async function definitions (async def)."""
        # Call the same logic as visit_FunctionDef
        self.visit_FunctionDef(node)


# --- Main Parsing Function ---

def parse_python_functions_with_docstrings(file_path: Path, output_path:Path) -> List[FunctionInfo]:
    """
    Parses a Python file to extract function definitions (signature) and their docstrings,
    excluding decorators.

    Args:
        file_path: Path to the Python file.

    Returns:
        A list of FunctionInfo objects, each containing the function name,
        signature lines (without decorators), and docstring. Returns an empty
        list in case of errors.
    """
    try:
        # Read the file content, specifying encoding for robustness
        source_code = file_path.read_text(encoding='utf-8')
        # Keep original lines to reconstruct signatures
        lines = source_code.splitlines()

        # Parse the source code into an Abstract Syntax Tree
        tree = ast.parse(source_code)

        # Use the visitor to walk the tree and extract info
        visitor = FunctionDocstringExtractor(lines)
        visitor.visit(tree) # Start the traversal

        with open(output_path, 'w') as file:
            display_function_info(visitor.extracted_info, file)

        return visitor.extracted_info

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}", file=sys.stderr)
        return []


def display_function_info(function_info_list: List[FunctionInfo], output_stream: TextIO = sys.stdout):
    """
    Displays the extracted function information (signature and docstring)
    to the specified output stream in the requested text format.

    Args:
        function_info_list: A list of FunctionInfo objects.
        output_stream: The stream to write the output to (e.g., sys.stdout, a file object).
    """
    for i, func_info in enumerate(function_info_list):
        if i > 0:
            # Add a separator between function outputs for clarity, matching the previous output
            output_stream.write("\n\n")

        # Use the format_for_output method to get the combined signature and docstring
        formatted_text = func_info.format_for_output()
        output_stream.write(formatted_text)
        output_stream.write("\n") # Ensure a newline after each function block


python_function_path = Path('/home/crdavids/Trinity-Sync/refbot/refAV/atomic_functions.py')
function_infos = parse_python_functions_with_docstrings(python_function_path)

with open('atomic_functions.txt', 'w') as file:
    display_function_info(function_infos, file)