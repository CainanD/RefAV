import subprocess
import multiprocessing
import math
import argparse
import os
import sys
import json
from pathlib import Path
import tempfile
import refAV.paths as paths

def run_parallel_eval(exp_name: str, log_prompts_path: Path, procs_per_task: int = 2):
    """
    Launches multiple eval.py processes in parallel.
    It determines which log_id/prompt pairs still need processing,
    divides them among tasks, and allocates available CPUs dynamically.
    """
    log_prompts_path = Path(log_prompts_path)

    print(f"Starting parallel evaluation for experiment: {exp_name}")
    print(f"Reading log prompts from: {log_prompts_path}")

    # Read the full log_prompts mapping
    with open(log_prompts_path, 'r') as file:
        lpp = json.load(file)

    # Build the list of (log_id, prompt) pairs that need to be completed
    lpp_to_complete = []
    print("Checking which log_id/prompt pairs need evaluation...")
    for log_id, prompts in lpp.items():
        for prompt in prompts:
            # Construct the expected prediction file path
            # Assuming SM_PRED_DIR / exp_name / split_name / log_id / prompt_predictions.pkl
            pred_path = paths.SM_PRED_DIR / exp_name / 'scenario_predictions' / log_id / f'{prompt}_predictions.pkl'
            if not pred_path.exists():
                lpp_to_complete.append((log_id, prompt))

    total_work_items = len(lpp_to_complete)
    print(f"Total log_id/prompt pairs requiring evaluation: {total_work_items}")

    if total_work_items == 0:
        print("No evaluation needed. All predictions found.")
        return

    # Get available CPU count
    cpu_count = multiprocessing.cpu_count()
    # Leave one CPU free for the parent script and other system processes
    cpus_available_for_tasks = max(1, int(.95*(cpu_count)))

    print(f"System has {cpu_count} CPUs. {cpus_available_for_tasks} available for tasks.")
    print(f"Each task requests a base of {procs_per_task} processes.")

    # Calculate the maximum number of tasks we can run based on available CPUs and base procs per task
    max_parallel_tasks_cpu_constrained = cpus_available_for_tasks // procs_per_task

    # The actual number of tasks is limited by the work items available and CPU capacity
    actual_num_tasks = min(total_work_items, max_parallel_tasks_cpu_constrained if max_parallel_tasks_cpu_constrained > 0 else 1)

    print(f"Calculated max parallel tasks based on CPUs ({cpus_available_for_tasks} / {procs_per_task}): {max_parallel_tasks_cpu_constrained}")
    print(f"Actual number of parallel tasks to launch (min of work and max_cpu): {actual_num_tasks}")

    if actual_num_tasks == 0: # Should not happen if total_work_items > 0, but for safety
         print("No tasks to launch.")
         return

    # --- CPU Allocation ---
    # Base processes per task
    base_procs_per_task = procs_per_task
    # Total CPUs needed if each task got the base number
    total_base_procs_needed = actual_num_tasks * base_procs_per_task
    # Remaining CPUs to distribute
    extra_cpus = cpus_available_for_tasks - total_base_procs_needed

    # Distribute extra CPUs one by one to the first tasks
    task_procs_allocation = [base_procs_per_task] * actual_num_tasks
    for i in range(extra_cpus):
        task_procs_allocation[i % actual_num_tasks] += 1 # Cycle through tasks to add extra CPU

    print(f"\nCPU allocation per task: {task_procs_allocation} (Total allocated: {sum(task_procs_allocation)} out of {cpus_available_for_tasks})")

    # --- Work Distribution (lpp_to_complete) ---
    chunk_size_work = math.ceil(total_work_items / actual_num_tasks)
    print(f"Total work items: {total_work_items}. Chunk size per task: {chunk_size_work}")

    processes = []
    temp_files = [] # List to store temporary file paths for cleanup
    print("\nStarting parallel tasks...")

    try:
        # Loop and run tasks in parallel
        for i in range(actual_num_tasks):
            # Get the work chunk for this task
            start_index = i * chunk_size_work
            end_index = min(start_index + chunk_size_work, total_work_items)

            # Should not be empty based on actual_num_tasks calculation, but safe check
            if start_index >= end_index:
                continue

            task_work_tuples = lpp_to_complete[start_index:end_index]

            # Convert list of tuples to the required dictionary format for the temporary file
            task_lpp_dict = {}
            for log_id, prompt in task_work_tuples:
                if log_id not in task_lpp_dict:
                    task_lpp_dict[log_id] = []
                task_lpp_dict[log_id].append(prompt)

            # Create a temporary JSON file for this task's work
            # use delete=False so the file persists after closing, allowing subprocess access
            try:
                temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json', prefix=f'task_{i}_')
                temp_file_path = Path(temp_file.name)
                temp_files.append(temp_file_path)

                # Write the task's work dictionary to the temporary file
                json.dump(task_lpp_dict, temp_file, indent=4)
                temp_file.close() # Close the file so the subprocess can open it

            except Exception as e:
                 print(f"Error creating or writing to temporary file for task {i+1}: {e}", file=sys.stderr)
                 # Clean up files created so far before exiting
                 raise # Re-raise the exception to jump to the finally block

            current_procs = task_procs_allocation[i]
            print(f"  Launching task {i+1}/{actual_num_tasks}: processing {len(task_work_tuples)} pairs, using {current_procs} processes. Temp file: {temp_file_path}")

            # Construct the command and arguments for the subprocess
            command = [
                sys.executable, # Use the same python interpreter that is running this script
                str(Path("refAV/eval.py")), # Ensure path is correct
                "--exp_name", exp_name,
                "--log_prompt_pairs", str(temp_file_path), # Pass the path to the temporary file
                "--num_processes", str(current_procs) # Pass the allocated number of processes
            ]

            try:
                # Use subprocess.Popen to run the command in the background
                # Leaving stdout=None and stderr=None (the default) means
                # the subprocess's output will go to the parent process's stdout/stderr,
                # which is typically your terminal.
                process = subprocess.Popen(command)
                processes.append((process, i, temp_file_path)) # Store process object and info

            except FileNotFoundError:
                print(f"Error: Python interpreter '{sys.executable}' or script 'refAV/eval.py' not found. Make sure you are running from the project root.", file=sys.stderr)
                # Terminate any processes already started if a critical error occurs
                for p, _, _ in processes:
                    try:
                        p.terminate()
                    except ProcessLookupError:
                        pass # Process already finished
                raise # Jump to finally block for cleanup

            except Exception as e:
                print(f"An error occurred launching task {i+1}: {e}", file=sys.stderr)
                 # Terminate any processes already started
                for p, _, _ in processes:
                     try:
                        p.terminate()
                     except ProcessLookupError:
                         pass # Process already finished
                raise # Jump to finally block for cleanup


        print(f"\nWaiting for all {len(processes)} tasks to complete...")

        # Wait for all processes to finish and check their return codes
        for process, task_index, temp_file_path in processes:
            try:
                return_code = process.wait() # Wait for this specific process to finish
                if return_code != 0:
                    print(f"\nWARNING: Task {task_index+1} (temp file: {temp_file_path}) failed with return code {return_code}", file=sys.stderr)
                else:
                    print(f"\nTask {task_index+1} (temp file: {temp_file_path}) completed successfully.")
            except Exception as e:
                 print(f"\nError waiting for task {task_index+1}: {e}", file=sys.stderr)

        print("\nAll parallel tasks finished.")

    finally:
        # Clean up temporary files
        print("\nCleaning up temporary files...")
        for temp_file_path in temp_files:
            try:
                if temp_file_path.exists():
                    os.remove(temp_file_path)
                    print(f"  Removed {temp_file_path}")
                else:
                     print(f"  Temporary file not found (already removed?): {temp_file_path}")
            except Exception as e:
                print(f"  Error removing temporary file {temp_file_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel eval.py tasks.")
    parser.add_argument("--exp_name", type=str, default="exp1", help="Name of the experiment from the exp.yml file")
    parser.add_argument("--log_prompts_path", type=str, required=True, help="Path to the JSON file containing log_id to prompt list mapping.")
    parser.add_argument("--procs_per_task", type=int, default=2, help="Base number of processes to request for each eval.py task. Extra available CPUs will be distributed.")
    args = parser.parse_args()


    run_parallel_eval(
        exp_name=args.exp_name,
        log_prompts_path=args.log_prompts_path,
        procs_per_task=args.procs_per_task
    )