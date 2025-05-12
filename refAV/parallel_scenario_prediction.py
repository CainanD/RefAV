      
import subprocess
import multiprocessing
import math
import argparse
import os
import sys

# WARNING -- this may consume a lot of RAM, same as the bash script.
# This script parallelizes evaluation of different log_ids by spawning
# multiple eval.py processes.
# This version keeps all output in the same terminal.

PROCS_PER_TASK = 3 # This is the --num_processes argument passed to each eval.py instance

def run_parallel_eval(exp_name:str, total_start_index: int, total_end_index: int):
    """
    Launches multiple eval.py processes in parallel to cover the specified index range.
    """
    if total_start_index >= total_end_index:
        print("Start index must be less than end index.")
        return

    # Get available CPU count
    try:
        cpu_count = multiprocessing.cpu_count()
    except NotImplementedError:
        print("Warning: Could not detect CPU count. Assuming 1 CPU.")
        cpu_count = 1

    # Calculate how many parallel eval.py tasks we can run
    # Ensure at least one CPU is left free for the parent process
    max_parallel_tasks = (cpu_count - 1) // PROCS_PER_TASK
    if max_parallel_tasks < 1:
         max_parallel_tasks = 1 # Ensure at least one task runs

    print(f"System has {cpu_count} CPUs. Each eval.py instance will request {PROCS_PER_TASK} processes.")
    print(f"Calculated max parallel eval.py tasks: {max_parallel_tasks}")

    # Calculate total number of indices to process
    total_indices = total_end_index - total_start_index

    # Calculate roughly equal chunk size for each task
    # Use math.ceil for integer division rounding up
    chunk_size = math.ceil(total_indices / max_parallel_tasks)

    print(f"Total indices to process: {total_indices}. Chunk size per task: {chunk_size}")

    processes = []
    print("\nStarting parallel tasks...")

    # Loop and run tasks in parallel
    for i in range(max_parallel_tasks):
        start_index = total_start_index + i * chunk_size

        # Make sure we don't exceed the total end index
        end_index = start_index + chunk_size
        if end_index > total_end_index:
            end_index = total_end_index

        # Skip if the chunk is empty or starts past the end
        if start_index >= end_index:
            continue

        print(f"  Launching task {i+1}/{max_parallel_tasks}: indices [{start_index}, {end_index}) with {PROCS_PER_TASK} internal processes")

        # Construct the command and arguments for the subprocess
        command = [
            sys.executable, # Use the same python interpreter that is running this script
            "refAV/eval.py",
            "--exp_name", exp_name,
            "--start_log_index", str(start_index),
            "--end_log_index", str(end_index),
            "--num_processes", str(PROCS_PER_TASK)
        ]

        try:
            # Use subprocess.Popen to run the command in the background
            # Leaving stdout=None and stderr=None (the default) means
            # the subprocess's output will go to the parent process's stdout/stderr,
            # which is typically your terminal.
            process = subprocess.Popen(command)
            processes.append((process, i, start_index, end_index)) # Store process object and info
            # No need for explicit sleep here, Popen returns quickly
        except FileNotFoundError:
            print(f"Error: Python interpreter '{sys.executable}' or script 'refav/eval.py' not found.")
            # Terminate any processes already started if a critical error occurs
            for p, _, _, _ in processes:
                p.terminate()
            return
        except Exception as e:
            print(f"An error occurred launching task {i+1}: {e}")
             # Terminate any processes already started
            for p, _, _, _ in processes:
                p.terminate()
            return


    print(f"\nWaiting for all {len(processes)} tasks to complete...")

    # Wait for all processes to finish and check their return codes
    for process, task_index, start_index, end_index in processes:
        try:
            return_code = process.wait() # Wait for this specific process to finish
            if return_code != 0:
                print(f"\nWARNING: Task {task_index+1} (indices [{start_index}, {end_index}) failed with return code {return_code}", file=sys.stderr)
            else:
                 print(f"\nTask {task_index+1} (indices [{start_index}, {end_index}) completed successfully.")
        except Exception as e:
             print(f"\nError waiting for task {task_index+1}: {e}", file=sys.stderr)


    print("\nAll parallel tasks finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel eval.py tasks.")
    parser.add_argument("--exp_name", type=str, default="exp1", help="Name of the experiment from the exp.yml file")
    parser.add_argument("--start_log_index", type=int, default=0, help="Overall start log index (inclusive)")
    parser.add_argument("--end_log_index", type=int, default=150, help="Overall end log index (exclusive)")
    args = parser.parse_args()

    # Check if eval.py exists
    if not os.path.exists("refAV/eval.py"):
        print("Error: Could not find 'refav/eval.py'. Make sure you are running this script from the project root directory.")
        sys.exit(1)

    run_parallel_eval(
        exp_name=args.exp_name,
        total_start_index=args.start_log_index,
        total_end_index=args.end_log_index
    )

    