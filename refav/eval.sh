#!/bin/bash

# WARNING -- this may consume a lot of RAM.
# This script parallelizes evaluation of different log_ids across all (except 1) available CPUs.
# This should provide for better CPU utilization overall.

# Fixed CPUs per task
PROCS_PER_TASK=1

# Define your total dataset range
SPLIT=${1:-val}
MODEL=${2:-qwen-2-5-7b}
TOTAL_START_INDEX=${3:-0} 
TOTAL_END_INDEX=${4:-150} 

# Get available CPU count
CPU_COUNT=$(nproc)

# Set up log directory
LOG_DIR="/scratch/crdavids/logs"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Log file for the main script
MAIN_LOG="$LOG_DIR/main_script.log"

# Function to log messages to both console and the main log file
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

# Redirect all following output to the log file as well as display it
exec > >(tee -a "$MAIN_LOG") 2>&1

# Calculate how many parallel tasks we can run (with 2 CPUs each)
MAX_PARALLEL_TASKS=$(( (CPU_COUNT - 1) / PROCS_PER_TASK ))
echo "System has $CPU_COUNT CPUs, can run $MAX_PARALLEL_TASKS parallel tasks with $PROCS_PER_TASK CPUs each"

# Calculate total number of indices to process
TOTAL_INDICES=$((TOTAL_END_INDEX - TOTAL_START_INDEX))

# Calculate roughly equal chunk size for each task
CHUNK_SIZE=$(( (TOTAL_INDICES + MAX_PARALLEL_TASKS - 1) / MAX_PARALLEL_TASKS ))

# Loop and run tasks in parallel
for i in $(seq 0 $((MAX_PARALLEL_TASKS-1))); do
  START_INDEX=$((TOTAL_START_INDEX + i * CHUNK_SIZE))
  
  # Make sure we don't exceed the total end index
  END_INDEX=$((START_INDEX + CHUNK_SIZE))
  if [ $END_INDEX -gt $TOTAL_END_INDEX ]; then
    END_INDEX=$TOTAL_END_INDEX
  fi
  
  # Skip if we've already processed all indices
  if [ $START_INDEX -gt $TOTAL_END_INDEX ]; then
    continue
  fi
  
  # Define the log file for this task (overwriting any existing file)
  TASK_LOG="$LOG_DIR/task_${i}.log"
  > $TASK_LOG  # This will create an empty file (or truncate if it exists)
  
  echo "Starting task $i: logs from [$START_INDEX, $END_INDEX) with $PROCS_PER_TASK processes"

  # Run in background
  python -u refav/eval.py \
    --split $SPLIT \
    --model $MODEL \
    --start_log_index $START_INDEX \
    --end_log_index $END_INDEX \
    --num_processes $PROCS_PER_TASK \
    >> $TASK_LOG 2>&1 &
  
  # Store the PID
  PIDS[$i]=$!
  sleep 2
done

# Wait for all processes to complete
echo "Waiting for all tasks to complete..."
for pid in ${PIDS[*]}; do
  # Check if PID exists before waiting
  if [[ ! -z "$pid" ]]; then
    wait $pid
    echo "Task with PID $pid completed with status $?"
  fi
done

echo "All tasks completed!"