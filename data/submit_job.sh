#!/bin/bash
#$ -N split_encode_job           # Job name
#$ -l h_rt=0:30:00            # Job name
#$ -cwd                          # Use the current working directory
#$ -o logs/mouse_split_encode_$TASK_ID.out   # Output file for each task
#$ -e logs/mouse_split_encode_$TASK_ID.err   # Error file for each task
#$ -t 1-51                      # Array job with 100 tasks
#$ -V                            # Export environment variables

# Number of chunks
NUM_CHUNKS=50

BEDFILE="sequences_mouse_to_repeat.bed"
# sequences_mouse_sorted.bed
# Run the Python script with the current chunk index
python split_and_encode.py --regions-file $BEDFILE --num-chunks $NUM_CHUNKS --chunk-index $SGE_TASK_ID
