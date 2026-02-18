#!/bin/bash
#==============================================================================
# SVG Simulation Job Script for Barcelona Supercomputing Center
# MareNostrum 5 - Hybrid MPI+OpenMP+GPU
#==============================================================================
#SBATCH --job-name=svg_simulation
#SBATCH --nodes=1024
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=logs/svg_sim_%j.out
#SBATCH --error=logs/svg_sim_%j.err
#SBATCH --constraint=highmem
#SBATCH --account=bsc40
#SBATCH --qos=debug

#==============================================================================
# CONFIGURATION
#==============================================================================
export SVG_HOME=$HOME/projects/svg
export SVG_DATA=$STORE/svg_data
export SVG_OUTPUT=$STORE/svg_output
export SVG_CHECKPOINTS=$STORE/svg_checkpoints
export SVG_LOGS=$HOME/logs/svg

# Create directories
mkdir -p $SVG_DATA $SVG_OUTPUT $SVG_CHECKPOINTS $SVG_LOGS

#==============================================================================
# MODULE LOADING
#==============================================================================
module purge
module load gcc/11.2.0
module load cuda/11.7
module load openmpi/4.1.4
module load python/3.9
module load hdf5/1.12.2
module load cmake/3.23.1

#==============================================================================
# ENVIRONMENT SETUP
#==============================================================================
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH=/tmp/cuda_cache_${SLURM_JOB_ID}

# MPI settings
export OMPI_MCA_btl=^openib
export OMPI_MCA_btl_tcp_if_include=ib0
export OMPI_MCA_pml=ob1
export OMPI_MCA_coll=^tuned

# Python path
export PYTHONPATH=$SVG_HOME:$PYTHONPATH
export PYTHONUNBUFFERED=1

#==============================================================================
# PERFORMANCE TUNING
#==============================================================================
# CPU frequency scaling
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Memory allocation
export MALLOC_TRIM_THRESHOLD_=-1
export MALLOC_MMAP_MAX_=65536

#==============================================================================
# LAUNCH SIMULATION
#==============================================================================
echo "=============================================================================="
echo "SVG Simulation - BSC MareNostrum 5"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "MPI ranks: $SLURM_NTASKS"
echo "OpenMP threads: $OMP_NUM_THREADS"
echo "GPUs per node: 4"
echo "Start time: $(date)"
echo "=============================================================================="

# Run simulation
time srun python $SVG_HOME/svg_simulation_v3.0.py

#==============================================================================
# POST-PROCESSING
#==============================================================================
echo "=============================================================================="
echo "Starting post-processing"
echo "Time: $(date)"
echo "=============================================================================="

# Combine VTU files and analyze
python $SVG_HOME/svg_postprocess_v3.0.py --step=9999 --visualize --train-ai

#==============================================================================
# VALIDATION
#==============================================================================
echo "=============================================================================="
echo "Running validation against Simons Observatory data"
echo "Time: $(date)"
echo "=============================================================================="

python $SVG_HOME/validate_with_observations.py \
    --simulation=$SVG_OUTPUT \
    --observations=$SVG_DATA/simons_observatory_2025.h5 \
    --output=validation_report.json

#==============================================================================
# CLEANUP
#==============================================================================
echo "=============================================================================="
echo "Simulation completed"
echo "End time: $(date)"
echo "=============================================================================="

# Archive logs
cp $SVG_LOGS/svg_sim_${SLURM_JOB_ID}.out $SVG_OUTPUT/
cp $SVG_LOGS/svg_sim_${SLURM_JOB_ID}.err $SVG_OUTPUT/

# Create job completion marker
echo "$(date): Job $SLURM_JOB_ID completed" >> $SVG_LOBS/completed_jobs.log