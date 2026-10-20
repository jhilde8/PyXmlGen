#!/bin/bash
#SBATCH -J CGs.repack
#SBATCH -A phy157_dwf
#SBATCH -o %x-%j.out
#SBATCH -t 2:00:00
#SBATCH -p batch
##SBATCH -q debug
#SBATCH -N 160
#SBATCH --mem=0
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=jonas.hildebrand@uconn.edu

# Repack job: every par.CG<f>.repack.NN.<conf>.xml that gen_cg_repack.py wrote
# for the configurations in CONFS, all running at once. The XMLs already carry
# their trajectory, so nothing is substituted. -N above must be the node total
# gen_cg_repack.py prints (sruns x nodes per srun, summed over CONFS), and -t
# its longest estimate plus margin.
# Srun K runs on lines [K*NODES, (K+1)*NODES) of nodes.<jobid>.txt, named
# with srun -w: unlike -r, a step cannot slide onto another srun's nodes.

if [ ! -f select_gpu ]
then

cat << EOF > select_gpu
#!/bin/bash
export GPU_MAP=(0 1 2 3 7 6 5 4)
export NUMA_MAP=(3 3 1 1 2 2 0 0)
export GPU=\${GPU_MAP[\$SLURM_LOCALID]}
export NUMA=\${NUMA_MAP[\$SLURM_LOCALID]}
export HIP_VISIBLE_DEVICES=\$GPU
unset ROCR_VISIBLE_DEVICES
echo RANK \$SLURM_LOCALID using GPU \$GPU
exec numactl -m \$NUMA -N \$NUMA \$*
EOF

chmod +x ./select_gpu
fi

export MPICH_GPU_SUPPORT_ENABLED=1

export OMP_NUM_THREADS=7
export OMP_PROC_BIND=close
export OMP_PLACES=cores

#####################################################################################
# inputs
#####################################################################################

FLAVOR=s
CONFS=(1540)

VOL=64.64.64.128
DEV_MEM=16000
SHM=8192

case $FLAVOR in
    l) NODES=64; RANKS=512; MPI_GEOM=4.4.4.8; SPLIT="--batched-solver-split 2.2.2.4 --shm-mpi 0" ;;
    s|c) NODES=4; RANKS=32; MPI_GEOM=2.2.2.4; SPLIT="" ;;
esac
# Light: --shm-mpi 0 keeps on-node halos in Grid's shm. Through MPI, the split
# grid's on-node halos used MPICH GPU IPC and GPU-faulted after a few batches.

#####################################################################################
# paths
#####################################################################################

PREFIX=/lustre/orion/phy157/world-shared/jhilde/k2pipipbc/main_64I/CG
BIN=/lustre/orion/phy157/world-shared/jhilde/Hadrons_nb/install/bin/HadronsXmlRun

#####################################################################################

source /lustre/orion/phy157/world-shared/jhilde/Grid/build/sourceme-rocm7.2.sh
cd $PREFIX

scontrol show hostnames $SLURM_JOB_NODELIST > nodes.$SLURM_JOB_ID.txt

XMLS=()
for CONF in "${CONFS[@]}"
do
    XMLS+=($PREFIX/xml/CG$FLAVOR/par.CG$FLAVOR.repack.[0-9][0-9].$CONF.xml)
done

echo "$FLAVOR repack: ${#XMLS[@]} sruns x $NODES nodes" \
     "= $((${#XMLS[@]} * $NODES)) nodes, job has $SLURM_JOB_NUM_NODES"
date

K=0
for XML in "${XMLS[@]}"
do
    RUN=$(basename $XML .xml)
    RUN=${RUN#par.}
    LOG=log.$RUN.txt

    HOSTS=$(sed -n "$((K * NODES + 1)),$(((K + 1) * NODES))p" nodes.$SLURM_JOB_ID.txt | paste -sd,)
    echo "srun $K: $RUN on nodes $((K * NODES))-$(((K + 1) * NODES - 1)): $HOSTS" > $LOG
    srun -N $NODES -n $RANKS -w $HOSTS --cpus-per-task=7 ./select_gpu $BIN $XML --mem=0 --shm $SHM --device-mem $DEV_MEM --grid $VOL --mpi $MPI_GEOM $SPLIT >> $LOG 2>&1 &
    K=$((K + 1))
done

wait
date
