#!/bin/bash
#SBATCH -J CGs
#SBATCH -A phy157_dwf
#SBATCH -o %x-%j.out
#SBATCH -t 4:30:00
#SBATCH -p batch
##SBATCH -q debug
#SBATCH -N 160
#SBATCH --mem=0
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=jonas.hildebrand@uconn.edu

# One job: every srun of one flavour (par.CG<f>.NN.xml from gen_cg.py), for
# every configuration in CONFS, all running at once. -N above must be
#   nodes per srun x number of sruns x number of configurations
# per configuration: l 3 x 64 = 192, s 40 x 4 = 160, c 27 x 4 = 108.
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
OUT_DIR=/lustre/orion/phy157/proj-shared/phy157_dwf/jhilde/main_64I/vw

XML_DIR=$PREFIX/xml/CG$FLAVOR

# Templates only: the per-configuration copies (par.CGs.00.1540.xml) do not
# match this pattern.
TEMPLATES=($XML_DIR/par.CG$FLAVOR.[0-9][0-9].xml)

mkdir -p $OUT_DIR/sloppy $OUT_DIR/exact
lfs setstripe -c 16 -S 4M -p capacity $OUT_DIR/sloppy
lfs setstripe -c 16 -S 4M -p capacity $OUT_DIR/exact

#####################################################################################

source /lustre/orion/phy157/world-shared/jhilde/Grid/build/sourceme-rocm7.2.sh
cd $PREFIX

scontrol show hostnames $SLURM_JOB_NODELIST > nodes.$SLURM_JOB_ID.txt

echo "$FLAVOR: ${#TEMPLATES[@]} sruns x ${#CONFS[@]} configurations x $NODES nodes" \
     "= $((${#TEMPLATES[@]} * ${#CONFS[@]} * $NODES)) nodes, job has $SLURM_JOB_NUM_NODES"
date

K=0
for CONF in "${CONFS[@]}"
do
    for TMP in "${TEMPLATES[@]}"
    do
        RUN=$(basename $TMP .xml)
        RUN=${RUN#par.}
        XML=$XML_DIR/par.$RUN.$CONF.xml
        LOG=log.$RUN.$CONF.txt

        sed "s/TRAJ_START/$CONF/g; s/TRAJ_END/$(($CONF + 1))/g" $TMP > $XML

        HOSTS=$(sed -n "$((K * NODES + 1)),$(((K + 1) * NODES))p" nodes.$SLURM_JOB_ID.txt | paste -sd,)
        echo "srun $K: $RUN conf $CONF on nodes $((K * NODES))-$(((K + 1) * NODES - 1)): $HOSTS" > $LOG
        srun -N $NODES -n $RANKS -w $HOSTS --cpus-per-task=7 ./select_gpu $BIN $XML --mem=0 --shm $SHM --device-mem $DEV_MEM --grid $VOL --mpi $MPI_GEOM $SPLIT >> $LOG 2>&1 &
        K=$((K + 1))
    done
done

wait
date
