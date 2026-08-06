#!/bin/sh
#SBATCH --mem=224G
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
#SBATCH --qos=normal
#SBATCH --time=8-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fadi.boutros@igd.fraunhofer.de
#SBATCH --cpus-per-gpu=16
#SBATCH --nodelist=ampere3


PROJECT_DIR="/igd/a1/homestud/fboutros/Condtional_IDifFace/IDPerturb/"
#Data_DIR="/igd/a1/homestud/fboutros/Condtional_IDifFace/data/FFHQ.zip"

EXECUTION_COMMAND="/bin/sh -c cd IDiff-Face; python sample.py"
TO="/opt/cache/fboutros/job${SLURM_JOB_ID}/IDPerturb"
OUT="/igd/a1/homestud/fboutros/Condtional_IDifFace/IDPerturb"

echo "> Dispatching Slurm Job ID-${SLURM_JOB_ID} Run on Node ${SLURM_JOB_NODELIST} using"
echo "....Project Directory: ${PROJECT_DIR}"
echo "....Execution Command: ${EXECUTION_COMMAND}"

mkdir -p $OUT
echo "....Output Folder ${OUT}"
mkdir -p $TO
echo "....Target Folder ${TO}"

echo "> Copy and unzip project"
cp -r "${PROJECT_DIR}/." $TO
#cp -r "${Data_DIR}/." $TO

# Unzip FFHQ Folder
#unzip -q "${TO}/FFHQ.zip" -d $TO

IMG=container-registry.gitlab.cc-asp.fraunhofer.de/fboutros/docker/pytorchdocker:latest

echo "....Image ${IMG}"
TO="/opt/cache/fboutros/job${SLURM_JOB_ID}"

rootless-docker run --gpus=all --shm-size 32g --quiet -v ${TO}:/workspace -v ${OUT}:/output -w /workspace --rm $IMG /bin/sh -c ./IDPerturb/create_database.sh
echo "....Run docker"

# echo "> Archiving the output files"
# cd $TO
# zip -r 2024-06-05_output_files-FFHQ.zip outputs # checkpoints samples main.log

# echo "> Copying the archived files to the output directory"
# cp output_files.zip $OUT


# rm -r $TO
