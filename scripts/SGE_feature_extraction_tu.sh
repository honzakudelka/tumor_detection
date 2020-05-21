#!/bin/bash
#
#$ -S /bin/bash
#$ -t 12-111:1
#$ -tc 16
#$ -q fastjobs
#$ -N cam16_T_pex0
#$ -l h=!(cmpgrid-73)
#$ -e /datagrid/personal/herinjan/milearning/microscopy-python/sge_logs
#$ -o /datagrid/personal/herinjan/milearning/microscopy-python/sge_logs
#

VENV_ROOT=/datagrid/personal/herinjan/python/cv-venv
TARGET=/datagrid/personal/herinjan/Development/microscopy-io/compute_features.py
EXEC_ROOT=/datagrid/personal/herinjan/milearning/microscopy-python



study_root="/datagrid/Medical/microscopy/CAMELYON16/patches/patches_meta_tumor"
output_dir="feature_descriptors"

# task enumerated from 0..
i=$(expr $SGE_TASK_ID - 1)
id=$(printf '%03d' $i)
echo "Running job $SGE_TASK_ID" 
sleep 2s


lev=6
pext=0
sz="1024"

input_file="${study_root}/tumor_${id}_l${lev}_s${sz}.csv"

file_suffix="_l${lev}_s${sz}_pext_${pext}_bgcorr_mswav_stru_hist_txt_features.csv"
output_file="$EXEC_ROOT/${output_dir}/cam16_tumor_${id}_${file_suffix}"
log_file="$EXEC_ROOT/logfiles/tumor_${id}_rtime.log"

# do something only if there is input and no output
#if [ ! -e $output_file ]
#then

    if [ -f "$input_file" ];
    then
        echo "Processing: $input_file"
        # open subshell
        (
        # 1. Load venv
        . ${VENV_ROOT}/bin/activate
        export OMP_NUM_THREADS=6

        # 2. Execute script
        python $TARGET $input_file $output_file $pext 'stru' 'hist' 'text' > $log_file

        unset OMP_NUM_THREADS

        deactivate

        )
    else
        echo "Requested file $input_file doesn't exist"
    fi
#else
#    echo " Output file $output_file exists, skipping... " 
#fi

