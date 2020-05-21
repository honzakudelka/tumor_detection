#!/bin/bash
#
#$ -S /bin/bash
#$ -t 1-130
#$ -tc 40
#$ -q fastjobs
#$ -N bgcorr_feat_whist
#$ -e /datagrid/personal/herinjan/milearning/microscopy-python/sge_logs
#$ -o /datagrid/personal/herinjan/milearning/microscopy-python/sge_logs
#

VENV_ROOT=/datagrid/personal/herinjan/python/cv-venv
PY_ROOT=/datagrid/personal/herinjan/Development/microscopy-io
TARGET=${PY_ROOT}/compute_features.py
EXEC_ROOT=/datagrid/personal/herinjan/milearning/microscopy-python

study_root="/datagrid/Medical/microscopy/petacc3/patches_metadata/batch_2"
output_dir="/datagrid/Medical/microscopy/petacc3/patches_metadata/feature_descriptors"
file_list="filtered_files.txt"


# task enumerated from 0..
i=$(expr $SGE_TASK_ID - 1)
id=$(printf '%03d' $i)
echo "Running job $SGE_TASK_ID" 
sleep 2s



sz="1024"
lev="6"
pe_level="1"

input_file=$(sed "${i}q;d" "${study_root}/${file_list}")
input_path="${study_root}/$input_file"

fname=${input_file%.*}

FILE_SUFFIX="pext_${pe_level}_bgcorr_mswvlt_hehist_a_stru_his_txt_features.csv"
output_file="${output_dir}/${fname}_${FILE_SUFFIX}"


log_file="$EXEC_ROOT/logfiles/${fname}_rtime.log"

# do something only if there is input and no output
#if [ ! -e $output_file ]
#then

    if [ -f "$input_path" ];
    then
        echo "Processing: $input_path"
        # open subshell
        (
        # 1. Load venv
        . ${VENV_ROOT}/bin/activate
        export OMP_NUM_THREADS=6
       # export PYTHONPATH=${PY_ROOT}:${PY_ROOT}/utils:$PYTHONPATH
       # export LD_LIBRARY_PATH=${PY_ROOT}/utils:$LD_LIBRARY_PATH

        # 2. Execute script
        python $TARGET $input_path $output_file $pe_level 'stru' 'hist' 'text' > $log_file

        unset OMP_NUM_THREADS

        deactivate

        )
    else
        echo "Requested file $input_path doesn't exist"
    fi
#else
#    echo " Output file $output_file exists, skipping... " 
#fi

