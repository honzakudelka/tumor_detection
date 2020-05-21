#!/bin/bash

study_input_root="/datagrid/Medical/microscopy/petacc3"
tissue_mask_root="/datagrid/Medical/microscopy/petacc3/classification/ilastik_l6"
output_root="/datagrid/Medical/microscopy/petacc3/patches_metadata"

python_venv_root="/local/herinjan/virtenv/hpvenv"
target_script="/local/herinjan/Development/microscopy-io/patch_extraction.py"

p_size=512
p_lev=6

function export_patches () {
    echo "  ...exporting patches"

    subject=$(basename $1)
    out_subj=${subject%.*}
    input_dir=$2
    annot_file=$3
    stage_num=$4
    muc_group=$5
    
    fid=$(echo $out_subj | cut -d'_' -f2)
    nid=$(echo $out_subj | cut -d'_' -f4)
    
    csv_out="${output_root}/${muc_group}/${out_subj}_l${p_lev}_s${p_size}.csv"
    png_out="${output_root}/${muc_group}/${out_subj}_l${p_lev}_s${p_size}.png"
    tu_mask="${study_input_root}/$input_dir/${subject}.$annot_file"
    
    echo $tu_mask
    
    tu_mask_param=""
    
    if [ -f $csv_out ]; then
        echo " >> CSV Output ${csv_out} already there"
        return
    fi
    
    if [ -f $tu_mask ]; then
        tu_mask_param="-t ${tu_mask}"
    fi

    in_file=${study_input_root}/${input_dir}/$subject
    tissue_mask=${tissue_mask_root}/Segm_stage${stage_num}_NON_${out_subj}_l${p_lev}.png
    
    (
        . ${python_venv_root}/bin/activate
        python $target_script -i $in_file -m $tissue_mask -p $p_size -l $p_lev $tu_mask_param --csv $csv_out --preview $png_out --min-coverage 0.8
        
        deactivate
    )
    
}

# helper function for parallel processing
MAX_PARALLEL_JOBS=6
function limit_cpu {
    while [ `jobs | wc -l` -ge $MAX_PARALLEL_JOBS ]
    do
      sleep 10
    done
}


while IFS=';' read fname stage tumor dir annot target
do
    
    f_basename=${fname%.*}
    echo " Extracting patches for $f_basename (Type: $tumor)"
    
    if [ "${tumor}" == "NM" ];
    then
        limit_cpu; export_patches $fname $dir $annot $stage "nonmucinous" &
    else
        limit_cpu; export_patches $fname $dir $annot $stage "mucinous" &
    fi
    
done < $1
