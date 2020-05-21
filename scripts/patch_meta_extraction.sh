#!/bin/bash

study_input_root="/datagrid/Medical/microscopy/CAMELYON16"
output_root="/datagrid/Medical/microscopy/CAMELYON16/patches"
python_venv_root="/local/herinjan/virtenv/hpvenv"
target_script="/local/herinjan/Development/microscopy-io/patch_extraction.py"

p_size=1024
p_lev=6

function export_ndpi_patches () {
    echo "  ...exporting patches"

    out_prefix="/datagrid/Medical/microscopy/petacc3/patches_metadata/batch_2"
    subject=$(basename $1)
    out_subj=${subject/\./_}
    
    csv_out="$out_prefix/${out_subj}_l${p_lev}_s${p_size}.csv"
    png_out="$out_prefix/${out_subj}_l${p_lev}_s${p_size}.png"
    
    if [ -f $csv_out ]; then
        return
    fi
    
    
    (
        . ${python_venv_root}/bin/activate
        python $target_script -i $1 -m $2 -p $p_size -l $p_lev --csv $csv_out --preview $png_out
        
        deactivate
    )
}

# helper function for parallel processing
MAX_PARALLEL_JOBS=4
function limit_cpu {
    while [ `jobs | wc -l` -ge $MAX_PARALLEL_JOBS ]
    do
      sleep 20
    done
}


study_root="/datagrid/Medical/microscopy/petacc3/batch_2"
find $study_root -name "*\.ndpi" | while read infile;
do

    echo "Processing file: $infile"
    image=$(basename $infile)
    imname=${image%.*}
    mask_file="/datagrid/Medical/microscopy/petacc3/classification/ilastik_l6/Segm_stage1_NON_${imname}_l6.png"
    mask_file2="/datagrid/Medical/microscopy/petacc3/classification/ilastik_l6/Segm_stage4_NON_${imname}_l6.png"
    
    if [ -f "${mask_file}" ];
    then
        limit_cpu; export_ndpi_patches $infile $mask_file 
    elif [ -f "${mask_file2}" ];
    then
        limit_cpu; export_ndpi_patches $infile $mask_file2 
        
    else
        echo "  > no segmentation mask found for ${image}"
    fi


done
