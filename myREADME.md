
# Notes 
 - Only seems to work with conda env, think this could be due to it using a newer python version
 - Refer to CUDA_VERSION_README.md to setup appropriate cuda version
 - Had to make modifications to utils/nms_rotated/src/poly_nms_cuda.cu to get it to work with torch 1.13
    - Torch migrated THC into ATen but this repo still used THC functions. Replaced with equivalent ATen functinons and remove THC state - presumably handled automatically by ATen.


# Training
```
python train.py --device 0 --data "./data/padded_cow_data_obb.yaml" --epochs 200 --batch-size 32 --img 1024 --hyp "data/hyps/obb/hyp.finetune_dota_CloseAug.yaml" --workers 8 --entity aulrichsen
```