
# Notes 
 - Only seems to work with conda env, think this could be due to it using a newer python version
```
conda activate yolov5_obb
```
 - Refer to CUDA_VERSION_README.md to setup appropriate cuda version
 - Had to make modifications to utils/nms_rotated/src/poly_nms_cuda.cu to get it to work with torch 1.13
    - Torch migrated THC into ATen but this repo still used THC functions. Replaced with equivalent ATen functinons and remove THC state - presumably handled automatically by ATen.
 - must pip install wandb (repo will run without installation) in order to used wandb logging.

# Training
python train.py --device 0 --data "./data/synth_cow_obb.yaml" --epochs 1000 --batch-size 8 --img 2048 --hyp "data/hyps/obb/hyp.my_hyps.yaml" --workers 8 --entity aulrichsen
python train.py --device 0 --data "./data/synth_cow_obb_2.yaml" --epochs 1000 --batch-size 32 --img 1024 --hyp "data/hyps/obb/hyp.finetune_dota_CloseAug.yaml" --workers 8 --entity aulrichsen


```
python train.py --device 0 --data "./data/padfilt_cow_obb.yaml" --epochs 1000 --batch-size 32 --img 1024 --hyp "data/hyps/obb/hyp.finetune_dota_CloseAug.yaml" --workers 8 --entity aulrichsen
```
```
python train.py --device 0 --data "./data/padded_cow_data_obb.yaml" --epochs 1000 --batch-size 8 --img 2048 --hyp "data/hyps/obb/hyp.my_hyps.yaml" --workers 8 --entity aulrichsen
```

# Detect
```
python detect.py --weights 'runs/train/exp15/weights/best.pt' --source 'dataset/cow_obb_padded/test/images' --imgs 1024 --device 0 --visualize --hide-labels
```
```
python detect.py --weights 'runs/train/exp36/weights/best.pt' --source 'dataset/cow_obb_filtered_padded/test/images' --imgs 1024 --device 0 --hide-labels
```
```
python detect.py --weights 'runs/train/exp18/weights/best.pt' --source 'dataset/cow_obb_padded/test/images' --imgs 2048 --device 0 --hide-labels
```
```
python detect.py --weights 'runs/train/exp27/weights/best.pt' --source 'dataset/cow_obb_filtered_padded/test/images' --imgs 2048 --device 0 --hide-labels
python detect.py --weights 'weights/cow_obb_best_weights.pt' --source 'dataset/barn_videos' --imgs 1024 --device 0 --hide-labels
```