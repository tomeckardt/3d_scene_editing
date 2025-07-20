# 3D Scene Editing

## Installation
IMPORTANT: Use the `--recursive` option to clone this repository.
```
git clone --recursive https://github.com/tomeckardt/3d_scene_editing.git
```
Create a conda environment or similar. Then run:
```
pip install -r requirements.txt
```

### Setting Up BlenderProc
Navigate to `submodules/BlenderProc-3DFront`and link all corresponding 3D-FRONT directories to the following locations:
```
examples/datasets/front_3d_with_improved_mat/3D-FRONT
examples/datasets/front_3d_with_improved_mat/3D-FRONT-texture
examples/datasets/front_3d_with_improved_mat/3D-FUTURE-model
```

## Usage
### Scene Modification and Rendering
Both modification and rendering are performed by `src/3dront_utils/render_scene_and_modify.py`. Example usage:
```
python3 render_scene_and_modify.py
```

TODO: Provide example usage, move it in main folder.

### Training
```
torchrun --nproc_per_node=1 train.py \
        --train_dataset "1000 @ ModFront3DV3(split='train', ROOT='../../renderings', aug_crop=16, resolution=224, transform=ColorJitter)" \
        --test_dataset "100 @ ModFront3DV3(split='test', ROOT='../../renderings', resolution=224, seed=777)" \
        --model "EditSceneModel.from_pretrained('naver/DUSt3R_ViTLarge_BaseDecoder_224_linear')" \
        --freeze "encoder" \
        --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
        --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
        --lr 0.0002 --min_lr 5e-07 --warmup_epochs 1 --epochs 40 --batch_size 4 --accum_iter 4 \
        --save_freq 4 --keep_freq 10 --eval_freq 4 \
        --output_dir "checkpoints/dust3r_demo_224_mod"
```
TODO: Move train.py in main folder, update the command accordingly.