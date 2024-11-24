# list=''249566dc08ab3f5d504f04a2e946ec52' '
# list=''70cab1d12f21dee4ce66a9b2acab5ca0' '
list=''cf99491acd8b4905ce9041ea469c7056' '

gpu_id=0

iters_sds=1000
iters_refine=600
iters_deblur=1000

batch_size=4

img_dir="./data/inputs"
source_img_path="./data/alignment/aligned_img/"
logdir="./data/logs/results"
deblur_zero123_path="./priors/hairEnhancer"

ply_name="_coarse.ply"
refine_ply_name="_refine.ply"

density_start_iter=50
density_end_iter=500
density_end_iter_deblur=500
densification_interval=200
opacity_reset_interval=301
densify_grad_threshold=0.0002

for item in $list; do
    echo $item
    python img_process/process_hair_real.py $source_img_path$item.png --size 512 --border_ratio 0.0  --out_dir $img_dir
    
    #coarse gaussian initialization
    CUDA_VISIBLE_DEVICES=$gpu_id python main_coarse.py --config configs/image.yaml input=$img_dir/$item-rgba.png save_path=$item outdir=$logdir iters=$iters_sds batch_size=$batch_size

    #view-wise refinement
    CUDA_VISIBLE_DEVICES=$gpu_id python main_refine.py --config configs/image.yaml use_vgg=True input=$img_dir/$item-rgba.png save_path=$item outdir=$logdir iters=$iters_refine batch_size=$batch_size load=$logdir/$item/$item$ply_name density_start_iter=$density_start_iter density_end_iter=$density_end_iter densification_interval=$densification_interval opacity_reset_interval=$opacity_reset_interval densify_grad_threshold=$densify_grad_threshold

    #pixel-wise refinement
    CUDA_VISIBLE_DEVICES=$gpu_id python main_enhance.py --config configs/image.yaml ref_size=512 zero123_path=$deblur_zero123_path use_vgg=True input=$img_dir/$item-rgba.png save_path=$item outdir=$logdir iters=$iters_deblur batch_size=$batch_size load=$logdir/$item/$item$refine_ply_name densification_interval=$densification_interval densify_grad_threshold=$densify_grad_threshold density_start_iter=$density_start_iter density_end_iter=$density_end_iter_deblur

    python to_video/to_video_compare_all.py --split=$item --output=$logdir
    python to_video/to_video_result.py --split=$item --output=$logdir

    # python -m img_process.process_hair_real $source_img_path$item.png --size 512 --border_ratio 0.0  --out_dir $img_dir
    
    # #coarse gaussian initialization
    # CUDA_VISIBLE_DEVICES=$gpu_id python -m main.main_coarse --config configs/image.yaml input=$img_dir/$item-rgba.png save_path=$item outdir=$logdir iters=$iters_sds batch_size=$batch_size

    # #view-wise refinement
    # CUDA_VISIBLE_DEVICES=$gpu_id python -m main.main_refine --config configs/image.yaml use_vgg=True input=$img_dir/$item-rgba.png save_path=$item outdir=$logdir iters=$iters_refine batch_size=$batch_size load=$logdir/$item/$item$ply_name density_start_iter=$density_start_iter density_end_iter=$density_end_iter densification_interval=$densification_interval opacity_reset_interval=$opacity_reset_interval densify_grad_threshold=$densify_grad_threshold

    # #pixel-wise refinement
    # CUDA_VISIBLE_DEVICES=$gpu_id python -m main.main_enhance --config configs/image.yaml ref_size=512 zero123_path=$deblur_zero123_path use_vgg=True input=$img_dir/$item-rgba.png save_path=$item outdir=$logdir iters=$iters_deblur batch_size=$batch_size load=$logdir/$item/$item$refine_ply_name densification_interval=$densification_interval densify_grad_threshold=$densify_grad_threshold density_start_iter=$density_start_iter density_end_iter=$density_end_iter_deblur

    # python -m to_video.to_video_compare_all --split=$item --output=$logdir
    # python -m to_video.to_video_result --split=$item --output=$logdir

done