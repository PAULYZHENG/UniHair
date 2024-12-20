# UniHair

This repository contains the official implementation for [Towards Unified 3D Hair Reconstruction from Single-View Portraits](https://arxiv.org/abs/2409.16863).

### [Project Page](https://unihair24.github.io/) | [Arxiv](https://arxiv.org/abs/2409.16863)


[unihair](https://github.com/user-attachments/assets/838be752-b3e9-46d5-9bb9-91712ab7966e)



### News

- TODO: release the rendering data.
- 2024.11.24: release the code!

## Install

```bash
git clone --recursive https://github.com/PAULYZHENG/UniHair.git
cd UniHair

# change CUDA to 11.3 before create the env
conda env create -f environment.yml
conda activate unihair

sh setup.sh
```

Tested on Ubuntu 22 with torch 1.12.1 & CUDA 11.3 on 3090/4090.

## Usage
Put images you want to test in data/img/
```bash
sh scripts/run_hairalign.sh
sh scripts/run_unihair.sh
```
Find your results in data/logs/

## Acknowledgement

This repository is based on some excellent works, many thanks to all the authors!

- [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [zero123](https://github.com/cvlab-columbia/zero123) and [zero123-hf](https://github.com/kxhit/zero123-hf)
- [3DFFA_V2](https://github.com/cleardusk/3DDFA_V2)
- [SAM](https://github.com/facebookresearch/segment-anything)


## Citation

```
@inproceedings{zheng2024towards,
  title={Towards Unified 3D Hair Reconstruction from Single-View Portraits},
  author={Zheng, Yujian and Qiu, Yuda and Jin, Leyang and Ma, Chongyang and Huang, Haibin and Zhang, Di and Wan, Pengfei and Han, Xiaoguang},
  booktitle={SIGGRAPH Asia 2024 Conference Papers},
  pages={1--11},
  year={2024}
}
```
