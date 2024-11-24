# UniHair

This repository contains the official implementation for [Towards Unified 3D Hair Reconstruction from Single-View Portraits](https://arxiv.org/abs/2409.16863).

### [Project Page](https://unihair24.github.io/) | [Arxiv](https://arxiv.org/abs/2409.16863)

<p style="text-align: center;">
            <iframe width="970" height="550" src="assets/fastforward.mp4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></p>


### News

- TODO: release the rendering data.
- 2024.11.24: release the code!

## Install

```bash
git clone --recursive https://github.com/PAULYZHENG/UniHair.git
cd UniHair

sh setup.sh
```

Tested on Ubuntu 22 with torch 1.12.1 & CUDA 11.3 on 3090/4090.

## Usage
Put images you want to test in data/img/
```bash
sh scripts/run_hairalign.sh
sh scripts/run_unihair.sh
```

## Acknowledgement

This repository is based on some excellent works, many thanks to all the authors!

- [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [3DFFA_V2](https://github.com/cleardusk/3DDFA_V2)
- [SAM](https://github.com/facebookresearch/segment-anything)


## Citation

```
@article{zheng2024towards,
  title={Towards Unified 3D Hair Reconstruction from Single-View Portraits},
  author={Zheng, Yujian and Qiu, Yuda and Jin, Leyang and Ma, Chongyang and Huang, Haibin and Zhang, Di and Wan, Pengfei and Han, Xiaoguang},
  journal={arXiv preprint arXiv:2409.16863},
  year={2024}
}
```
