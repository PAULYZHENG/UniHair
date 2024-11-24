pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install git+https://github.com/ashawkey/kiuikit
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/NVlabs/nvdiffrast/

pip install ./diff-gaussian-rasterization
pip install ./simple-knn

cd 3DDFA_V2
sh ./build.sh
cd ..

# cd xformers
# python setup.py install
# cd ..


mkdir ckpts
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ./ckpts/sam_vit_h_4b8939.pth

