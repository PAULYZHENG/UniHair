conda env create -f environment.yml
conda activate unihair

pip install ./diff-gaussian-rasterization
pip install ./simple-knn

cd 3DDFA_V2
sh ./build.sh
cd ..

cd xformers
sh ./build.sh
cd ..

mkdir ckpts
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ./ckpts/sam_vit_h_4b8939.pth

