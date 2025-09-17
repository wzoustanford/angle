#!/usr/bin/bash 

conda env create -f moore_metaworld.yml
conda activate moore_metaworld
git clone https://github.com/Farama-Foundation/Metaworld.git
cd Metaworld
git checkout a98086a
pip install -e .
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xvf mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco/
mv mujoco210 ~/.mujoco/
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
source ~/.bashrc
conda activate moore_metaworld
sudo apt-get install -y libglew-dev
sudo apt-get install -y patchelf

#wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
#git clone https://github.com/wzoustanford/angle.git
#cd angle/RL/MOORE
#git checkout 0e7a1bc2c6bf89219a3ba3bb5e2ca8db60480c02
