# Initialization

Run following commands on your compute instance:

```sh
# install R / python / cuda / cudnn
sudo apt-get install r-base python3 python3-pip p7zip-full
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update && sudo apt-get install cuda

# initialization of R
Rscript -e "install.packages('remotes'); remotes::install_github('wush978/pvm'); pvm::import.packages()"
Rscript train-fold.R

# initialization of python
pip3 install virtualenv

# clone this project
git clone https://github.com/wush978/KaggleQuora
# switch to directory
cd KaggleQuora
# submodule
git submodule init && git submodule update --init --recursive
# embedding files
7z x embeddings/glove.6B.50d.txt.7z && mv glove.6B.50d.txt embeddings/
# init python3 env
python3 -m virtualenv .
source bin/activate
pip3 install keras tensorflow-gpu nltk gensim h5py pandas nltk
# pip3 install google-compute-engine # required for GCE
```

## CUDNN

Please download cudnn 5.1 with cuda 8.0 for linux to your local machine. Then upload the file via:

```sh
gcloud beta compute scp ./cudnn-8.0-linux-x64-v5.1.tgz <machine name>:~/
```

Then run the following commands in your compute instance:

```sh
7z x cudnn-8.0-linux-x64-v5.1.tgz && tar -xf cudnn-8.0-linux-x64-v5.1.tar
```

Then manually install these files:

```sh
sudo cp cuda/include/* /usr/include/
sudo cp cuda/lib64/* /usr/lib/x86_64-linux-gnu/
```

# Test your environment

Under project root:

```sh
source bin/activate
python3 -c "import tensorflow"
```

Then we could add gpu to this machine
