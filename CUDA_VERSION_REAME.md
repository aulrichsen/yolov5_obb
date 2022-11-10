# How to change CUDA version to suit this repo

## Notes
 - If this is done inside a venv, it will not be pointed to outside the venv
    - Download should still be there
 - This example is cuda 11.7 but use your desired version

## Helpful commands
List cuda installations
```
apt list --installed | grep cuda
```
Find cuda location
```
whereis cuda
```
Linux version
```
lsb_release -a
```
Display installed cuda toolkit versions
```
ls /usr/local
```

## Check Version
```
nvcc -V
```
```
nvidia-smi
```
 - nvcc -V most important

## Download appropriate version
Install desired cuda toolkit version
https://developer.nvidia.com/cuda-toolkit-archive
 - Note: if downgrading, must specify cuda version on final command, otherwis most recent version will be used
 e.g.
```
sudo apt-get -y install cuda-11.7
```

Cuda driver install (probably not necessary)
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
 - Can likely skip to section 4

## Point to correct version
Set path
```
export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
```
Change environment variable
```
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
 - May need to reboot after this
 - Validate by checking version