## Installation
First you need to clone the repo:
```
git clone --recursive https://github.com/Kickblip/handover-toolkit
cd handover-toolkit
```

We recommend creating a virtual environment. You can use venv:
```bash
python3.10 -m venv .htk
source .htk/bin/activate
```

or alternatively conda:
```bash
conda create --name htk python=3.10
conda activate htk
```

Then, you can install the rest of the dependencies. This is for CUDA 11.7, but you can adapt accordingly:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -e .[all]
pip install -v -e third-party/ViTPose
```

Resolving numpy version issues
```bash
pip uninstall -y opencv-python numpy
pip install "numpy==1.26.4"
pip install "opencv-python==4.11.0.86"
```

From here download the same trained models as HaMeR:
```bash
bash fetch_demo_data.sh
```

Besides these files, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de) and register to get access to the downloads section.  We only require the right hand model. You need to put `MANO_RIGHT.pkl` under the `_DATA/data/mano` folder.

## CUDA

This code relies on CUDA. Here's the link to install the CUDA Toolkit 11.7 for our machine:

https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local