# RDPNet

IEEE TIP 2021: Regularized Densely-connected Pyramid Network for Salient Instance Segmentation

PyTorch Training and testing code are available. We have achieved SOTA performance on the salient instance segmentation (SIS) task.

If you run into any problems or feel any difficulties to run this code, do not hesitate to leave issues in this repository.

My e-mail is: wuyuhuan @ mail.nankai (dot) edu.cn

### Citations

If you are using the code/model/data provided here in a publication, please consider citing:

    @article{wu2021regularized,
      title={Regularized Densely-connected Pyramid Network for Salient Instance Segmentation},
      author={Wu, Yu-Huan and Liu, Yun and Zhang, Le and Gao, Wang and Cheng, Ming-Ming},
      journal={IEEE Transactions on Image Processing},
      year={2021},
      publisher={IEEE}
    }

### Requirements

* PyTorch 1.1/1.0.1, Torchvision 0.2.2.post3, CUDA 9.0/10.0/10.1, apex
* Validated on Ubuntu 16.04/18.04, PyTorch 1.1/1.0.1, CUDA 9.0/10.0/10.1, NVIDIA TITAN Xp

### Installing

Please check [INSTALL.md](INSTALL.md).

Note: we have provided an early tested apex version (url: [here](https://github.com/NVIDIA/apex/tree/f2b3a62c8941027253b2decba96ba099f611387e)) and place
it in our root folder (./apex/). You can also try other apex versions, which are not tested by us.

### Data

Before training/testing our network, please download the data: [[Google Drive, 0.7G]](https://drive.google.com/file/d/1FR7K6gdIStio-QEimxGqN-VHgGrIJMN6), [[Baidu Yun, yhwu]](https://pan.baidu.com/s/1s5tdYinlwVKTg09mA_l-Fg).

The above zip file contains data of the ISOD and SOC dataset.

Note: if you are blocked by Google services, you can contact me via e-mail and I will send you a copy of data and model weights.

We have processed the data to json format so you can use them without any preprocessing steps. 
After completion of downloading, extract the data and put them to `./datasets/` folder.
Then, the `./datasets/` folder should contain two folders: `isod/, soc/`.

### Train

It is very simple to train our network. We have prepared a script to run the training step.
You can at first train our ResNet-50-based network on the ISOD dataset:

```
cd scripts
bash ./train_isod.sh
```

The training step should cost less than `1 hour` for single GTX 1080Ti or TITAN Xp. The script file will also store the network code, config file, log, and model weights.

We also provide ResNet-101 and ResNeXt-101 training scripts, and they are all in the `scripts` folder.

The default training code is for single gpu training since the training time is very low. You can also try multi gpus training by replacing 
`--nproc_per_node=1 \` with `--nproc_per_node=2 \` for 2-gpu training.


### Test

It is also very simple to test our network. First you need to download the model weights:

* ResNet-50 (ISOD dataset): [[Google Drive, 0.14G]](https://drive.google.com/file/d/1P9HnPbeHKL_1EzKOcVYhjiYXyUiKCaYP/view?usp=sharing), [[Baidu Yun, yhwu]](https://pan.baidu.com/s/1pWCp6lwmEQW-07WGLl_zgw)
* ResNet-50 (SOC dataset): [[Google Drive, 0.14G]](https://drive.google.com/file/d/1faQeoplwGPcWoMzrTHKs2YX9BYjUaCwD/view?usp=sharing), [[Baidu Yun, yhwu]](https://pan.baidu.com/s/1gN2a5Nd6eNBtd774uHk8XQ)

Taking the test on the ISOD dataset for example: 
1. Download [the ISOD trained model weights](), put it to `model_zoo/` folder.
2. cd the `scripts` folder, then run `bash test_isod.sh`.
3. Testing step usually costs less than a minute. We use the official `cocoapi` for evaluation.


### Visualize

We provide a simple python script to visualize the result: `demo/visualize.py`.

1. Be sure that you have downloaded the ISOD pretrained weights [[Google Drive, 0.14G]](https://drive.google.com/file/d/1P9HnPbeHKL_1EzKOcVYhjiYXyUiKCaYP/view?usp=sharing).
2. Put images to the `demo/examples/` folder. I have prepared some images in this paper so do not worry that you have no images.
3. cd demo, run `python visualize.py`


### Acknowlogdement

This repository is built under the help of the following three projects for academic use only:

* [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

* [FCOS](https://github.com/tianzhi0549/FCOS)

* [S4Net](https://github.com/RuochenFan/S4Net)
