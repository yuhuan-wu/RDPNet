# RDPNet

[IEEE TIP 2021: Regularized Densely-connected Pyramid Network for Salient Instance Segmentation](https://ieeexplore.ieee.org/document/9382868)

PyTorch training and testing code are available. We have achieved SOTA performance on the salient instance segmentation (SIS) task.

If you run into any problems or feel any difficulties to run this code, do not hesitate to leave issues in this repository.

My e-mail is: wuyuhuan @ mail.nankai (dot) edu.cn

[[Official Ver.]](https://ieeexplore.ieee.org/document/9382868)
[[PDF]](https://mmcheng.net/wp-content/uploads/2021/03/21TIPInstSal.pdf)

:fire: News! We will add pretrained models on the newly released COME15K dataset and corresponding results on its test set.

### Requirements

* PyTorch 1.1/1.0.1, Torchvision 0.2.2.post3, CUDA 9.0/10.0/10.1, apex
* Validated on Ubuntu 16.04/18.04, PyTorch 1.1/1.0.1, CUDA 9.0/10.0/10.1, NVIDIA TITAN Xp

### Installing

Please check [INSTALL.md](INSTALL.md).

Note: we have provided an early tested **apex** version (url: [here](https://github.com/NVIDIA/apex/tree/f2b3a62c8941027253b2decba96ba099f611387e)) and place
it in our root folder (./apex/). You can also try other apex versions, which are not tested by us.

### Data

Before training/testing our network, please download the data: 

* ISOD and SOC datasets: [[Google Drive, 0.7G]](https://drive.google.com/file/d/1FR7K6gdIStio-QEimxGqN-VHgGrIJMN6), [[Baidu Yun, yhwu]](https://pan.baidu.com/s/1s5tdYinlwVKTg09mA_l-Fg). The above zip file contains data of the ISOD and SOC dataset.

* COME15K datasets: First download original data on the [official site](https://github.com/JingZhang617/cascaded_rgbd_sod). Then please downloaed [json format annotations](https://drive.google.com/drive/folders/1-nLBBxkTuGR_-RHjIpl2PJtwA3rB3IRf?usp=sharing) made by me. Extract them to `datasets/COME15K/`.

Note: if you are blocked by Google and Baidu services, you can contact me via e-mail and I will send you a copy of data and model weights.

We have processed the data to json format so you can use them without any preprocessing steps. 
After completion of downloading, extract the data and put them to `./datasets/` folder.
Then, the `./datasets/` folder should contain two folders: `isod/, soc/, COME15K/`.

### Train

It is very simple to train our network. We have prepared a script to run the training step.
You can at first train our ResNet-50-based network on the ISOD dataset:

```
cd scripts
bash ./train_isod.sh
```

The training step should cost less than `1 hour` for single GTX 1080Ti or TITAN Xp. This script will also store the network code, config file, log, and model weights.

We also provide ResNet-101 and ResNeXt-101 training scripts, and they are all in the `scripts` folder.

The default training code is for single gpu training since the training time is very low. You can also try multi gpus training by replacing 
`--nproc_per_node=1 \` with `--nproc_per_node=2 \` for 2-gpu training.


### Test / Evaluation / Results

It is also very simple to test our network. First you need to download the model weights:

* ResNet-50 (ISOD dataset): [[Google Drive, 0.14G]](https://drive.google.com/file/d/1P9HnPbeHKL_1EzKOcVYhjiYXyUiKCaYP/view?usp=sharing), [[Baidu Yun, yhwu]](https://pan.baidu.com/s/1pWCp6lwmEQW-07WGLl_zgw)
* ResNet-50 (SOC dataset): [[Google Drive, 0.14G]](https://drive.google.com/file/d/1faQeoplwGPcWoMzrTHKs2YX9BYjUaCwD/view?usp=sharing), [[Baidu Yun, yhwu]](https://pan.baidu.com/s/1gN2a5Nd6eNBtd774uHk8XQ)

Taking the test on the ISOD dataset for example: 
1. Download [the ISOD trained model weights](), put it to `model_zoo/` folder.
2. cd the `scripts` folder, then run `bash test_isod.sh`.
3. Testing step usually costs less than a minute. We use the official `cocoapi` for evaluation.

:fire: For the COME-E set of the COME15K dataset, we achieve `43.1% AP and 67.3% AP@0.5` trained on the COME15K training set. 
You can simply reproduce the result given the training config `configs/r50-come.yaml`.

**Note1**: We strongly recommend to use `cocoapi` to evaluate the performance. Such evaluation is also automatically done with the testing process.

**Note2**: Default cocoapi evaluation outputs AP, AP50, AP75 peformance. To output the score of AP70, you need to change the `cocoeval.py` in cocoapi.
See [**changes**](https://github.com/yuhuan-wu/cocoapi/commit/143563fe819d47080aabe1b5d6d4bb85669b8844#) in this commitment:

````
BEFORE: stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
AFTER:  stats[2] = _summarize(1, iouThr=.70, maxDets=self.params.maxDets[2])
````

**Note3**: If you are not familiar with the evalutation metric `AP, AP50, AP75`, you can refer to the introduction website [here](https://cocodataset.org/#detection-eval). Our official paper also introduces them in the `Experiments` section.

### Visualize

We provide a simple python script to visualize the result: `demo/visualize.py`.

1. Be sure that you have downloaded the ISOD pretrained weights [[Google Drive, 0.14G]](https://drive.google.com/file/d/1P9HnPbeHKL_1EzKOcVYhjiYXyUiKCaYP/view?usp=sharing).
2. Put images to the `demo/examples/` folder. I have prepared some images in this paper so do not worry that you have no images.
3. cd demo, run `python visualize.py`
4. Visualized images are generated in the same folder. You can change the target folder in `visualize.py`.

### TODO

1. Release the weights for **real-world applications**
2. Add [Jittor](https://github.com/Jittor/jittor) implementation
3. Train with the enhanced base detector (FCOS TPAMI version) for better performance. Currently the base detector is the FCOS conference version with a bit lower performance.
4. Add results with the [P2T](https://arxiv.org/abs/2106.12011) transformer backbone.

### Other Tips

I am free to answer your question if you are interested in `salient instance segmentation`.
I also encourage everyone to contact me via my e-mail. My e-mail is: wuyuhuan @ mail.nankai (dot) edu.cn

### License

The code is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) for NonCommercial use only.


### Citations

If you are using the code/model/data provided here in a publication, please consider citing our work:

````
@article{wu2021regularized,
   title={Regularized Densely-Connected Pyramid Network for Salient Instance Segmentation},
   volume={30},
   ISSN={1941-0042},
   DOI={10.1109/tip.2021.3065822},
   journal={IEEE Transactions on Image Processing},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Wu, Yu-Huan and Liu, Yun and Zhang, Le and Gao, Wang and Cheng, Ming-Ming},
   year={2021},
   pages={3897–3907}
}
````


### Acknowlogdement

This repository is built under the help of the following five projects for academic use only:

* [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

* [FCOS](https://github.com/tianzhi0549/FCOS)

* [Apex](https://github.com/NVIDIA/apex)

* [MS COCO Dataset](https://cocodataset.org/)

* [S4Net](https://github.com/RuochenFan/S4Net)
