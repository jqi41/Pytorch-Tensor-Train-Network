# TC: Tensor-Train-Neural-Network

```
git clone https://github.com/uwjunqi/Tensor-Train-Neural-Network.git
cd Tensor-Train-Neural-Network
```

## Installation

The main dependencies are *h5py*, *Numpy* and *PyTorch*. To download and install *tc*:

### CUDA 10.0 setup

```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install -c anaconda h5py 
conda install -c conda-forge matplotlib 
python setup.py install
```

## Tensor-Train Neural Network for Image Processing
```
cd image
python train_tt_image.py
```


## Tensor-Train Neural Network for Speech Enhancement

```
cd speech
extract_feat.py --train_clean_list_fn="data/train_clean.scp" --train_noisy_list_fn="data/train_noisy.scp" --test_clean_list_fn="test_clean.scp" --test_noisy_list_fn="test_noisy.scp"
```

```shell
python train_tt_speech.py
```

## Contributing

Pull requests are welcome!

Besides, using the [issue tracker](https://github.com/uwjunqi/Tensor-Train-Neural-Network/issues), feel free to contact me at <jqi41@gatech.edu>. 


## Paper Citation:

If you use the codes for your research work, please cite the following paper:

[1] Jun Qi, Hu Hu, Yannan Wang, Chao-Han Huck Yang, Sabato Marco Siniscalchi, Chin-Hui Lee, "Tensor-Train Network based Tensor-to-Vector Regression for Multi-Channel Speech Enhancement,” in Proc. IEEE Intl. Conf. on Acoustic, Speech, and Signal Processing (ICASSP), Barcelona, Spain, 2020. 

[2] Jun Qi, Jun Du, Sabato Marco Siniscalchi, Chin-Hui Lee, "A Theory on Deep Neural Network based Vector-to-Vector Regression with an Illustration of Its Expressive Power in Speech Enhancement", in IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP), Vol 27 ,  no. 12, pp. 1932-1943, Dec 2019. 

```@article{qi2019theory,
  title={A Theory on Deep Neural Network Based Vector-to-Vector Regression With an Illustration of Its Expressive Power in Speech Enhancement},
  author={Qi, Jun and Du, Jun and Siniscalchi, Sabato Marco and Lee, Chin-Hui},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={27},
  number={12},
  pages={1932--1943},
  year={2019},
  publisher={IEEE}
}```



