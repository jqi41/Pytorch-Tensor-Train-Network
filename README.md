# TC: Tensor-Train-Neural-Network

The package particularly aims at Tensor Decompositions to Deep Neural Networks, particularly for Tensor-to-Vector Regression tasks, e.g., Speech and Image Enhancement. 

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

- TTN AE result

![img](https://github.com/uwjunqi/Tensor-Train-Neural-Network/blob/master/image/ttn.png)

- DNN AE result

![img](https://github.com/uwjunqi/Tensor-Train-Neural-Network/blob/master/image/ae_results.png)

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

### Note that the implementation of the PyTorch codes are not totally the same as our Tensorflow codes for the experimental setups in [1] Qi et al. ICASSP 2020 and [2] Qi et al.TSP. 
This repo is released for general use and also included some image TTN examples. 
If you are interested on reproducing the results of [1] and [2] please contact. (jqi41 at gatech dot edu)


If you use the codes for your research work, please consider citing the following papers:

[1] Jun Qi, Hu Hu, Yannan Wang, Chao-Han Huck Yang, Sabato Marco Siniscalchi, Chin-Hui Lee, "Tensor-to-Vector Regression for Multi-Channel Speech Enhancement based on Tensor-Train Network,” in Proc. IEEE Intl. Conf. on Acoustics, Speech, and Signal Processing (ICASSP), Barcelona, Spain, 2020. 
https://arxiv.org/abs/2002.00544

[2] Jun Qi, Xiaoli Ma, Sabato Marco Siniscalchi, Chin-Hui Lee, "Upper Bounding Mean Absolute Errors for Deep Tensor Regression Based on Tensor-Train Networks," submit to IEEE Transactions on Signal Processing (TSP). 

[3] Reference Report Novikov et al. [T3F](https://github.com/Bihaqo/t3f)
