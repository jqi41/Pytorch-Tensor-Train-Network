# Pytorch-Tensor-Train-Network

The package mainly aims to provide a Pytorch implementation for Tensor Decompositions to Deep Neural Networks, particularly for Tensor-to-Vector Regression tasks, e.g., Speech and Image Enhancement. 

```
git clone https://github.com/uwjunqi/Pytorch-Tensor-Train-Network.git
cd Pytorch-Tensor-Train-Network
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

### Note that the implementation of the PyTorch code is not totally the same as our Tensorflow code for the experimental setups in [1] Qi et al. ICASSP 2020 and [2] Qi et al. Interspeech 2020. Parts of Pytorch backbone are initialized and based on the T3F project. We sincerely thank the authors.
This repo is released for general use and also included some image TTN examples. 
If you are interested on reproducing the results of [1] and [2] please contact. (jqi41 at gatech dot edu)

If you use the codes for your research work, please consider citing the following papers:

[1] Jun Qi, Chao-Han Huck Yang, Pin-Yu Chen, Javier Tejedor, "Exploiting Low-Rank Tensor-Train Deep Neural Networks Based on Riemannian Gradient Descent with Illustration of Speech Processing," IEEE/ACM Transactions on Audio, Speech and Language Processing (T-ASLP), Vol. 31, pp. 633-642, 2023

[2] Jun Qi, Hu Hu, Yannan Wang, Chao-Han Huck Yang, Sabato Marco Siniscalchi, Chin-Hui Lee, "Tensor-to-Vector Regression for Multi-Channel Speech Enhancement based on Tensor-Train Network,” in Proc. IEEE Intl. Conf. on Acoustics, Speech, and Signal Processing (ICASSP), Barcelona, Spain, 2020 
https://arxiv.org/abs/2002.00544

[3] Jun Qi, Hu Hu, Huck Yang, Sabato Marco Siniscalchi, Chin-Hui Lee, “Exploring Deep Hybrid Tensor-to-Vector Network Architectures for Regression Based Speech Enhancement,”  in Proc. Annual Conference of the International Speech Communication Association (INTERSPEECH), Shanghai, China, 2020 https://arxiv.org/abs/2007.13024v2

## Reference:

We also borrow some c header files from the open-source project and the recently released journal version [4] from Novikov et al. Please also check their works as below.

[3] Novikov, A., Podoprikhin, D., Osokin, A., & Vetrov, D. P. (2015). Tensorizing neural networks. In Advances in neural information processing systems (pp. 442-450).

[4] Novikov et al. [T3F](https://github.com/Bihaqo/t3f) Tensor Train Decomposition on TensorFlow (T3F), JMLR 2020, http://jmlr.org/papers/v21/18-008.html

