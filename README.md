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



## Usage of training a speech enhancement system

```
cd speech
extract_feat.py --train_clean_list_fn="data/train_clean.scp" --train_noisy_list_fn="data/train_noisy.scp" --test_clean_list_fn="test_clean.scp" --test_noisy_list_fn="test_noisy.scp"
```

```shell
python train_tt.py
```

## Contributing

Pull requests are welcome!

Besides, using the [issue tracker](https://github.com/uwjunqi/Tensor-Train-Neural-Network/issues), feel free to contact me at <jqi41@gatech.edu>. 

