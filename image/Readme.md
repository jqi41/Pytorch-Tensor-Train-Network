Testing Tensor-Train Neural Network on MNIST dataset. 

`$ python train_tt_image.py --input_tensor [7, 4, 7, 4] --hidden_tensors [[8, 4, 8, 4], [8, 4, 8, 4], [8, 4, 8, 8]]`

```
parser.add_argument('', default=200, help='Mini-batch size')
parser.add_argument('--input_tensor', default=[7, 4, 7, 4])
parser.add_argument('--hidden_tensors', default=[[8, 4, 8, 4], [8, 4, 8, 4], [8, 4, 8, 8]])
parser.add_argument('--data_path', metavar='DIR', default='exp/feats/feats_m109_15dB.h5', help='Feature container in h5 format')
parser.add_argument('--save_model_path', default='model_tt.hdf5', help='The path to the saved model')
parser.add_argument('--n_epochs', default=5, help='The total number of epochs', type=int)
parser.add_argument('--save_model', default='tt_mnist.pt', help='Directory of the saved model path')
```
- Autoencoder

- Tensor-Train Autoencoder

- Related References

```bib
@inproceedings{qi2020tensor,
  title={Tensor-to-vector regression for multi-channel speech enhancement based on tensor-train network},
  author={Qi, Jun and Hu, Hu and Wang, Yannan and Yang, Chao-Han Huck and Siniscalchi, Sabato Marco and Lee, Chin-Hui},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7504--7508},
  year={2020},
  organization={IEEE}
}

@article{qi2020exploring,
  title={Exploring Deep Hybrid Tensor-to-Vector Network Architectures for Regression Based Speech Enhancement$\}$$\}$},
  author={Qi, Jun and Hu, Hu and Wang, Yannan and Yang, Chao-Han Huck and Siniscalchi, Sabato Marco and Lee, Chin-Hui},
  journal={Proc. Interspeech 2020},
  pages={76--80},
  year={2020}
}
```
