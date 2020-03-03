# Tensor-Train Neural Networks for Speech Processing. 

### 1. Data preparation: 
```
Pleaese download the noisy speech data from the website: https://datashare.is.ed.ac.uk/handle/10283/2791 
Noisy training data: noisy_trainset_56spk_wav.zip (5.240Gb)
Noisy testing data:  noisy_testset_wav.zip (162.6Mb)
Clean training data: clean_trainset_56spk_wav.zip (4.442Gb)
Clean testing data:  clean_testset_wav.zip (147.1Mb)
```

### 2. Down-sampling the speech data from 48KHz to 16KHz. 

### 3. Feature extractiong: 
```
python extract_feat.py --clean_list_fn=${clean_data_scp} --noise_list_fn=${noise_data_scp} --data_fn={feat_fn}
```

### 4. Model traning: 
```
python train_tt.py (Tensor-Train) 
python train_dnn.py (DNN)
```

### 5. Synthesizing enhanced speech
```
python synthesis_dnn.py (DNN) or synthesis_tt.py (Tensor-Train)
```

### 6. Attaining PESQ scores
```
python pesq_metric.py --data_enhanced_scp=$enhanced_scp --data_clean_scp=$clean_data_scp
```
