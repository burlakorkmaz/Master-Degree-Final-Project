# [Frontiers in Artificial Intelligence] Automated detection of dolphin whistles with convolutional networks and transfer learning

This repository implements the deep learning based automated dolphin whistle detection method.


## Folder Structure
<pre>
.  
└── Train_test/  
    ├── Dataset/  
    │   └── dolphin_signal_dataset/  
    │       ├── Train/  
    │       └── Test/  
    │       └── dolphin_signal_train.csv  
    │       └── dolphin_signal_test.csv  
    ├── figs/  
    ├── models/  
    ├── train_vgg.py  
    ├── test_vgg.py 
    ├── train_cnn.py
    ├── test_cnn.py
    
    
└── Whistle_detection/  
    ├── images/  
    ├── models/  
    │   └── model_vgg.h5  
    ├── predicted_images/  
    │   ├── negative/  
    │   └── positive/  
    ├── recordings/  
    ├── predict.py  
    ├── save_spectrogram.m  
    
</pre>    


## Citation
### Master's Thesis
Link to master's thesis: [https://thesis.unipd.it/handle/20.500.12608/31586](https://thesis.unipd.it/handle/20.500.12608/31586) 
<pre>
@article{korkmazdeep,
  title={Deep learning techniques for biological signal processing: Automatic detection of dolphin sounds},
  author={KORKMAZ, BURLA NUR}
}
</pre>

### Publication
Link to paper: [https://doi.org/10.3389/frai.2023.1099022](https://doi.org/10.3389/frai.2023.1099022)
<pre>
@article{nur2023automated,
  title={Automated detection of dolphin whistles with convolutional networks and transfer learning},
  author={Nur Korkmaz, Burla and Diamant, Roee and Danino, Gil and Testolin, Alberto},
  journal={Frontiers in Artificial Intelligence},
  volume={6},
  pages={1099022},
  year={2023},
  publisher={Frontiers Media SA}
}
</pre>
