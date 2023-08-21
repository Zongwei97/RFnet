# RFnet

This is the official implementation of [Robust RGB-D Fusion network for Saliency Detection](https://arxiv.org/pdf/2208.01762.pdf), 3DV 2022

# Abstract

Efficiently exploiting multi-modal inputs for accurate RGB-D saliency detection is a topic of high interest. Most existing works leverage cross-modal interactions to fuse the two streams of RGB-D for intermediate features' enhancement. In this process, a practical aspect of the low quality of the available depths has not been fully considered yet. In this work, we aim for RGB-D saliency detection that is robust to the low-quality depths which primarily appear in two forms: inaccuracy due to noise and the misalignment to RGB.  To this end, we propose a robust RGB-D fusion method that benefits from  (1) layer-wise, and (2) trident spatial, attention mechanisms. On the one hand, layer-wise attention (LWA) learns the trade-off between early and late fusion of RGB and depth features, depending upon the depth accuracy. On the other hand, trident spatial attention (TSA) aggregates the features from a wider spatial context to address the depth misalignment problem. 
The proposed LWA and TSA mechanisms allow us to efficiently exploit the multi-modal inputs for saliency detection while being robust against low-quality depths. Our experiments on five benchmark datasets demonstrate that the proposed fusion method performs consistently better than the state-of-the-art fusion alternatives. 

![Graphical Abstract](https://github.com/Zongwei97/RFnet/blob/main/Imgs/abstract.png)
# Train and Test

One key: 

```
python run_RFNet.py
```


# Saliency Maps

The saliency maps can be downloaded here: [Google Drive](https://drive.google.com/file/d/1efZfbZ11L2cBs5Mwnt1awwHjFuiy-1DQ/view?usp=sharing)

![Qualitative Comparison](https://github.com/Zongwei97/RFnet/blob/main/Imgs/qualitative.png)


# Citation

If you find this repo useful, please consider citing:

```
@INPROCEEDINGS{wu2022robust,
  author={Wu, Zongwei and Gobichettipalayam, Shriarulmozhivarman and Tamadazte, Brahim and Allibert, Guillaume and Paudel, Danda Pani and Demonceaux, Cédric},
  booktitle={2022 International Conference on 3D Vision (3DV)}, 
  title={Robust RGB-D Fusion for Saliency Detection}, 
  year={2022},
  volume={},
  number={},
  pages={403-413},
  doi={10.1109/3DV57658.2022.00052}}
  
```

# Related works
- ICCV 23 - Source-free Depth for Object Pop-out [[Code]( https://github.com/Zongwei97/PopNet)]
- ACMMM 23 - Object Segmentation by Mining Cross-Modal Semantics [[Code](https://github.com/Zongwei97/XMSNet))]
- TIP 23 - HiDANet: RGB-D Salient Object Detection via Hierarchical Depth Awareness [[Code](https://github.com/Zongwei97/HIDANet)]
- 3DV 22 - Modality-Guided Subnetwork for Salient Object Detection [[Code](https://github.com/Zongwei97/MGSnet)]


# Acknowledgments
This repository is heavily based on [SPNet](https://github.com/taozh2017/SPNet). Thanks to their great work!


