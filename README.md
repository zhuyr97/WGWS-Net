# WGWS-Net (CVPR 2023)
This is the official implementation of the CVPR 2023 paper [Learning Weather-General and Weather-Specific Features for Image Restoration Under Multiple Adverse Weather Conditions](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Learning_Weather-General_and_Weather-Specific_Features_for_Image_Restoration_Under_Multiple_CVPR_2023_paper.pdf).

## Abstract

_Image restoration under multiple adverse weather conditions aims to remove weather-related artifacts by using a single set of network parameters. In this paper, we find that image degradations under different weather conditions contain general characteristics as well as their specific characteristics. Inspired by this observation, we design an efficient unified framework with a two-stage training strategy to explore the weather-general and weather-specific features. The first training stage aims to learn the weather-general features by taking the images under various weather conditions as inputs and outputting the coarsely restored results. The second training stage aims to learn to adaptively expand the specific parameters for each weather type in the deep model, where the requisite positions for expanding weather-specific parameters are automatically learned. Hence, we can obtain an efficient and unified model for image restoration under multiple adverse weather conditions. Moreover, we build the first real-world benchmark dataset with multiple weather conditions to better deal with real-world weather scenarios. Experimental results show that our method achieves superior performance on all the synthetic and real-world benchmark datasets.

<p align=center><img width="90%" src="figs/framework.png"/></p>

## Datasets
| Setting   | Weather Types          | Datasets                           | Training Configurations  |
| :---------: | :----------------------: | :----------------------------------: | :---------------------------------------------------: |
| Setting 1 | (Rain, RainDrop, Snow) | ([Outdoor-Rain](https://github.com/liruoteng/HeavyRainRemoval), [RainDrop](https://github.com/rui1996/DeRaindrop), [Snow100K](https://sites.google.com/view/yunfuliu/desnownet)) | Uniformly sampling 9000 images pairs                |
| Setting 2 | (Rain, Haze, Snow)     | ([Rain1400](https://xueyangfu.github.io/projects/cvpr2017.html), [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0), [Snow100K](https://sites.google.com/view/yunfuliu/desnownet))       | Uniformly sampling 5000 images pairs                |
| Setting 3 | (Rain, Haze, Snow)     | (SPA+, [REVIDE](https://github.com/BookerDeWitt/REVIDE_Dataset), RealSnow)            | Uniformly sampling 160000 images patches            |

**Note**:  
- The training configurations follow the previous methods.
- `SPA+`:  we reveal the duplication and redundancy issues in SPA (Real-rain dataset) and handle these problems in SPA+. We first merge the images with repeated background 
scenes and densify the rain streaks by exploiting the temporal information. Using SPA+ could achieve comparable performance and better handle the dense rain scenes. The amount of SPA+ is a quarter of the original SPA dataset, which obviously facilitates future research. SPA+ could be downloaded from [here](https://pan.baidu.com/s/1fgI4G-OEiLTAV-sfSXVYVA?pwd=cvpr)  (Code: cvpr)
- `RealSnow`: inspired by SPA-Net, we build the first real-world desnowing dataset by using the background-static videos to acquire real-world snowing image pairs. RealSnow could be downloaded from [here](https://pan.baidu.com/s/1XkQh_Us5a09sanusSxEvEg?pwd=cvpr)  (Code: cvpr)

*  [Setting 1] 
*  [Setting 2]
*  [Setting 3]

## Pretrained models
[Setting 1](https://drive.google.com/drive/folders/1B0R3SI6D5PkAJGkx_axUm6V5NpjkQllo?usp=share_link) | [Setting 2](https://drive.google.com/drive/folders/1B0R3SI6D5PkAJGkx_axUm6V5NpjkQllo?usp=share_link) | [Setting 3](https://drive.google.com/drive/folders/1B0R3SI6D5PkAJGkx_axUm6V5NpjkQllo?usp=share_link)

## Train
1. The training stage 1:
      ```python
      python training_Setting1_Stage1.py --experiment_name [experiment_name]  --base_channel 18 --fix_sample 9000 --BATCH_SIZE 4 --Crop_patches 224 --EPOCH 100 --T_period 
             50  --learning_rate 0.0002   --addition_loss VGG --depth_loss True --Aug_regular False --print_frequency 100
      ```   
2. The training stage 2: 
      ```python
      python training_Setting1_wDP_Stage2.py --experiment_name [experiment_name] --lam 0.008 --VGG_lamda 0.2  
             --learning_rate 0.0001 --fix_sample 9000 --Crop_patches 224 --BATCH_SIZE 12 --EPOCH 120 --T_period 30 --flag K1 --base_channel 18 --print_frequency 100         
             --pre_model  [path to the pre-trained weights at the training stage 1]
      ```   

## Test
You can directly test the performance of the pre-trained model as follows
1. Modify the paths of datasets and pre-trained weights.
2. Test the model

   2.1 On the Setting 1
      ```python
      python testing_model_Seting1.py --flag K1 --base_channel 18 --num_block 6 --save_path [path to your save_path]
      ```
   2.2 On the Setting 2
      ```python
      python testing_model_Seting2.py --flag K1 --base_channel 20 --num_block 6 --save_path [path to your save_path]
      ```   
   2.3 On the Setting 3
      ```python
      python testing_model_Seting3.py --flag K1 --base_channel 18 --num_block 6 --save_path [path to your save_path]
      ```   
You can check the processed results in `[path to your save_path]`.

## Citation
If you find this work or code useful for your research, please cite:
```
@inproceedings{zhu2023Weather,
  title={Learning Weather-General and Weather-Specific Features for Image Restoration Under Multiple Adverse Weather Conditions},
  author={Yurui Zhu and Tianyu Wang and Xueyang Fu and Xuanyu Yang and Xin Guo and Jifeng Dai and Yu Qiao and Xiaowei Hu},
  booktitle={Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}

```

## Contact
If you have any questions, please contact zyr@mail.ustc.edu.cn
