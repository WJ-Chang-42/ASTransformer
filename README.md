# Transformer-based Monocular Depth Estimation with Attention Supervision

## Pretrained models
You can download pre-trained model
* [Trained with KITTI](https://drive.google.com/file/d/1nU_3PwIntr_781ZZ_8IB6pqxAGyR_14K/view?usp=sharing)

   |  cap  |  a1   |  a2   |  a3   | Abs Rel | RMSE  | RMSE log |
   | :---: | :---: | :---: | :---: | :-----: | :---: | :------: |
   | 0-80m | 0.963 | 0.995 | 0.999 |  0.058  | 2.685 |  0.089   |

* [Trained with NYU Depth V2](https://drive.google.com/file/d/1NfoPZA25FYUgBMpieO2Ok6Fe436OqkRF/view?usp=sharing)

   |  cap  |  a1   |  a2   |  a3   | Abs Rel | log10 | RMSE  |
   | :---: | :---: | :---: | :---: | :-----: | :---: | :---: |
   | 0-10m | 0.902 | 0.985 | 0.997 |  0.103  | 0.044 | 0.374 |
   
 ## Evaluation
 Evaluate on KITTI
 ```bash
 python3 eval.py --model_dir "PATH/KITTI.pkl" --evaluate --batch_size 1 --dataset KITTI --data_path "PATH" --gpu_num 0 --encoder AST
 ```
 
 Evaluate on NYU2
 ```bash
 python3 eval.py --model_dir "PATH/NYU2.pkl" --evaluate --batch_size 1 --dataset NYU --data_path "PATH" --gpu_num 0 --encoder AST
 ```
 
 [DPT-Large trained by us](https://drive.google.com/drive/folders/1FYlLpkYnEZqMatBxlYtHODyngfSvnKz8)
 ```bash
 python3 eval.py --model_dir "MODEL_PATH" --evaluate --batch_size 1 --dataset "DATASET" --data_path "PATH" --gpu_num 0 --other_method DPT-Large
 ```
 [Adabins trained by us](https://drive.google.com/drive/folders/1bkLrd1ELmZtygk9mp5MKjbWGbQZa63M8)
 ```bash
 python3 eval.py --model_dir "MODEL_PATH" --evaluate --batch_size 1 --dataset "DATASET" --data_path "PATH" --gpu_num 0 --other_method Adabins
 ```

## Dataset Preparation
We referred to [LapDepth](https://github.com/tjqansthd/LapDepth-release) in the data preparation process. Thanks for sharing the processed dataset.

### KITTI
**1. [Official ground truth](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip)**  
   * Download official KITTI ground truth on the link and make KITTI dataset directory.
```bash
    $ cd ./datasets
    $ mkdir KITTI && cd KITTI
    $ mv ~/Downloads/data_depth_annotated.zip ./datasets/KITTI
    $ unzip data_depth_annotated.zip
```
**2. Raw dataset**  
   * Construct raw KITTI dataset using following commands.
```bash
    $ cd ./datasets/KITTI
    $ aria2c -x 16 -i ./datasets/kitti_archives_to_download.txt
    $ parallel unzip ::: *.zip
```
**3. Dense g.t dataset**  
   We take an inpainting method from [DenseDepth](https://github.com/ialhashim/DenseDepth) to get dense g.t for gradient loss.  
   (You can train our model using only data loss without gradient loss, then you don't need dense g.t)  
   Corresponding inpainted results from **'`./datasets/KITTI/data_depth_annotated/2011_xx_xx_drive_xxxx_sync/proj_depth/groundtruth/image_02`'** are should be saved in **'`./datasets/KITTI/data_depth_annotated/2011_xx_xx_drive_xxxx_sync/dense_gt/image_02`'**.  
KITTI data structures are should be organized as below:                           

    |-- datasets
      |-- KITTI
         |-- data_depth_annotated  
            |-- 2011_xx_xx_drive_xxxx_sync
               |-- proj_depth  
                  |-- groundtruth            # official G.T folder
            |-- ... (all drives of all days in the raw KITTI)  
         |-- 2011_09_26                      # raw RGB data folder  
            |-- 2011_09_26_drive_xxxx_sync
         |-- 2011_09_29
         |-- ... (all days in the raw KITTI)  


### NYU Depth V2
**1. Training set**  
    Make NYU dataset directory
```bash
    $ cd ./datasets
    $ mkdir NYU_Depth_V2 && cd NYU_Depth_V2
```
* Constructing training data using following steps :
    * Download Raw NYU Depth V2 dataset (450GB) from this **[Link](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_raw.zip).**  
    * Extract the raw dataset into '`./datasets/NYU_Depth_V2`'  
    (It should make **'`./datasets/NYU_Depth_V2/raw/....`'**).  
    * Run './datasets/sync_project_frames_multi_threads.m' to get synchronized data. (need Matlab)  
    (It shoud make **'`./datasets/NYU_Depth_V2/sync/....`'**).  
* Or, you can directly download whole 'sync' folder from our Google drive **[Link](https://drive.google.com/file/d/106oW6C7dfLHQYCNXZw9pn9q61ewNIZV1/view?usp=sharing)** into **'`./datasets/NYU_Depth_V2/`'**

**2. Testing set**  
    Download official nyu_depth_v2_labeled.mat and extract image files from the mat file.
```bash
    $ cd ./datasets
    ## Download official labled NYU_Depth_V2 mat file
    $ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
    ## Extract image files from the mat file
    $ python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ./NYU_Depth_V2/official_splits/
```
