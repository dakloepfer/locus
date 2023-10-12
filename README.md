# LoCUS
Code for the ICCV 2023 paper ["LoCUS: Learning Multiscale
3D-consistent Features from Posed Images"](https://www.robots.ox.ac.uk/~vgg/research/locus)


![teaser](assets/splash_horizontal.png)


## Data
The data used to train the LoCUS model is the [Matterport3D](https://niessner.github.io/Matterport/) dataset.

### Data Preparation
1. Download the dataset using the instructions on the [dataset website](https://niessner.github.io/Matterport/).
2. Run `scripts/data_preparation_scripts/matterport3d_precompute.py` to generate better depth maps.
3. Run `scripts/data_preparation_scripts/matterport3d_calc_img_overlap.py` to compute the overlaps between images, information that may be used for more effective data sampling.
4. (Optional) Run `scripts/data_preparation_scripts/matterport3d_compute_segmentations.py` to compute the segmentation maps from the labelled triangle meshes. Use the `--instance_segmentation` flag to generate the labels for the instance segmentation with object re-identification task.

## Training the model
To train the model, run `train.py` with the desired configuration files and command line options. Some default configurations can be found in the `configs` folder.

## Testing the model
To train the model, run `test.py` with the desired configuration files and command line options. Some default configurations can be found in the `configs` folder.

## Pre-Trained Weights
We release pre-trained weights for two models at the following links: 
* [DINO backbone, $\rho_j=0.2m$](https://www.robots.ox.ac.uk/~vgg/research/locus/locus_dino_rad0.2m_ep20.pth)
* [DINO backbone, $\rho_j=0.6m$](https://www.robots.ox.ac.uk/~vgg/research/locus/locus_dino_rad0.6m_ep20.pth).

The models were trained using the settings described in the paper, and only vary in the positive region radius $\rho_j$ used, which is 0.2m and 0.6m respectively. For detailed results for these two models, please refer to the ablation study in the [supplementary material](https://www.robots.ox.ac.uk/~vgg/research/locus/locus-supp.pdf).

After downloading a file from one of the links, the path to that file can be used as the `--ckpt_path` argument in the `train.py` and `test.py` scripts.


## Paper
If you find this work useful, please consider citing:
```
@InProceedings{Kloepfer_2023_ICCV,
    author    = {Kloepfer, Dominik A. and Campbell, Dylan and Henriques, Jo\~ao F.},
    title     = {LoCUS: Learning Multiscale 3D-consistent Features from Posed Images},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {16634-16644}
}
```