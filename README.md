# LoCUS
Code for the ICCV 2023 paper ["LoCUS: Learning Multiscale
3D-consistent Features from Posed Images"](https://www.robots.ox.ac.uk/~vgg/research/locus)



Pre-trained weights coming soon...

![teaser](assets/splash_horizontal.png)


## Data
The data used to train the LoCUS model is the [Matterport3D](https://niessner.github.io/Matterport/) dataset.

### Data Preparation
1. Download the dataset using the instructions on the [dataset website](https://niessner.github.io/Matterport/).
2. Run `scripts/data_preparation_scripts/matterport3d_precompute.py` to generate better depth maps.
3. Run `scripts/data_preparation_scripts/matterport3d_calc_img_overlap.py` to compute the overlaps between images, information that may be used for more effective data sampling.

## Training the model
To train the model, run `train.py` with the desired configuration files and command line options. Some default configurations can be found in the `configs` folder.

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