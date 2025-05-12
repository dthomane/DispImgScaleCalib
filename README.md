# Metric Scaling and Extrinsic Calibration of Monocular Neural Network-Derived 3D Point Clouds in Railway Applications
The repository contains the official implementation of our [publication](https://www.mdpi.com/2076-3417/15/10/5361)

![Teaser image of example results](/teaser/teaserimg.png)


## Installation
We recommend using a virtual environment such as conda for managing dependencies. The core code has been tested with Python 3.11 and works with preprocessed depth and segmentation data. For this, you only need the following Python packages:
- numpy
- matplotlib
- scipy
- opencv
- open3d
- DBSCANN1d (https://pypi.org/project/dbscan1d/)
You can create a minimal conda environment for testing with preprocessed data using the provided environment.yml file.

## Quick Start (with Preprocessed Data)
You can run our minimal example download by typing (preprocessed files are included in example folder)
```
python example.py
```
(Without arguments preprocessed depth and segmentation is used)

## Using OSDaR23 Dataset
We have prepared calibration and evaluation scripts for the OSDaR23 Dataset. 

You can download the preprocessed depth and segmentation data from [here](https://tubcloud.tu-berlin.de/s/A7XGaEspa6GxpzJ)

Please download the original scenes from the official dataset page [here](https://data.fid-move.de/dataset/osdar23)

> [!IMPORTANT]
> Since the preprocessed data is derived from the OSDaR23 Dataset, it is published under the same license: [Creative Commons Attribution-ShareAlike 3.0 Germany](https://creativecommons.org/licenses/by-sa/3.0/deed.de)
> 
> Many thanks to:
> 
> <em>Roman Tilly , Philipp Neumaier , Karsten Schwalbe , Pavel Klasek , Rustam Tagiew , Patrick Denzler , Tobias Klockau , Martin Boekhoff , Martin Köppel , (2023). Open Sensor Data for Rail 2023 [Data set]. TIB. https://doi.org/10.57806/9mv146r0</em>



If you have downloaded the preprocessed data, please organize it using the following folder structure:
```
rootfolder
├── preprocessed
│   ├── unidepth
|   |   ├── 1_calibration_1.1
:   :   :   :
|   |   └── 21_station_wedel_21.3
│   └── internimage
|   |   ├── 3_fire_site_3.1
:   :   :   :
|   |   └── 21_station_wedel_21.3
├── dataset
|   |   ├── 1_calibration_1.1
:   :   :   :
|   |   └── 21_station_wedel_21.3
```
and provide the path to rootfolder:
```
python calibrateOsdar23.py path/to/rootfolder
python evaluateOsdar23.py path/to/rootfolder
```
(Without arguments preprocessed depth and segmentation is used)

Calibration creates a folder calib_results in rootfolder including JSON-files with calibration info. Evaluation create a folder calib_results in rootfolder with CSV-files containing evaluation results


## Using Depth Models Directly
If you prefer to use a depth model directly (instead of preprocessed data), make sure your environment is properly configured. You will need to provide the path to the repository of the depth model you wish to use.

Since the functionality is highly dependent on the specific model and your environment (including correct CUDA setup), we do not provide environment configurations for these models.

To use a depth model:

1. Visit the model's official GitHub page.
2. Clone the repository.
3. Follow their installation instructions.
4. Download any required weights or additional resources.

We have prepared integration for the following models:
- unidepth
- depthpro
- depthanything
- adabins
- midas

When using a depth model, you must specify the model name along with the path to its repository when calling our functions:
```
python example.py --depthmodel unidepth --depthmodelrepo path/to/repo/of/unidepth
```

## Using InternImage Directly
If you want to use InternImage directly (instead of preprocessed data), make sure your environment is properly configured due to the InternImage installation instructions.

Since the functionality is highly dependent on the specific model and your environment (including correct CUDA setup), we do not provide environment configurations for InternImage

To use a depth model:

1. Download our pretrained weights here (Link follows soon)
2. Provide the path to these weights.

```
python example.py --segmentation internimage --internimage_chekpoint path/to/pretrained/weights
```

## Citation
When using this code please cite our publications:
```
@Article{app15105361,
  AUTHOR = {Thomanek, Daniel and Gühmann, Clemens},
  TITLE = {Metric Scaling and Extrinsic Calibration of Monocular Neural Network-Derived 3D Point Clouds in Railway Applications},
  JOURNAL = {Applied Sciences},
  VOLUME = {15},
  YEAR = {2025},
  NUMBER = {10},
  ARTICLE-NUMBER = {5361},
  URL = {https://www.mdpi.com/2076-3417/15/10/5361},
  ISSN = {2076-3417},
  DOI = {10.3390/app15105361}
}

```
