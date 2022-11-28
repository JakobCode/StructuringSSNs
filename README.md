# Structuring Uncertainty for Fine-Grained Sampling in Stochastic Segmentation Networks
<center>
<img src="https://user-images.githubusercontent.com/105631549/170479247-8186327c-09d5-42b3-bc33-1a03da550926.gif" width=75% >
</center>

## Content 
This repository come along with the paper <i>Structuring Uncertainty for Fine-Grained Sampling in Stochastic Segmentation Networks</i> (<a href="https://openreview.net/forum?id=odOQU9PYrkD">Link to Paper</a>), published at <i>Neurips 2022</i>.
Specifically, the repository contains the following components:
* <u>Prepare Models</u> <br> Contains everything what needs to be done before the predicted factors can be rotated, i.e., the training and evaluation procedure of different stochastic segmentation models (for details see below). 
* <u>Factor Rotations</u> <br> This folder contains everything related to the factor rotations, i.e., the rotation script and different metrics to evaluate single rotations and reproduce the results from the paper. 
* <u>Demo</u> <br> Contains a Jupyter notebook demonstrating the proposed approach. The folder also contains 5 example images for exploring the approach. 

## Citation
If our code or our results are useful in your reasearch, please consider citing:

```
@inproceedings{nussbaum2022.structuring,
 author = {Nussbaum, Frank and Gawlikowski, Jakob and Niebling, Julia},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Structuring Uncertainty for Fine-Grained Sampling in Stochastic Segmentation Networks},
 volume = {35},
 year = {2022}
}
```

# Working with this repository
### Pre-requirements
This repository has been tested under ```Python 3.7``` in a *unix* development environment. <br> 
For a setup, clone the repository and ``cd``to the root of it. <br>
Create a new environment and activate it, for instance, on unix via
```
python -m venv venv && source venv/bin/activate
```
Then install the needed packages via:
```
pip install --upgrade pip
pip install -r requirements.txt
```
## Getting started with examples
To get a first impression of the approach, we offer a demo jupyter notebook located in ```./Demo/``` (a quick demo video for fine-grained sampling is found in the file 'finegrainedsampling_demo.avi', see also above.
We prepared some test images for usage in this notebook (with pre-computed rotations). 
The notebook can be run within your browser. Use the command ```jupyter notebook``` and navigate in the browser to ```Demo/demo.ipynb```)

## Reproduction of the results presented in the paper
The following visualization gives an overview of the components contained in this repository that are needed for the *prediction, rotation and evaluation* of different representations of predicted factor models.
<center>
<img src="https://user-images.githubusercontent.com/105631549/182422309-18d8019d-ab24-4100-9d44-cdda425d26a7.png" width=75% >
</center>

In more detail, the following steps can be undertaken to reproduce the results from the paper:
1) Train the SSNs (see below)
2) Evaluate test images using trained SSNs. The predicted factor model parameters are saved into a Pickle file stored in the corresponding subfolder of ```./Results/```
3) Compute the rotations. The rotation files are also stored in the pkl-file. 
4) Visualize and manipulate rotated factors using the gui provided with the Jupyter Notebook ```./Demo/demo.ipynb```

In the following, we describe these steps in more detail.

### Train a Stochastic Segmentation Network
Training pipelines for the *LIDC* [1,2,3], the *SEN12MS* [5], and the *CamVid* dataset are pre-implemented. <br><br>
The main steps needed for the training process are similar for all three data sets. <br><br>

Exemplarily, for the SEN12MS dataset, the steps would be the following:
1) Download the data as stated below.
2) Setup the experiment configuration, the configuration used in the publication is given in the available config files.
3) Start the training with the path to the experiment configuration as parameter: <br>
```python ./PrepareModels/train_SEN12MS.py --config ./PrepareModels/Config_SEN12MS.yml```
The training scripts automatically build an experiment ```./Results/<experiment_name>/<experiment_folder>``` and place a copy of the configuration file there. 

### Evaluation of trained networks and saving of the predicted factor models
In this part, the predictive performance of the trained networks can be evaluated on the test split of the corresponding data set. **Further, for each test data sample/image, a pickle file is saved**. These pickle files contain information about the predicted uncertainty, as for example the predicted *mean*, *low-rank covariance factor*, and the *diagonal* of the predicted factor model. <br>

The evaluation process can be started similarly for all three data sets, with an experiment folder as parameter. For example, for the SEN12MS dataset it is:<br>
```python ./PrepareModels/eval_SEN12MS.py --experiment_folder <path_to_exp_folder>``` <br><br>
For saving the pickle files, a folder structure is created within the experiment folder: ```<experiment_folder>/results/pkls```. 

### Compute Rotations
Different rotations can be computed using ```./FactorRotations/PreComputeRotations.py```:<br><br>
```python ./FactorRotations/PreComputeRotations.py --pkl_source <path_to_pkl_source> --rotation <rotation> --no_reps <no_reps>```
* *pkl_source* - is either the path to a single pickle-file or a folder containing multiple pickle files.
* *rotation* - 'all' for all rotations or one of 'varimax', 'fpvarimax', 'equamax', 'fpequamax', 'quartimax', 'fpquartimax'.
* *no_reps* - number of repetions/restarts, default: 1.

### Evaluate Rotations
The computed rotations can be evaluated individually using ```./FactorRotations/EvalRotatedFactors.py```:<br><br>
```python ./FactorRotations/EvalRotatedFactors.py --pkl_source <path_to_pkl_source>```
* *pkl_source* - is either the path to a single pickle-file or a folder containing multiple pickle files.<br><br>

The results of multiple files can be summarized and plotted using ```./FactorRotations/PrintRotationResults.py```:<br><br>
```python ./FactorRotations/PrintRotationResults.py --pkl_source <path_to_pkl_source>```
* *pkl_source* - a folder containing multiple pickle files with evaluated rotations.<br><br>

### Demo
For a visual evaluation of the rotations, the demo notebook placed in ```./Demo/``` can be used. It loads pickle files containing different rotations, computed as described above. We deliver some example files with pre-computed rotations that are located in ```./Demo/Examples/```. In order to load your own files, the root path within the demo can be adjusted.

## Data sets and other resources
#### LIDC
The original LIDC data set can be found here:
* Original Data Set [1,2,3]:&nbsp;&nbsp;&nbsp;&nbsp; [Cancer Image Archive](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI#1966254a2b592e6fba14f949f6e23bb1b7804cc) &nbsp;([TCIA Data Usage Policy](https://wiki.cancerimagingarchive.net/x/c4hF) and the [Creative Commons Attribution 3.0 Unported License](https://creativecommons.org/licenses/by/3.0/))

In order to work with the LIDC data set download the processed data in the following the link in the list below and place it into the stated target folder. Follow the description in the repo below and place the file ```data_lidc.pickle``` in ```./Data/LIDC/```. The first time the data is loaded, a processed hdf5-file is generated and stored. 
* Preprocessed Data:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://github.com/stefanknegt/Probabilistic-Unet-Pytorch (Apache-2.0 license)
This repository contains code snippits to handle the data processing and handling of the LIDC data samples. All code from external sources is marked in the header of the script and the references are also listed below.
* Code Snippets [4]:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://github.com/MiguelMonteiro/PHiSeg-code (Apache-2.0 license)

---

#### SEN12MS
In order to work with the SEN12MS data set, the data should be downloaded and placed into the stated target folder:
* Data Set [5]:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://mediatum.ub.tum.de/1474000 (Apache-2.0 license) 
* Test splits: download the files defining training, validation and test split and place them into the root data folder as well: https://github.com/schmitt-muc/SEN12MS/tree/master/splits
* Target Folder:&nbsp;&nbsp;&nbsp;&nbsp;```./Data/SEN12MS/data``` (Lisense: [Creative Commons-4.0](http://creativecommons.org/licenses/by/4.0))

For the data handling and training pipeline we took and modified available code from the official SEN12MS repository. The affected scripts are marked in the headers of the files. 
* Official Repository [5]:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[https://github.com/schmitt-muc/SEN12MS](https://github.com/schmitt-muc/SEN12MS)&nbsp;&nbsp;(license: [custumn license](https://github.com/schmitt-muc/SEN12MS/blob/master/LICENSE.txt))

---

#### CamVid
The CamVid project [6] was realized at Cambridge University. 
* Project page:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)

We build the training pipeline based on the implementations available [here](https://github.com/bfortuner/pytorch_tiramisu). Code taken from this repository is mentioned in the header of the corresponding files. 
* Data Set:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid)
Starting form the root of our repository, use the following command to download the data and place it in the correct directory: <br>
```cd Data && mkdir -p CamVid/data && git clone git@github.com:alexgkendall/SegNet-Tutorial.git tmp && mv tmp/CamVid/* CamVid/data/ && rm -rf tmp && cd```
* Code Snippets:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[https://github.com/bfortuner/pytorch_tiramisu](https://github.com/bfortuner/pytorch_tiramisu)&nbsp;&nbsp;&nbsp;(License: MIT license)
* Target Folder:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```./Data/CamVid/data```

## References
> **[1]** Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R., Reeves, A. P., Zhao, B., Aberle, D. R., Henschke, C. I., Hoffman, E. A., Kazerooni, E. A., MacMahon, H., Van Beek, E. J. R., Yankelevitz, D., Biancardi, A. M., Bland, P. H., Brown, M. S., Engelmann, R. M., Laderach, G. E., Max, D., Pais, R. C. , Qing, D. P. Y. , Roberts, R. Y., Smith, A. R., Starkey, A., Batra, P., Caligiuri, P., Farooqi, A., Gladish, G. W., Jude, C. M., Munden, R. F., Petkovska, I., Quint, L. E., Schwartz, L. H., Sundaram, B., Dodd, L. E., Fenimore, C., Gur, D., Petrick, N., Freymann, J., Kirby, J., Hughes, B., Casteele, A. V., Gupte, S., Sallam, M., Heath, M. D., Kuhn, M. H., Dharaiya, E., Burns, R., Fryd, D. S., Salganicoff, M., Anand, V., Shreter, U., Vastagh, S., Croft, B. Y., Clarke, L. P. (2015). Data From LIDC-IDRI [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX

> **[2]** Armato SG 3rd, McLennan G, Bidaut L, McNitt-Gray MF, Meyer CR, Reeves AP, Zhao B, Aberle DR, Henschke CI, Hoffman EA, Kazerooni EA, MacMahon H, Van Beeke EJ, Yankelevitz D, Biancardi AM, Bland PH, Brown MS, Engelmann RM, Laderach GE, Max D, Pais RC, Qing DP, Roberts RY, Smith AR, Starkey A, Batrah P, Caligiuri P, Farooqi A, Gladish GW, Jude CM, Munden RF, Petkovska I, Quint LE, Schwartz LH, Sundaram B, Dodd LE, Fenimore C, Gur D, Petrick N, Freymann J, Kirby J, Hughes B, Casteele AV, Gupte S, Sallamm M, Heath MD, Kuhn MH, Dharaiya E, Burns R, Fryd DS, Salganicoff M, Anand V, Shreter U, Vastagh S, Croft BY.  The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. Medical Physics, 38: 915--931, 2011. DOI: https://doi.org/10.1118/1.3528204

> **[3]** Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox, L., & Prior, F. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of Digital Imaging, 26(6), 1045–1057. https://doi.org/10.1007/s10278-013-9622-7

> **[4]** Baumgartner, Christian F., et al. "Phiseg: Capturing uncertainty in medical image segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2019.

> **[5]** Schmitt, M., et al. "SEN12MS–A CURATED DATASET OF GEOREFERENCED MULTI-SPECTRAL SENTINEL-1/2 IMAGERY FOR DEEP LEARNING AND DATA FUSION." ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences 4 (2019): 153-160.

## Python Packges Used

### Data Handling
[einops==0.4.1](https://pypi.org/project/einops)<br>
[h5py==3.1.0](https://pypi.org/project/h5py)<br>
[importlib_metadata==4.11.4](https://pypi.org/project/importlib_metadata)<br>
[numpy==1.17.5](https://pypi.org/project/numpy)<br>
[opencv-python==4.5.5.64](https://pypi.org/project/opencv-python)<br>
[pydicom==2.3.0](https://pypi.org/project/pydicom)<br>
[pandas==1.3.5](https://pypi.org/project/pandas)<br>
[pyyaml==6.0](https://pypi.org/project/pyyaml)<br>
[sklearn](https://pypi.org/project/sklearn)<br>
[tqdm==4.64.0](https://pypi.org/project/tqdm)

### Demo
[ipympl==0.8.8](https://pypi.org/project/ipympl)<br>
[jupyter==1.0.0](https://pypi.org/project/jupyter)

### Medical Data Handling
[MedPy==0.4.0](https://pypi.org/project/MedPy)<br>
[nibabel==3.2.2](https://pypi.org/project/nibabel)

### Remote Sensing Data Handling
[rasterio==1.2.10](https://pypi.org/project/rasterio)

### SSN training and testing
[torch==1.11.0](https://pypi.org/project/torch)<br>
[torchvision==0.12.0](https://pypi.org/project/torchvision)

### Visualization
[matplotlib==3.1.2](https://pypi.org/project/matplotlib)
