# Repository for Going Offline: An Evaluation of the Offline Phase in Stream Clustering

This repository contains the code for the paper submission "Going Offline: An Evaluation of the Offline Phase in Stream Clustering" for ECML PKDD 2025.
It is based on code from the [River Stream learning repository](https://github.com/online-ml/river).

# Note: as the project used some external code, to ensure that this project is functional just from the requirements file, these aspects have been commented out surrounded by ```# -- {method key(s)} --```.

## Usage

The main code used to perform the evaluation is ```runMethod.py```. It allows for the execution of both the CluStream variants and the competitors. CluStream, DenStream, DBSTREAM, and STREAMKmeans come from the River repository and are included as is. The other competitors need to be obtained from their repositories as linked below. They require some modification to use the functions called for them (specifically learn_one/learn and predict_one/predict).
The main method has several parameters that allow for c

| **Parameter**        | **Default Value** | **Function**                                                                                                                                                                                                                                                                      |
|----------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --method                  | clustream         | Stream Clustering method to evaluate (CluStream-W is ```wclustream```, CluStream-S is ```scaledclustream```, CluStream-G is ```scope_full```)  |
| --ds                  | densired10        | Dataset to perform the experiments on  |
| --offline                  | 1000              | Timesteps for offline phase/for evaluation  |
| --sumlimit                  | 100               | Maximal number of micro-clusters  |
| --gennum                  | 1000              | Approximate number of points that CluStream-S and CluStream-G produce  |
| --category                  | all               | Which offline algorithms to include (For the paper, we used ```all``` for all offline algorithms, ```not_projdipmeans``` to run everything but Projected Dip-Means or the keys for the specific offline clustering methods); aside from that there are options to choose all centroid-based methods with ```means```, all centroid-based methods without k-estimation with ```nkestmeans```, all centroid-based methods with k-estimation with ```kestmeans```, all Spectral Clustering methods with ```spectral```, all density-connectivity-based methods with ```denscon```, all non-density-connectivity-based density-based methods with ```density``` and all density-based approaches with ```density_all``` |
| --startindex  | 0                 | Starting index for configuration (to only run on a subset of all parameters), inclusive |
| --endindex                  | np.inf            | Stopping index for configuration (to only run on a subset of all parameters), inclusive |
| --automl                  | 1                 | Whether to use parameters obtained through AutoML or to perform a grid search (requires a parameter dictionary for the desired  dataset and stream clustering setup), Integer Boolean  |
| --used_full                  | 0                 | Whether AutoML used subsampled data or the full dataset, Integer Boolean  |

The method ```file_handler.py```processes the run results into a more manageable format (but needs manual changes for the moment). 
The produced dictionaries for the experiments performed for the paper are available in the folder ```dicts```.

## AutoML

The AutoML parameter optimization pipeline is based on [SMAC3](https://github.com/automl/SMAC3).
The subsets are produced with ```subset_selector.py```. The settings used are automatically executed if the file is run, though the respective datasets must be downloaded first (aside from DENSIRED-10, which is already included in this repository).
To optimize the stream clustering parameters, the code ```parameter_estimator.py``` needs to be run, which has the following parameters.

| **Parameter**        | **Default Value** | **Function**                                                                                                                                                                                                                                                                      |
|----------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --method                  | clustream         | Stream Clustering method to optimize. For any online-offline CluStream variant, use ```clustream```, otherwise use the method keys |
| --ds                  | densired10        | Dataset to perform the optimization on  |
| --use_full                  | 0                 | Whether to use the full dataset or subsets, Integer Boolean  |
| --subset                  | -1                | If a specific subset number is meant to be run (will use 0 to 4 if -1 is given) |

To perform offline optimization for any CluStream variant, use ```clustream_microclusterer.py``` first to get the micro-clusters. This will produce mc and assign files in the ```param_data```-folder.


| **Parameter**        | **Default Value** | **Function**                                                                                                                                                                                                                                                                      |
|----------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --ds                  | densired10        | Dataset to perform the optimization on  |
| --use_full                  | 0                 | Whether to use the full dataset or subsets, Integer Boolean  |

Afterward, it is possible to run the offline optimization with ```parameter_estimator_offline.py```


| **Parameter**        | **Default Value** | **Function**  |
|----------------------|-------------------|--------------------------------------------------|
| --method                  | clustream         | CluStream variant to optimize for. Use the method keys. |
| --offline                  | kmeans            | Offline method to optimize. Use method keys.  |
| --ds                      | densired10        | Dataset to perform the optimization on  |
| --use_full                | 0                 | Whether to use the full dataset or subsets, Integer Boolean  |

## Offline Reconstruction Quality

To measure the offline reconstruction quality, the file ```reconstruction_quality.py``` is used.

| **Parameter**        | **Default Value** | **Function**  |
|----------------------|-------------------|--------------------------------------------------|
| --method                  | clustream         | CluStream variant to evaluate the offline reconstruction data for. |
| --ds                      | densired10        | Dataset to perform the evaluation on  |
| --index                | 0                 | Parameter index to evaluate the micro-clusters for  |
| --gen_folder                | ./gen_data                 | Where the generated points are stored (CluStream-G, CluStream-W, and CluStream-S) |
| --mc_folder                | ./mc_data                 | Where the generated points are stored (CluStream-G, CluStream-W, and CluStream-S) |
| --bop_folder                | ./bop                 | Where to store the BoP files. The BoP implementation uses the one from [here](https://github.com/Klaus-Tu/Bag-of-Prototypes) |
| --bop_centroids                | 100               | Number of centroids for BoP |
| --max_length                | np.inf               | Maximum evaluation length (uses minimum value of out of this and the full dataset): this was used to get results for running experiments, all results in the paper are from the full length of the dataset |
| --batch_size                | 1000               | Batch size for examination (must match evaluation length of stream clustering) |
| --value_scale                | 100               | Factor on all results (100 to get the value in percent) |

The MMD calculation is from the [Transfer Learning Repo](https://github.com/jindongwang/transferlearning/tree/master).

## Stream Clustering Methods

The paper was set up with several competitors as well as its own methods

| **Name**        | **Key** | **Where to get**  |
|----------------------|-------------|--------------------------------------------------|
| CluStream                  | clustream          | included, originally based on [River](https://github.com/online-ml/river). |
| CluStream-W                  | wclustream          | included |
| CluStream-S                  | scaledclustream          | included |
| CluStream-G                  | scope_full          | included |
| CluStream-O var. k           | clustream_no_offline          | included |
| CluStream-O fixed k           | clustream_no_offline_fixed          | included |
| STREAMKmeans                  | streamkmeans          | included, originally from [River](https://github.com/online-ml/river) |
| DenStream                  | denstream          | included, originally from [River](https://github.com/online-ml/river) |
| DBSTREAM                  | dbstream          | included, originally from [River](https://github.com/online-ml/river) |
| EMCStream | emcstream | [EMCStream repository](https://gitlab.com/alaettinzubaroglu/emcstream), modification required |
| MCMSTStream | mcmststream | [MCMSTStream repository](https://github.com/senolali/MCMSTStream), modification required |
| GB-FuzzyStream | gbfuzzystream | [GB-FuzzyStream repository](https://github.com/xjnine/GBFuzzyStream), modification required |


## Offline Clustering

This repository allows for 14 offline clustering methods to be used (additional ones are partially set up, but were excluded from the paper early on and as such may be incomplete)

| **Name**        | **Key** | **Where to get**  |
|----------------------|-------------|--------------------------------------------------|
| k-Means                  | kmeans          | [Scikit-Learn](https://scikit-learn.org) |
| Weighted k-Means                  | wkmeans          | [Scikit-Learn](https://scikit-learn.org) |
| SubkMeans                  | subkmeans          | [ClustPy](https://github.com/collinleiber/ClustPy) |
| X-Means                  | xmeans          | [ClustPy](https://github.com/collinleiber/ClustPy) |
| Projected Dip-Means                  | projdipmeans          | [ClustPy](https://github.com/collinleiber/ClustPy) |
| Spectral Clustering                  | spectral          | [Scikit-Learn](https://scikit-learn.org) |
| SCAR                 | scar          | [SCAR repository](https://github.com/SpectralClusteringAcceleratedRobust/SCAR) (extract into folder ```offline_methods/SCAR```)|
| SpectACl                 | spectacl          | [SpectACl repository](https://bitbucket.org/Sibylse/spectacl) (extract into folder ```offline_methods/spectacl```)|
| DBSCAN                 | dbscan          | [Scikit-Learn](https://scikit-learn.org) |
| HDBSCAN                 | hdbscan          | [Scikit-Learn](https://scikit-learn.org) |
| RNN-DBSCAN                 | rnndbscan          | already in repository in folder ```offline_methods``` |
| MDBSCAN                 | mdbscan          | already in repository in folder ```offline_methods``` |
| DPC                 | dpca          | [DPCA repository](https://github.com/colinwke/dpca) (take the ```cluster.py```-file, rename it to ```DPC.py``` and put it into the folder ```offline_methods/DPC```)|
| SNN-DPC                 | snndpc          | [SNN-DPC repository](https://github.com/liurui39660/SNNDPC) (take the ```SNNDPC.py```-file and put it into the folder ```offline_methods```)|
| DBHD                 | dbhd          | already in repository in folder ```offline_methods``` |
| CluStream-O k=100/x | nooffline | included, only available for CluStream |

## External Content

Additional datasets were taken from the [USP DS repository](https://sites.google.com/view/uspdsrepository), [Computational Intelligence Group @ UFSCar's data stream repository](https://github.com/CIG-UFSCar/DS_Datasets) and [Tomas Barton's Clustering benchmark repository](https://github.com/deric/clustering-benchmark).

## Experiment Files

We added most of the files in /dicts that store the results to the repository, however the full result pkl-files for KDDCUP99 for CluStream, CluStream-S and CluStream-G were too large to be added on GitHub, as such we only added the parameters, as well as the summary reports of the default, default_best (parameter optimization for default online paramters) and best runs for these experiments.
