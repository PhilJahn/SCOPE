# Repository for Going Offline: An Evaluation of the Offline Phase in Stream Clustering

This repository contains the code for the paper submission "Going Offline: An Evaluation of the Offline Phase in Stream Clustering" for ECML PKDD 2025.
It is based on code from the [River Stream learning repository](https://github.com/online-ml/river).

## Usage

The main code used to perform the evaluation is ```runMethod.py```. It allows for the execution of both the CluStream variants and the competitors. CluStream, DenStream, DBSTREAM, and STREAMKmeans come from the River repository and are included as is. The other competitors need to be obtained from their repositories as linked below in the section **External Content**. They require some modification to use the functions called for them (specifically learn_one/learn and predict_one/predict).
The main method has several parameters that allow for c

| **Parameter**        | **Default Value** | **Function**                                                                                                                                                                                                                                                                      |
|----------------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --method                  | clustream          | Stream Clustering method to evaluate (CluStream-W is ```wclustream```, CluStream-S is ```scaledclustream```, CluStream-G is ```scope_full```)  |
| --ds                  | complex9          | Dataset to perform the experiments one  |
| --offline                  | 1000          | Timesteps for offline phase/for evaluation  |
| --sumlimit                  | 100          | Maximal number of micro-clusters  |
| --gennum                  | 1000          | Approximate number of points that CluStream-S and CluStream-G produce  |
| --category                  | all | Which offline algorithms to include (For the paper, we used ```all``` for all offline algorithms, ```not_projdipmeans``` to run everything but Projected Dip-Means or the keys for the specific offline clustering methods) |
| --startindex  | 0          | Starting index for configuration (to only run on a subset of all parameters), inclusive |
| --endindex                  | np.inf        | Stopping index for configuration (to only run on a subset of all parameters), inclusive |
| --automl                  | 1         | Whether to use parameters obtained through AutoML or to perform a grid search (requires a parameter dictionary for the desired  dataset and stream clustering setup), Integer Boolean  |
| --used_full                  | 1          | Whether AutoML used subsampled data or the full dataset, Integer Boolean  |


## External Content


Additional datasets were taken from the [USP DS repository](https://sites.google.com/view/uspdsrepository) and [Tomas Barton's Clustering benchmark repository](https://github.com/deric/clustering-benchmark).
