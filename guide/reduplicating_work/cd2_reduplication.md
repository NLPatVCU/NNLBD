Evaluating Hallmarks of Cancer (HOC) Datasets Using the Reduplicated CD-2 Model
===============================================================================

To reproduce this study, we use the [CD-2 Model](./../cd2_redup_model/README.md) to train and evaluate on the [cancer landmark discovery](https://lbd.lionproject.net/downloads), or Hallmarks of Cancer (HOC), datasets. These datasets are used to identify five recent literature-based discoveries which include identifying hallmarks of cancer through implicit *A-B-C* relationship triplets prior to their known year of discovery. They support both open and closed discovery.

**NOTE: This work reproduces the model for closed discovery only.**

# Table Of Contents
1. [Easy Method](#easy_method)
2. [Manual Method](#manual_method)


# Easy Method <a name="easy_method"></a>

This method can be reduplicated using our pre-configured JSON configuration files, the included datasets and embeddings. Run one of the following commands below to reduplicate the model.

## Feature Scaled

| Dataset |                                     Command                                   |
|:-------:|:-----------------------------------------------------------------------------:|
|   HOC1  | `python LBDDriver ../json_files/HOC/cd2_experiments/feature_scaling/cs1.json` |
|   HOC2  | `python LBDDriver ../json_files/HOC/cd2_experiments/feature_scaling/cs2.json` |
|   HOC3  | `python LBDDriver ../json_files/HOC/cd2_experiments/feature_scaling/cs3.json` |
|   HOC4  | `python LBDDriver ../json_files/HOC/cd2_experiments/feature_scaling/cs4.json` |
|   HOC5  | `python LBDDriver ../json_files/HOC/cd2_experiments/feature_scaling/cs5.json` |

## Non-Feature Scaled

| Dataset |                                        Command                                   |
|:-------:|:--------------------------------------------------------------------------------:|
|   HOC1  | `python LBDDriver ../json_files/HOC/cd2_experiments/no_feature_scaling/cs1.json` |
|   HOC2  | `python LBDDriver ../json_files/HOC/cd2_experiments/no_feature_scaling/cs2.json` |
|   HOC3  | `python LBDDriver ../json_files/HOC/cd2_experiments/no_feature_scaling/cs3.json` |
|   HOC4  | `python LBDDriver ../json_files/HOC/cd2_experiments/no_feature_scaling/cs4.json` |
|   HOC5  | `python LBDDriver ../json_files/HOC/cd2_experiments/no_feature_scaling/cs5.json` |

As the models finish training, they will produce their respective directories according to the `model_save_path` setting.

```
../saved_models/*
```

Each directory contains the following files:

```
evaluation_rank_vs_epoch.png         <- Plotted model reported metric graph
evaluation_ties_vs_epoch.png         <- Plotted model reported metric graph
model_config.json                    <- Saved model Keras configuration file
model_metrics.txt                    <- TSV list of model reported evaluation metrics
model_settings.cfg                   <- NNLBD model configuration file (Do Not Edit!)
model_token_id_key_data              <- Input/Output Term Mappings
model.h5                             <- The saved model
<name_of_configuration_file>.json    <- Copy of your configuration file
training_accuracy_vs_epoch.png       <- Plotted model reported metric graph
training_f1_vs_epoch.png             <- Plotted model reported metric graph
training_loss_vs_epoch.png           <- Plotted model reported metric graph
training_precision_vs_epoch.png      <- Plotted model reported metric graph
training_recall_vs_epoch.png         <- Plotted model reported metric graph
```


# Manual Method <a name="manual_method"></a>

Requirements
============
 - [Python 2.7](python.org)
 - [Python 3.6.x to 3.10.x](python.org)
 - TensorFlow 1.15.2 to 2.9.0
 - NNLBD package
 - [Cancer landmark discovery datasets](https://lbd.lionproject.net/downloads)
 - [Neural networks for open and closed Literature-based Discovery](https://github.com/cambridgeltl/nn_for_LBD) (NN for LBD) Python 2.7 package


*A-priori* Pre-processing
=========================

*NOTE:* This step depends on the [NN for LBD package](https://github.com/cambridgeltl/nn_for_LBD). Download this package and unarchive it. We will refer to the extracted package's root directory as `nn_for_LBD`. Next, setup a virtual environment according to their instructions. This package is used to perform the necessary *a-priori* data preprocessing setups to create our training datasets, evaluation datasets, and their accompanying word embeddings.

**Windows Users: Their package was exclusively developed for Linux operating systems. It can be run under windows using a Linux bash simulated environment. We recommend [git-for-windows](https://gitforwindows.org/) through CMD or using [CMDER](https://github.com/cmderdev/cmder). Setting up a Linux VM is recommended. We've tested their package using [Linux Mint](https://linuxmint.com/) without issue. However, after you've generated your datasets and embeddings, you can remove the VM and all requirements associated with their package.**

***WARNING: The 'NN for LBD' package does not normalize text (i.e. lowercase) when generating the training datasets, evaluation datasets, and word embeddings. Our case study modifies this package to lowercase all text prior to generating these datasets and embeddings. This forces the generation of unique concept relationship triplets and embeddings. In its original state, their package potentially generates multiple variations of various concepts within relationship triplets and their embeddings (i.e. this depends on text-casing within the data used to generate these datasets and embeddings, and the data is cased).***

The HOC datasets requires some *a-priori* processing before running through the NNLBD system. As described [here](https://lbd.lionproject.net/downloads), each dataset contains three files:

 - nodes.csv
 - edges.csv
 - meta.csv

After your downloads are complete, unarchive the data files within the `nn_for_LBD/data` directory (i.e. the directory where you've setup the 'NN for LBD' package). The resulting directory structure should appear similar to below:

```
./nn_for_LBD/data/PR000001138_PR000006736/edges.csv
              .../PR000001138_PR000006736/nodes.csv
              .../PR000001138_PR000006736/meta.csv
              .../PR000001754_MESHD000236/edges.csv
              .../PR000001754_MESHD000236/nodes.csv
              .../PR000001754_MESHD000236/meta.csv
              .../PR000006066_MESHD013964/edges.csv
              .../PR000006066_MESHD013964/nodes.csv
              .../PR000006066_MESHD013964/meta.csv
              .../PR000011170_MESHD010190/edges.csv
              .../PR000011170_MESHD010190/nodes.csv
              .../PR000011170_MESHD010190/meta.csv
              .../PR000011331_PR000005308/edges.csv
              .../PR000011331_PR000005308/nodes.csv
              .../PR000011331_PR000005308/meta.csv
```


We're only intersted in the file: `edges.csv`. So the remaining files can be deleted to save space, but we recommended keeping all of them until you've generated all necessary files. The `edges.csv` file is a comma-separated file which contains edge relationships between vertices in a knowledge graph. Within these files, we're only interested in a few columns:

 - ":START_ID"
 - ":END_ID"
 - "year:int"
 - "metric_jaccard:float[]"

With this column data, we can compose the new file: `edges_with_scores.csv`, for each dataset. We have provided the Perl script [create_edges_with_scores_file.pl](/miscellaneous_scripts/create_edges_with_scores_file.pl) to simply this step. Set the `$edges_csv_file_path` and `$edges_with_scores_path` variables accordingly and execute the file. The script will create the `edges_with_score.csv` file using the specified `$edges_with_scores_path` variable path.

To do this manually, just remove all other columns outside of what we've listed above. This must also be done for each HOC dataset and will result in the following directory structure.

```
./nn_for_LBD/data/PR000001138_PR000006736/edges_with_scores.csv
              .../PR000001754_MESHD000236/edges_with_scores.csv
              .../PR000006066_MESHD013964/edges_with_scores.csv
              .../PR000011170_MESHD010190/edges_with_scores.csv
              .../PR000011331_PR000005308/edges_with_scores.csv
```

Now, activate your Python 2.7 virtual environment for the `nn_for_LBD` package and navigate to the directory `./nn_for_LBD`. We need to edit the file: `experiment_batch_cases.sh`.

At the beginning of the file, the following variables must be changed to generate our training and evaluation sets, and our word embeddings.

```bash
setup_experiment=False
create_representations=False
do_lbd=True
```

to

```bash
setup_experiment=True
create_representations=True
do_lbd=False
```

Execute the script via one of the following commands below:

```bash
./experiment_batch_cases.sh
sh experiment_batch_cases.sh
bash experiment_batch_cases.sh
```

*NOTE: This will take a while, so take a break, catch-up an episode of your favorite show for the moment. But don't forget to come back. See you later.*

After this is finished, you will be left with the files in the `nn_for_LBD` directory:

```
Training/Evaluation Dataset Files:
==================================
./train_cs1_closed_discovery_without_aggregators.tsv
./train_cs2_closed_discovery_without_aggregators.tsv
./train_cs3_closed_discovery_without_aggregators.tsv
./train_cs4_closed_discovery_without_aggregators.tsv
./train_cs5_closed_discovery_without_aggregators.tsv
./test_cs1_closed_discovery_without_aggregators.tsv
./test_cs2_closed_discovery_without_aggregators.tsv
./test_cs3_closed_discovery_without_aggregators.tsv
./test_cs4_closed_discovery_without_aggregators.tsv
./test_cs5_closed_discovery_without_aggregators.tsv

Word Embedding Files:
=====================
./test_modified_cs1.embeddings.bin
./test_modified_cs2.embeddings.bin
./test_modified_cs3.embeddings.bin
./test_modified_cs4.embeddings.bin
./test_modified_cs5.embeddings.bin
```

*NOTE: The shell script generates `plain text` embeddings, then converts them to `binary` vectors. You can use either variant. The system will automatically determine which has been selected and load them accordingly.*

Now we are ready to begin LBD experimentation using the CD-2 model. You may remove the `./nn_for_LBD` directory and any associated files. **Please, keep the aforementioned files before removing the main `./nn_for_LBD` directory.**


Preparing The Model
===================

Now that we have prepared our HOC datasets (training and evaluation), and our accompanying embeddings, we can begin training and evaluating the performance of the model. In order to do this, we must generate a configuration file. We first need to specify model parameters:


```json
{
    "crichton_closed_discovery_train_and_eval_1": [
        {
            "network_name"       : "cd2",
            "model_type"         : "closed_discovery",
            "activation_function": "softplus",
            "loss_function"      : "binary_crossentropy",
            "embedding_path"     : "<path_to_file>",
            "train_data_path"    : "<path_to_file>",
            "model_save_path"    : "<path_to_file>",
            "gold_b_instance"    : "<gold_abc_triplet_relationship",
        }
    ]
}
```

It's important to know what our gold *A-B-C* triplet relationship is. This is necessary to evaluate (i.e. rank) our gold *B* concept among all remaining concepts in the vocabulary for closed discovey. The HOC datasets provide these relationships [here](https://lbd.lionproject.net/downloads). However, for convenience, we list them at the bottom of this page.

An example of a complete JSON configuration file is shown below:

```json
{
    "global_settings": [
        {
            "_comment": "Global Variable Settings",
            "device_name": "/gpu:0",
            "number_of_iterations": 5
        }
    ],
    "crichton_closed_discovery_train_and_eval_1": [
        {
            "_comment": "CD2 Model - CS1 Reduced",
            "network_model": "cd2",
            "model_type": "closed_discovery",
            "activation_function": "softplus",
            "loss_function": "binary_crossentropy",
            "embedding_path": "../vectors/HOC/test_modified_cs1.embeddings.bin",
            "train_data_path": "../data/HOC/train_cs1_closed_discovery_without_aggregators.tsv",
            "eval_data_path": "../data/HOC/test_cs1_closed_discovery_without_aggregators.tsv",
            "model_save_path": "../saved_models/cd2_cs1_model",
            "epochs": 300,
            "verbose": 2,
            "learning_rate": 0.001,
            "batch_size": 1024,
            "dropout": 0.1,
            "run_eval_number_epoch": 1,
            "embedding_modification": "concatenate",
            "gold_b_instance": "PR:000001754\tPR:000002307\tMESH:D000236\t0.0",
            "set_per_iteration_model_path": "True",
            "restrict_output": "True",
            "feature_scale_value": 10.0
        }
    ]
}
```

*NOTE: This is configuration file only includes the minimum requirements to begin replicating our experiments. Other settings may need to be changed depending on the desired experiment you wish to run. We provide a complete listing of all configuration file settings and their descriptions [here](./../configuration_file.md).*

This configuration file generates a CD-2 closed discovery model, trains the model over the HOC1 dataset (CS1), and evaluates the model over the HOC1 (CS1) test dataset using the evaluation datasets and the gold *B* relation *PR:000001754 -> PR:000002307 -> MESH:D000236*. In this setup, we're ***concatenating*** our *A*, *B*, and *C* input representations. However, we can also compare performance by changing this setting to ***average*** or ***Hadamard*** representations.


Running Our Model
=================

To begin model training and evaluation, we execute the model using the configuration file as shown below:

```cmd
python LBDDriver.py <name_of_configuration_file>.json
```


Pre-processing
==============

Since each dataset contains *A-B-C* triplets separated by tab characters, minimal pre-processing is needed. When reading the training datasets, evaluation datasets, and word embeddings, we lowercase all text. This reduces the chance that the case text will produce variations of the same concept elements within the input and output vocabularies. We performed a similar step during the *A-priori Pre-processing* step. Thus, we've mitigated this issue for our word embeddings.


Model Training
==============

During training, the model will report training and evaluation metrics after every training epoch. This can be changed via the setting: `run_eval_number_epoch`. Example output for one model run is shown below:


```bash
Building LBD Experiment Run ID: crichton_closed_discovery_train_and_eval_1

BaseModel::Initialize_GPU() - CUDA Supported / GPU Is Available
BaseModel::Initialize_GPU() - GPU/CUDA Supported And Enabled
Preparing Evaluation Data
Beginning Model Data Preparation/Model Training
LBD::Prepare_Model_Data() - Loading Embeddings: ../vectors/HOC/test_modified_cs1.embeddings.bin
LBD::Prepare_Model_Data() - Reading Training Data: ../data/HOC/train_cs1_closed_discovery_without_aggregators.tsv
LBD::Prepare_Model_Data() - Generating Token IDs From Training Data
LBD::Prepare_Model_Data() - Binarizing/Vectorizing Model Inputs & Outputs From Training Data
CrichtonDataLoader::Worker_Thread_Function() - Warning: Vectorizing Input/Output Data - Element Does Not Exist
                                     - Line: "node1     node2   node3   label"
CD2Model::Fit() - Executing Model Training
196/196 - 4s - loss: 0.3981 - accuracy: 0.8490 - Precision: 0.9122 - Recall: 0.7692 - F1_Score: 0.8328 - 4s/epoch - 19ms/step
CD2Model::Fit() - Finished Model Training
Epoch : 1 - Gold B: PR:000002307 - Rank: 379 Of 2294 Number Of B Terms - Score: 2.1351935863494873 - Number Of Ties: 0
CD2Model::Fit() - Executing Model Training
196/196 - 2s - loss: 0.3347 - accuracy: 0.8706 - Precision: 0.9472 - Recall: 0.7864 - F1_Score: 0.8567 - 2s/epoch - 9ms/step
CD2Model::Fit() - Finished Model Training
Epoch : 2 - Gold B: PR:000002307 - Rank: 221 Of 2294 Number Of B Terms - Score: 2.325946092605591 - Number Of Ties: 0
CD2Model::Fit() - Executing Model Training
196/196 - 2s - loss: 0.2660 - accuracy: 0.9079 - Precision: 0.9513 - Recall: 0.8599 - F1_Score: 0.9028 - 2s/epoch - 8ms/step
CD2Model::Fit() - Finished Model Training
Epoch : 3 - Gold B: PR:000002307 - Rank: 219 Of 2294 Number Of B Terms - Score: 2.6589300632476807 - Number Of Ties: 0
CD2Model::Fit() - Executing Model Training
196/196 - 2s - loss: 0.2744 - accuracy: 0.9007 - Precision: 0.9512 - Recall: 0.8457 - F1_Score: 0.8923 - 2s/epoch - 9ms/step
CD2Model::Fit() - Finished Model Training
Epoch : 4 - Gold B: PR:000002307 - Rank: 26 Of 2294 Number Of B Terms - Score: 3.2427830696105957 - Number Of Ties: 1
CD2Model::Fit() - Executing Model Training
196/196 - 2s - loss: 0.2266 - accuracy: 0.9245 - Precision: 0.9554 - Recall: 0.8911 - F1_Score: 0.9214 - 2s/epoch - 9ms/step
CD2Model::Fit() - Finished Model Training
Epoch : 5 - Gold B: PR:000002307 - Rank: 60 Of 2294 Number Of B Terms - Score: 3.354067087173462 - Number Of Ties: 1
...

Epoch: 1 - Rank: 379 - Value: 2.1351935863494873 - Number Of Ties: 0
Epoch: 2 - Rank: 221 - Value: 2.325946092605591 - Number Of Ties: 0
Epoch: 3 - Rank: 219 - Value: 2.6589300632476807 - Number Of Ties: 0
Epoch: 4 - Rank: 26 - Value: 3.2427830696105957 - Number Of Ties: 1
Epoch: 5 - Rank: 60 - Value: 3.354067087173462 - Number Of Ties: 1
...

Generating Model Metric Charts

Best Rank: 26
Best Ranking Epoch: 4
Number Of Ties With Best Rank: 1

LBD::Save_Model() - Saving Model To Path: ./saved_models/cd2_cs1_model_1

...
```

*NOTE: We only show a sample model run using 5 epochs, and omit some information to reduce the log output for easier viewing.*

If you do not wish to see this output or wish to continue using the same terminal/bash session, we recommend using `nohup` or redirecting the terminal std output to a file via the `redirection operator` command below:

```cmd
python LBDDriver.py <name_of_configuration_file>.json > model_output.log
```

You can retrieve the running process back using the `fg` command, if you wish to terminate it.


Model Evaluation
================

After our model has finished training, it will produce the following directories:

```
../saved_models/cd2_cs1_model_1
../saved_models/cd2_cs1_model_2
../saved_models/cd2_cs1_model_3
../saved_models/cd2_cs1_model_4
../saved_models/cd2_cs1_model_5
```

Each directory contains the following files:

```
evaluation_rank_vs_epoch.png         <- Plotted model reported metric graph
evaluation_ties_vs_epoch.png         <- Plotted model reported metric graph
model_config.json                    <- Saved model Keras configuration file
model_metrics.txt                    <- TSV list of model reported evaluation metrics
model_settings.cfg                   <- NNLBD model configuration file (Do Not Edit!)
model_token_id_key_data              <- Input/Output Term Mappings
model.h5                             <- The saved model
<name_of_configuration_file>.json    <- Copy of your configuration file
training_accuracy_vs_epoch.png       <- Plotted model reported metric graph
training_f1_vs_epoch.png             <- Plotted model reported metric graph
training_loss_vs_epoch.png           <- Plotted model reported metric graph
training_precision_vs_epoch.png      <- Plotted model reported metric graph
training_recall_vs_epoch.png         <- Plotted model reported metric graph
```

These files include the saved model, configuration settings to load and re-use the model, as well as pertinent model evaluation details. The image files provide a visual depiction of the training and evaluation metrics provided by the model. However, the `model_metrics.txt` file is what we're interested in. This file contains the evaluation metrics and other important details used to determine our model's efficacy. An example of this file is provided below. As the model randomizes its layer weights upon creation, we recommend averaging the performance metrics over multiple model runs to detemine performance (e.g. average over 5 runs).

| Epoch |    Gold B    | Rank | # Of Ties |        Score       |	# Of B Terms |         Loss        |      Accuracy      |      Precision     |       Recall      |    F1_Score         |
|:-----:|:------------:|:----:|:---------:|:------------------:|:---------------:|:-------------------:|:------------------:|:------------------:|:-----------------:|:---------------:|
|   1   | PR:000002307 | 379  |     0     | 2.1351935863494873 |       2294      | 0.3980754017829895  | 0.8489750027656555 | 0.9121790528297424 | 0.769170343875885  | 0.8328071236610413 |
|   2   | PR:000002307 | 221  |     0     | 2.325946092605591  |       2294      | 0.3346840739250183  | 0.870639979839325  | 0.9471961855888367 | 0.7863627076148987 | 0.8566874265670776 |
|   3   | PR:000002307 | 219  |     0     | 2.6589300632476807 |       2294      | 0.2659952640533447  | 0.9078500270843506 | 0.9513208866119385 | 0.8599274158477783 | 0.9028177261352539 |
|   4   | PR:000002307 | 26   |     1     | 3.2427830696105957 |       2294      | 0.2743602991104126  | 0.9006749987602234 | 0.951215386390686  | 0.8456607460975647 | 0.892277181148529  |
|   5   | PR:000002307 | 60   |     1     | 3.354067087173462  |       2294      | 0.22662276029586792 | 0.9245499968528748 | 0.9553894400596619 | 0.8911453485488892 | 0.9214124083518982 |


Final Notes
===========

Preliminary testing of this models shows it performs well on the HOC datasets. This model is redupilcated using TensorFlow/Keras and later versions of Python to compare its performance to future work. We recommend consulting the original authors manuscript [here](https://doi.org/10.1371/journal.pone.0232891) for a comprehensive evaluation of the model's performance.


True HOC Relationships
======================
|   Dataset  |         A Term        |         B Term       |              C Term              |
|:----------:|:---------------------:|:--------------------:|:--------------------------------:|
| HOC1 (CS1) | NF-B (PR:000001754)   | Bcl-2 (PR:000002307) | Adenoma (MESH:D000236)           |
| HOC2 (CS2) | NOTCH1 (PR:000011331) | senescence (HOC:42)  | C/EBP (PR:000005308)             |
| HOC3 (CS3) | IL-17 (PR:000001138)  | p38 (PR:000003107)   | MKP-1 (PR:000006736)             |
| HOC4 (CS4) | Nrf2 (PR:000011170)   | ROS (CHEBI:26523)    | pancreatic cancer (MESH:D010190) |
| HOC5 (CS5) | CXCL12 (PR:000006066) | senescence (HOC:42)  | thyroid cancer (MESH:D013964)    |