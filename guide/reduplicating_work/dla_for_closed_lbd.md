Exploring a Neural Network Architecture for Closed Literature-based Discovery
=============================================================================

In this study, we use our [Base Multi-Label Models](./../base_ml_model/README.md) to train and evaluate on the [cancer landmark discovery](https://lbd.lionproject.net/downloads), or Hallmarks of Cancer (HOC), datasets. These datasets are used to identify five recent literature-based discoveries which include identifying hallmarks of cancer through implicit *A-B-C* relationship triplets prior to their known year of discovery. They support both open and closed discovery. Our study evaluates our model's performance for closed discovery.


# Table Of Contents
1. [Easy Method](#easy_method)
2. [Manual Method](#manual_method)
3. [Reproducing Our Random Experiments](#reproducing_random_experiments)


# Easy Method <a name="easy_method"></a>

This method can be reduplicated using our pre-configured JSON configuration files, the included datasets and embeddings. Run one of the following commands below to reduplicate the model.

## Feature Scaled

| Dataset |  Input Type |   Output Type  |                                                   Command                                                   |
|:-------:|:-----------:|:--------------:|:-----------------------------------------------------------------------------------------------------------:|
|   HOC1  |   Average   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs1_hinton_sigmoid_avg_fo.json`      |
|         |   Average   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs1_hinton_sigmoid_avg_ro.json`      |
|         | Concatenate |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs1_hinton_sigmoid_concat_fo.json`   |
|         | Concatenate | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs1_hinton_sigmoid_concat_ro.json`   |
|         |  Hadamard   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs1_hinton_sigmoid_hadamard_fo.json` |
|         |  Hadamard   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs1_hinton_sigmoid_hadamard_ro.json` |
|   HOC2  |   Average   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs2_hinton_sigmoid_avg_fo.json`      |
|         |   Average   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs2_hinton_sigmoid_avg_ro.json`      |
|         | Concatenate |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs2_hinton_sigmoid_concat_fo.json`   |
|         | Concatenate | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs2_hinton_sigmoid_concat_ro.json`   |
|         |  Hadamard   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs2_hinton_sigmoid_hadamard_fo.json` |
|         |  Hadamard   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs2_hinton_sigmoid_hadamard_ro.json` |
|   HOC3  |   Average   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs3_hinton_sigmoid_avg_fo.json`      |
|         |   Average   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs3_hinton_sigmoid_avg_ro.json`      |
|         | Concatenate |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs3_hinton_sigmoid_concat_fo.json`   |
|         | Concatenate | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs3_hinton_sigmoid_concat_ro.json`   |
|         |  Hadamard   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs3_hinton_sigmoid_hadamard_fo.json` |
|         |  Hadamard   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs3_hinton_sigmoid_hadamard_ro.json` |
|   HOC4  |   Average   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs4_hinton_sigmoid_avg_fo.json`      |
|         |   Average   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs4_hinton_sigmoid_avg_ro.json`      |
|         | Concatenate |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs4_hinton_sigmoid_concat_fo.json`   |
|         | Concatenate | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs4_hinton_sigmoid_concat_ro.json`   |
|         |  Hadamard   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs4_hinton_sigmoid_hadamard_fo.json` |
|         |  Hadamard   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs4_hinton_sigmoid_hadamard_ro.json` |
|   HOC5  |   Average   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs5_hinton_sigmoid_avg_fo.json`      |
|         |   Average   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs5_hinton_sigmoid_avg_ro.json`      |
|         | Concatenate |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs5_hinton_sigmoid_concat_fo.json`   |
|         | Concatenate | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs5_hinton_sigmoid_concat_ro.json`   |
|         |  Hadamard   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs5_hinton_sigmoid_hadamard_fo.json` |
|         |  Hadamard   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/feature_scaling/cs5_hinton_sigmoid_hadamard_ro.json` |

## Non-Feature Scaled

| Dataset |  Input Type |   Output Type  |                                                     Command                                                    |
|:-------:|:-----------:|:--------------:|:--------------------------------------------------------------------------------------------------------------:|
|   HOC1  |   Average   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs1_hinton_sigmoid_avg_fo.json`      |
|         |   Average   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs1_hinton_sigmoid_avg_ro.json`      |
|         | Concatenate |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs1_hinton_sigmoid_concat_fo.json`   |
|         | Concatenate | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs1_hinton_sigmoid_concat_ro.json`   |
|         |  Hadamard   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs1_hinton_sigmoid_hadamard_fo.json` |
|         |  Hadamard   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs1_hinton_sigmoid_hadamard_ro.json` |
|   HOC2  |   Average   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs2_hinton_sigmoid_avg_fo.json`      |
|         |   Average   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs2_hinton_sigmoid_avg_ro.json`      |
|         | Concatenate |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs2_hinton_sigmoid_concat_fo.json`   |
|         | Concatenate | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs2_hinton_sigmoid_concat_ro.json`   |
|         |  Hadamard   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs2_hinton_sigmoid_hadamard_fo.json` |
|         |  Hadamard   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs2_hinton_sigmoid_hadamard_ro.json` |
|   HOC3  |   Average   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs3_hinton_sigmoid_avg_fo.json`      |
|         |   Average   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs3_hinton_sigmoid_avg_ro.json`      |
|         | Concatenate |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs3_hinton_sigmoid_concat_fo.json`   |
|         | Concatenate | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs3_hinton_sigmoid_concat_ro.json`   |
|         |  Hadamard   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs3_hinton_sigmoid_hadamard_fo.json` |
|         |  Hadamard   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs3_hinton_sigmoid_hadamard_ro.json` |
|   HOC4  |   Average   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs4_hinton_sigmoid_avg_fo.json`      |
|         |   Average   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs4_hinton_sigmoid_avg_ro.json`      |
|         | Concatenate |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs4_hinton_sigmoid_concat_fo.json`   |
|         | Concatenate | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs4_hinton_sigmoid_concat_ro.json`   |
|         |  Hadamard   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs4_hinton_sigmoid_hadamard_fo.json` |
|         |  Hadamard   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs4_hinton_sigmoid_hadamard_ro.json` |
|   HOC5  |   Average   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs5_hinton_sigmoid_avg_fo.json`      |
|         |   Average   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs5_hinton_sigmoid_avg_ro.json`      |
|         | Concatenate |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs5_hinton_sigmoid_concat_fo.json`   |
|         | Concatenate | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs5_hinton_sigmoid_concat_ro.json`   |
|         |  Hadamard   |   Full Output  | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs5_hinton_sigmoid_hadamard_fo.json` |
|         |  Hadamard   | Reduced Output | `python LBDDriver ../json_files/HOC/hinton_experiments/no_feature_scaling/cs5_hinton_sigmoid_hadamard_ro.json` |

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
 - Perl 5.x (Not really necessary/Only used during *a-priori* pre-processing)
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

*NOTE: You may remove all other files and the `NN for LBD` package, and its requirements if you wish. They're no longer necessary.*

These files are almost ready for experimentation. However, the training and testing files contain negative samples which are not utilized within our study. Our model automatically creates negative samples considering its output space is multi-class (multi-label really). The approach taken by their study utilizes these datasets to train and evaluate their models using single-class classification. Thus, negative samples are needed to generalize their model. However, we repurpose these datasets within our study. To remove these samples, along with other unnecessary information, we recommend using our [convert_hoc_data_to_nnlbd_format_v2.py](/miscellaneous_scripts/convert_hoc_data_to_nnlbd_format_v2.py) script. Edit the variables `file_path` and `new_file_path` to make these changes. If you wish to perform this manually, omit the `label` column within each dataset and any instances with label `0.0` (e.g. these are negative sample instances). Also remove the header line (i.e. first line): '`node1 node2 node3 label`'.

Lastly, these files are in `open discovery format` (i.e. `a_concept b_concept c_concept`) and must be converted to `closed discovery format` (i.e. `a_concept c_concept b_concepts`). To accomplish this, we provide our [convert_nnlbd_open_discovery_data_to_closed_discovery_format.py](/miscellaneous_scripts/convert_nnlbd_open_discovery_data_to_closed_discovery_format.py) script. Set the parameters `file_path` and `new_file_path` accordingly, and run the script to convert the data.

Let's say these newly converted files follow the directory structure below:

```
Training/Evaluation Dataset File:
=================================
./train_cs1_closed_discovery_without_aggregators_mod
./train_cs2_closed_discovery_without_aggregators_mod
./train_cs3_closed_discovery_without_aggregators_mod
./train_cs4_closed_discovery_without_aggregators_mod
./train_cs5_closed_discovery_without_aggregators_mod
./test_cs1_closed_discovery_without_aggregators_mod
./test_cs2_closed_discovery_without_aggregators_mod
./test_cs3_closed_discovery_without_aggregators_mod
./test_cs4_closed_discovery_without_aggregators_mod
./test_cs5_closed_discovery_without_aggregators_mod

Word Embedding Files:
=====================
./test_modified_cs1.embeddings.bin
./test_modified_cs2.embeddings.bin
./test_modified_cs3.embeddings.bin
./test_modified_cs4.embeddings.bin
./test_modified_cs5.embeddings.bin
```

*NOTE: The shell script generates `plain text` embeddings, then converts them to `binary` vectors. You can use either variant. The system will automatically determine which has been selected and load them accordingly.*

Now we are ready to begin LBD experimentation using the Multi-Label Models. You may remove the `./nn_for_LBD` directory and any associated files. **Please, keep the aforementioned files before removing the main `./nn_for_LBD` directory.**


Preparing Our Model
===================

Now that we have prepared our HOC datasets (training and evaluation), and our accompanying embeddings, we can begin training and evaluating the performance of the `Hinton` model. In order to do this, we must generate a configuration file. We first need to specify model parameters:

```json
{
    "network_name"   : "hinton",
    "model_type"     : "closed_discovery",
    "loss_function"  : "binary_crossentropy",
    "embedding_path" : "<path_to_file>",
    "train_data_path": "<path_to_file>",
    "model_save_path": "<path_to_file>",
    "gold_b_instance": "<gold_abc_triplet_relationship",
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
    "closed_discovery_train_and_eval_1": [
        {
            "_comment": "HOC1 Hinton - Closed Discovery",
            "network_model": "hinton",
            "model_type": "closed_discovery",
            "embedding_path": "../vectors/HOC/test_modified_cs1.embeddings.bin",
            "train_data_path": "../data/HOC/train_cs1_closed_discovery_without_aggregators_mod",
            "eval_data_path": "../data/HOC/test_cs1_closed_discovery_without_aggregators_mod",
            "model_save_path": "../saved_models/cs1_hinton_model",
            "epochs": 400,
            "verbose": 2,
            "learning_rate": 0.001,
            "batch_size": 256,
            "dropout": 0.1,
            "run_eval_number_epoch": 1,
            "embedding_modification": "average",
            "gold_b_instance": "PR:000001754\tPR:000002307\tMESH:D000236",
            "feature_scale_value": 10.0,
            "set_per_iteration_model_path": "True"
        }
    ]
}
```

*NOTE: This is configuration file only includes the minimum requirements to begin replicating our experiments. Other settings may need to be changed depending on the desired experiment you wish to run. We provide a complete listing of all configuration file settings and their descriptions [here](./../configuration_file.md).*

This configuration file generates a multi-label closed discovery model, trains the model over the HOC1 dataset (CS1), and evaluates the model over the HOC1 (CS1) test dataset using the gold *B* relation *PR:000001754 -> PR:000002307 -> MESH:D000236*. In this setup, we're ***averaging*** our *A* and *C* input representations. However, we can also compare performance by changing this setting to ***concatenation*** or ***Hadamard*** representations.


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

During training, the model will report training and evaluation metrics after every training epoch. This can be changed via the setting: `run_eval_number_epoch`. Example output is shown below:


```bash
Building LBD Experiment Run ID: closed_discovery_train_and_eval_1

BaseModel::Initialize_GPU() - CUDA Supported GPU Is Available
BaseModel::Initialize_GPU() - GPU/CUDA Supported And Enabled
Beginning Model Data Preparation/Model Training
LBD::Prepare_Model_Data() - Loading Embeddings: ../vectors/HOC/test_modified_cs1.embeddings.bin
LBD::Prepare_Model_Data() - Reading Training Data: ../data/HOC/train_cs1_closed_discovery_without_aggregators_mod
LBD::Prepare_Model_Data() - Generating Token IDs From Training Data
LBD::Prepare_Model_Data() - Binarizing/Vectorizing Model Inputs & Outputs From Training Data
RumelhartHintonModel::Fit() - Executing Model Training
782/782 - 331s - loss: 0.0711 - accuracy: 0.0091 - Precision: 5.1978e-07 - Recall: 0.0063 - F1_Score: 1.0338e-06
RumelhartHintonModel::Fit() - Finished Model Training
LBD::Fit() - Elapsed Time: 345.12 secs
Epoch : 1 - Gold B: PR:000002307 - Rank: 7314 Of 90701 Number Of B Terms - Score: 0.000302251 - Number Of Ties: 0
          - Eval Rank: 536 Of 2173 Number Of Eval B Terms - Score: 0.000302251 - Number Of Ties: 0
RumelhartHintonModel::Fit() - Executing Model Training
782/782 - 414s - loss: 3.4067e-04 - accuracy: 0.0124 - Precision: 0.0000e+00 - Recall: 0.0000e+00 - F1_Score: 0.0000e+00
RumelhartHintonModel::Fit() - Finished Model Training
LBD::Fit() - Elapsed Time: 414.33 secs
Epoch : 2 - Gold B: PR:000002307 - Rank: 476 Of 90701 Number Of B Terms - Score: 7.805802e-05 - Number Of Ties: 0
          - Eval Rank: 345 Of 2173 Number Of Eval B Terms - Score: 7.805802e-05 - Number Of Ties: 0
RumelhartHintonModel::Fit() - Executing Model Training
782/782 - 415s - loss: 1.9669e-04 - accuracy: 0.0124 - Precision: 0.0000e+00 - Recall: 0.0000e+00 - F1_Score: 0.0000e+00
RumelhartHintonModel::Fit() - Finished Model Training
LBD::Fit() - Elapsed Time: 415.19 secs
...
Epoch : 111 - Gold B: PR:000002307 - Rank: 2 Of 90701 Number Of B Terms - Score: 0.009079145 - Number Of Ties: 0
          - Eval Rank: 2 Of 2173 Number Of Eval B Terms - Score: 0.009079145 - Number Of Ties: 0
RumelhartHintonModel::Fit() - Executing Model Training
782/782 - 422s - loss: 8.6211e-05 - accuracy: 0.0370 - Precision: 0.0358 - Recall: 1.4486e-04 - F1_Score: 2.8852e-04
RumelhartHintonModel::Fit() - Finished Model Training
LBD::Fit() - Elapsed Time: 422.35 secs

Epoch: 1 - Rank: 7314 - Value: 0.000302251 - Number Of Ties: 0 - Eval Rank: 536 - Eval Value: 0.000302251 - Eval Number Of Ties: 0
Epoch: 2 - Rank: 476 - Value: 7.805802e-05 - Number Of Ties: 0 - Eval Rank: 345 - Eval Value: 7.805802e-05 - Eval Number Of Ties: 0
...
Epoch: 111 - Rank: 2 - Value: 0.009079145 - Number Of Ties: 0 - Eval Rank: 2 - Eval Value: 0.009079145 - Eval Number Of Ties: 0

Generating Model Metric Charts

Best Rank: 2
Best Ranking Epoch: 111
Number Of Ties With Best Rank: 0

Eval Best Rank: 2
Eval Best Ranking Epoch: 111
Eval Number Of Ties With Best Rank: 0
LBD::Save_Model() - Saving Model To Path: ./saved_models/cs1_hinton_model_1

...
```

*NOTE: We omit some information to reduce the log output for easier viewing.*

If you do not wish to see this output or wish to continue using the same terminal/bash session, we recommend using `nohup` or redirecting the terminal std output to a file via the `redirection operator` command below:

```cmd
python LBDDriver.py <name_of_configuration_file>.json > model_output.log
```

You can retrieve the running process back using the `fg` command, if you wish to terminate it.


Model Evaluation
================

After our model has finished training, it will produce the following directories:

```
../saved_models/cs1_hinton_model_1
../saved_models/cs1_hinton_model_2
../saved_models/cs1_hinton_model_3
../saved_models/cs1_hinton_model_4
../saved_models/cs1_hinton_model_5
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

These files include the saved model, configuration settings to load and re-use the model, as well as pertinent model evaluation details. The image files provide a visual depiction of the training and evaluation metrics provided by the model. However, the `model_metrics.txt` file is what we're interested in. This file contains the evaluation metrics and other important details used to determine our model's efficacy. An example of this file is provided below. We average the metrics of all 5 runs to detemine the performance of our model.

| Epoch |    Gold B    | Rank | # Of Ties |    Score      | # Of B Terms | Eval Rank | # Of Ties |     Score     | # Of Eval B Terms |          Loss          |       Accuracy           |        Precision      |         Recall         |        F1_Score       |
|:-----:|:------------:|:----:|:---------:|:-------------:|:------------:|:---------:|:---------:|:-------------:|:-----------------:|:----------------------:|:------------------------:|:---------------------:|:----------------------:|:---------------------:|
|   1   | PR:000002307 | 7314 |     0     | 0.000302251   |     90701    |    536    |     0     | 0.000302251   |        2173       | 0.07111504673957825    |   0.009080000221729279   | 5.197774726184434e-07 | 0.006268981844186783   | 1.0338306992707658e-06|
|   2   | PR:000002307 | 476  |     0     | 7.805802e-05  |     90701    |    345    |     0     | 7.805802e-05  |        2173       | 0.00034066830994561315 |   0.012384999543428421   |           0.0         |           0.0          |          0.0          |
|   3   | PR:000002307 | 479  |     0     | 5.0855582e-05 |     90701    |    348    |     0     | 5.0855582e-05 |        2173       | 0.00019669125322252512 |   0.012384999543428421   |           0.0         |           0.0          |          0.0          |
|   4   | PR:000002307 | 486  |     0     | 5.267593e-05  |     90701    |    355    |     0     | 5.267593e-05  |        2173       | 0.00016577007772866637 |   0.012384999543428421   |           0.0         |           0.0          |          0.0          |
|   5   | PR:000002307 | 522  |     0     | 6.364535e-05  |     90701    |    372    |     0     | 6.364535e-05  |        2173       | 0.0001559162774356082  |   0.012384999543428421   |           0.0         |           0.0          |          0.0          |
|  ...  | PR:000002307 | ...  |    ...    |      ...      |      ...     |    ...    |    ...    |      ...      |        ...        |           ...          |            ...           |           ...         |           ...          |          ...          |
|  111  | PR:000002307 |  2   |     0     | 0.009079145   |     90701    |     2     |     0     | 0.009079145   |        2173       | 8.69941504788585e-05   |   0.0346050001680851     | 0.02813299000263214   | 0.00010989449947373942 | 0.0002189337828895077 |


Final Notes
===========

Following these steps, you should be able to reproduce our work. However, exact reported rankings may not be observed. We do not freeze our model weights when building the model. Thus, between experimental runs, the evaluation rankings will differ. This is our reasoning for reporting the average among 5 runs. Our thinking behind this is, if the result is not truly random, then we should be able to obtain similar performance regardless of model weight randomization. Computing the averaged among all runs should obtain similar performance to our reported values.


True HOC Relationships
======================
|   Dataset  |         A Term        |         B Term       |              C Term              |
|:----------:|:---------------------:|:--------------------:|:--------------------------------:|
| HOC1 (CS1) | NF-B (PR:000001754)   | Bcl-2 (PR:000002307) | Adenoma (MESH:D000236)           |
| HOC2 (CS2) | NOTCH1 (PR:000011331) | senescence (HOC:42)  | C/EBP (PR:000005308)             |
| HOC3 (CS3) | IL-17 (PR:000001138)  | p38 (PR:000003107)   | MKP-1 (PR:000006736)             |
| HOC4 (CS4) | Nrf2 (PR:000011170)   | ROS (CHEBI:26523)    | pancreatic cancer (MESH:D010190) |
| HOC5 (CS5) | CXCL12 (PR:000006066) | senescence (HOC:42)  | thyroid cancer (MESH:D013964)    |


# Reproducing Our Random Experiments <a name="reproducing_random_experiments"></a>

To validate the efficacy of our method when ranking the true HOC relationships as high-ranked implicit relations for all datasets, we introduce and rank a set of random relationship triplets. The premise behind this approach is as follows: If the model identifies the true HOC relationships as implicit with a high rank (numerically low ranking value), introducing random relations into the method and ranking them should result in a low rank (numerically high ranking value); thus validating the efficacy of our method. Alternatively, if the system ranks these random triplet relationships with a high rank it invalidates our method as not suitable for closed LBD.

In the event we randomly generated a triplet which is a true relationship that exists in either the training or evaluation dataset, we generate 10 random triplets and report the averaged rank among all experiments. To explore this approach, we use two rules when generating the random triplets:

1. We ensure all concepts are unique within the relationship triplet.
2. We ensure no triplets are repeated.

 To reproduce our random experiments, use one of the following commands:

| Dataset |  Input Type |  Experiment ID |                                                     Command                                                                           |
|:-------:|:-----------:|:--------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|
|   HOC1  | Concatenate |        0       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs1_false_experiments/cs1_hinton_sigmoid_concat_ro_0.json`   |
|         | Concatenate |        1       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs1_false_experiments/cs1_hinton_sigmoid_concat_ro_1.json`   |
|         | Concatenate |        2       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs1_false_experiments/cs1_hinton_sigmoid_concat_ro_2.json`   |
|         | Concatenate |        3       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs1_false_experiments/cs1_hinton_sigmoid_concat_ro_3.json`   |
|         | Concatenate |        4       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs1_false_experiments/cs1_hinton_sigmoid_concat_ro_4.json`   |
|         | Concatenate |        5       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs1_false_experiments/cs1_hinton_sigmoid_concat_ro_5.json`   |
|         | Concatenate |        6       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs1_false_experiments/cs1_hinton_sigmoid_concat_ro_6.json`   |
|         | Concatenate |        7       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs1_false_experiments/cs1_hinton_sigmoid_concat_ro_7.json`   |
|         | Concatenate |        8       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs1_false_experiments/cs1_hinton_sigmoid_concat_ro_8.json`   |
|         | Concatenate |        9       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs1_false_experiments/cs1_hinton_sigmoid_concat_ro_9.json`   |
|   HOC2  | Concatenate |        0       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs2_false_experiments/cs2_hinton_sigmoid_concat_ro_0.json`   |
|         | Concatenate |        1       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs2_false_experiments/cs2_hinton_sigmoid_concat_ro_1.json`   |
|         | Concatenate |        2       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs2_false_experiments/cs2_hinton_sigmoid_concat_ro_2.json`   |
|         | Concatenate |        3       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs2_false_experiments/cs2_hinton_sigmoid_concat_ro_3.json`   |
|         | Concatenate |        4       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs2_false_experiments/cs2_hinton_sigmoid_concat_ro_4.json`   |
|         | Concatenate |        5       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs2_false_experiments/cs2_hinton_sigmoid_concat_ro_5.json`   |
|         | Concatenate |        6       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs2_false_experiments/cs2_hinton_sigmoid_concat_ro_6.json`   |
|         | Concatenate |        7       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs2_false_experiments/cs2_hinton_sigmoid_concat_ro_7.json`   |
|         | Concatenate |        8       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs2_false_experiments/cs2_hinton_sigmoid_concat_ro_8.json`   |
|         | Concatenate |        9       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs2_false_experiments/cs2_hinton_sigmoid_concat_ro_9.json`   |
|   HOC3  | Concatenate |        0       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs3_false_experiments/cs3_hinton_sigmoid_concat_ro_0.json`   |
|         | Concatenate |        1       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs3_false_experiments/cs3_hinton_sigmoid_concat_ro_1.json`   |
|         | Concatenate |        2       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs3_false_experiments/cs3_hinton_sigmoid_concat_ro_2.json`   |
|         | Concatenate |        3       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs3_false_experiments/cs3_hinton_sigmoid_concat_ro_3.json`   |
|         | Concatenate |        4       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs3_false_experiments/cs3_hinton_sigmoid_concat_ro_4.json`   |
|         | Concatenate |        5       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs3_false_experiments/cs3_hinton_sigmoid_concat_ro_5.json`   |
|         | Concatenate |        6       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs3_false_experiments/cs3_hinton_sigmoid_concat_ro_6.json`   |
|         | Concatenate |        7       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs3_false_experiments/cs3_hinton_sigmoid_concat_ro_7.json`   |
|         | Concatenate |        8       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs3_false_experiments/cs3_hinton_sigmoid_concat_ro_8.json`   |
|         | Concatenate |        9       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs3_false_experiments/cs3_hinton_sigmoid_concat_ro_9.json`   |
|   HOC4  | Concatenate |        0       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs4_false_experiments/cs4_hinton_sigmoid_concat_ro_0.json`   |
|         | Concatenate |        1       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs4_false_experiments/cs4_hinton_sigmoid_concat_ro_1.json`   |
|         | Concatenate |        2       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs4_false_experiments/cs4_hinton_sigmoid_concat_ro_2.json`   |
|         | Concatenate |        3       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs4_false_experiments/cs4_hinton_sigmoid_concat_ro_3.json`   |
|         | Concatenate |        4       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs4_false_experiments/cs4_hinton_sigmoid_concat_ro_4.json`   |
|         | Concatenate |        5       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs4_false_experiments/cs4_hinton_sigmoid_concat_ro_5.json`   |
|         | Concatenate |        6       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs4_false_experiments/cs4_hinton_sigmoid_concat_ro_6.json`   |
|         | Concatenate |        7       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs4_false_experiments/cs4_hinton_sigmoid_concat_ro_7.json`   |
|         | Concatenate |        8       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs4_false_experiments/cs4_hinton_sigmoid_concat_ro_8.json`   |
|         | Concatenate |        9       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs4_false_experiments/cs4_hinton_sigmoid_concat_ro_9.json`   |
|   HOC5  | Concatenate |        0       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs5_false_experiments/cs5_hinton_sigmoid_concat_ro_0.json`   |
|         | Concatenate |        1       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs5_false_experiments/cs5_hinton_sigmoid_concat_ro_1.json`   |
|         | Concatenate |        2       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs5_false_experiments/cs5_hinton_sigmoid_concat_ro_2.json`   |
|         | Concatenate |        3       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs5_false_experiments/cs5_hinton_sigmoid_concat_ro_3.json`   |
|         | Concatenate |        4       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs5_false_experiments/cs5_hinton_sigmoid_concat_ro_4.json`   |
|         | Concatenate |        5       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs5_false_experiments/cs5_hinton_sigmoid_concat_ro_5.json`   |
|         | Concatenate |        6       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs5_false_experiments/cs5_hinton_sigmoid_concat_ro_6.json`   |
|         | Concatenate |        7       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs5_false_experiments/cs5_hinton_sigmoid_concat_ro_7.json`   |
|         | Concatenate |        8       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs5_false_experiments/cs5_hinton_sigmoid_concat_ro_8.json`   |
|         | Concatenate |        9       | `python LBDDriver ../json_files/HOC/hinton_experiments/false_experiments/cs5_false_experiments/cs5_hinton_sigmoid_concat_ro_9.json`   |


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

The file `model_metrics.txt` is a TSV-formatted file which contains all model training and evaluation metrics.
