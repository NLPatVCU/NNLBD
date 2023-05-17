Frequently Asked Questions
==========================

We answer some frequently asked questions about running our system here. Hopefully we have the answer you're looking for.


## Table of Contents
1. [How can I test my NNLBD system environment?](#how_to_test_nnlbd_system)
2. [How can I run your system only using the CPU?](#how_to_run_on_cpu)
3. [How can I run your system on a GPU?](#how_to_run_on_gpu)
4. [How do I run experiments using NNLBD?](#how_to_run_experiments)
5. [How can I change the GPU the system uses to run experiments?](#how_to_change_desired_gpu)
6. [How can I save a model?](#how_to_load_model)
7. [How can I load a model?](#how_to_save_model)
8. [How can I refine an existing model?](#how_to_refine_model)
9. [What exactly does the model save?](#what_does_the_model_save)
10. [What's the basic structure of a configuration file?](#configuration_file_structure)
11. [Can you run multiple experiments in a single configuration file?](#multiple_experiment_configuration_file)
12. [How can I reproduce your previous work?](#how_to_reproduce_previous_work)
13. [What models are supported when using the 'closed_discovery_train_and_eval_x' or 'closed_discovery_refine_and_eval_x' tasks?](#supported_closed_discovery_train_refine_eval_tasks_models)
14. [Why does the CD-2 model have it's own task specification?](#cd2_model_task_specification)
15. [What's needed to generate the HOC datasets and embeddings with a Windows OS?](#a_priori_preprocessing_on_windows)
16. [How can I run multiple iterations of the same experiment?](#how_to_run_multiple_experiment_iterations)
17. [What embedding formats are supported?](#supported_embedding_formats)
18. [Can I convert between Word2vec binary and plain text embeddings](#convert_between_w2v_binary_and_plain_text)
19. [I see the global setting 'enable_gpu_polling'. What does this do?](#what_is_gpu_polling)
20. [When loading a model trained on a different GPU, my model runs on CPU. How can I fix this?](#gpu_issue_with_loaded_models)
21. [Why did you reduplicate an existing model if the authors released their code?](#why_reduplicate_cd2)
22. [Do you plan to add more models to the system?](#add_more_models)
23. [What models do you plan to add next?](#what_models_are_you_adding_next)
24. [Why did you code x-y-z like a-b-c and not use o-p-q instead?](#your_coding_sucks)


# How can I test my NNLBD system environment? <a name="how_to_test_nnlbd_system"></a>

After performing the installation steps on the main page, you can test your environment by running the following command:

```cmd
python LBDDriver.py ../json_files/tests/cui_mini_multi_task_test.json
```

This configuration file contains the following tasks:

1. Train an open discovery `Rumelhart` model on the `cui_mini` dataset using `concatenated` input representations and `vectors_random_cui_mini` embeddings.
2. Train a closed discovery `Hinton` model on the `cui_mini` dataset using `concatenated` input representations and `vectors_random_cui_mini` embeddings.
3. Train and evaluate a closed discovery `Hinton` model on the `cui_mini` dataset using `concatenated` input representations and `vectors_random_cui_mini` embeddings. This task will evaluate the `gold_b_instance = C001\tISA\tC002` after every training epoch. This task will also be saved to `../saved_models/cui_mini_hinton`.

    *NOTE: We provide a description of all saved files [here](#what_does_the_model_save).*


# How can I run your system only using the CPU? <a name="how_to_run_on_cpu"></a>

If your system does not have a CUDA-based GPU, or the appropriate CUDA version installed, it will use the CPU by default. Otherwise, if you explicitly wish to use the CPU you can do the following:

Include the `global_setting` in your configuration file: `"device_name": "/cpu:x",` and change the `x` to your desired CPU ID value (i.e. integer value).

Example: `"device_name": "/cpu:0",`

# How can I run your system on a GPU? <a name="how_to_run_on_gpu"></a>

Our system depends on `TensorFlow`, which in-turn depends on `CUDA`. So this question gets a little tricky and can be tedious. First, you must determine if you already have `CUDA` installed on your system. You can use the following command below:

```cmd
nvcc --version
```

You should see output similar to this.

```cmd
nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Feb_14_22:08:44_Pacific_Standard_Time_2021
Cuda compilation tools, release 11.2, V11.2.152            <- CUDA version is listed here
Build cuda_11.2.r11.2/compiler.29618528_0                  <-                ... and here
```

In this instance, we're running `CUDA 11.2`. Now comes the tricky part. If you do not have CUDA already installed, then you will need to determine which version is compatible with your installed version of Python 3 [here for Linux](https://www.tensorflow.org/install/source#gpu) or [here for Windows](https://www.tensorflow.org/install/source_windows#gpu).

- If you chose to use `Python 3.6.x`, then you can install `CUDA 10.0 + TensorFlow 1.15.x`, or `CUDA 11.2 + TensorFlow 2.4.0`.
- If you chose to use `Python 3.10.x`, then you should install `CUDA 11.2` and `TensorFlow 2.9.0`.
- etc.

Just take note that specific versions of `Python` and `TensorFlow` pairs have specific `CUDA` version requirements, and vice versa. Examine the provided compatiblity links carefully and determine which versions work best for you based on our listed requirements for the system.

- If you have CUDA already installed, then install the version of Python and TensorFlow your CUDA version is listed to be compatible with.
- If you do not have CUDA installed, then you can choose the versions that are within our tested TensorFlow (1.15.x-2.9.0) and Python (3.6.x-3.10.2) requirements. CUDA installation instructions are provided [here for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [here for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)... Good luck.

After the suffering has ended and CUDA is installed, verify it's installation using the `nvcc --version` command to double-check. Assuming you've received output similar to what we've shown above, you can now use our system with a GPU. Our system will scan for GPUs by default and select the first one if CUDA is available. No further configuration is necessary. However, if you wish to use a specific GPU, then include the `global_setting` in your configuration file: `"device_name": "/gpu:x",` and change the `x` to your desired GPU ID value (i.e. integer value).

Example: `"device_name": "/gpu:0",`

*NOTE: Alternative versions of Python and TensorFlow, other than what we've provided, may work with without issue. However, we make no guarantees as these remain untested.*


# How do I run experiments using NNLBD? <a name="how_to_run_experiments"></a>

After downloading the archived repository, extract it to any directory of your choosing. We'll refer to this as `./<root_dir>`. After creating your virtual environment and installing the Python required package, activate your virtual environment and navigate the the directory: `./<root_dir>/NNLBD`. This directory should follow structure as shown below:

```cmd
./<root_dir>/NNLBD/DataLoader/*
               .../Misc/*
               .../Models/*
               .../__init__.py
               .../LBD.py
               .../LBDDriver.py
```

To run experiments using NNLBD, you must provide a configuration file, and run via the command below:

```cmd
python LBDDriver.py <name_of_configuration_file>.json
```

We provide further configuration file details [here](#whats-the-basic-structure-of-a-configuration-file) and [here](./configuration_file.md).

*NOTE: This assumes you've installed the necessary requirements to use the system.*


# How can I change the GPU the system uses to run experiments? <a name="how_to_change_desired_gpu"></a>

By default, our system runs all processes on `/gpu:0` if not specified otherwise. To change this to a desired GPU device, add the following setting to `global_settings` within your configuration file: `"device_name": "/gpu:x",` and change the value of `x` to your desired device ID (i.e. any integer value > 0). We provide an example of this change within a configuration file [here](#how-can-i-refine-an-existing-model).


# How can I save a model? <a name="how_to_save_model"></a>

In your configuration file, add the setting `"model_save_path"` and provide a save directory. See the example below:

```json
"model_save_path": <model_save_path>,
```


# How can I load a model? <a name="how_to_load_model"></a>

In your configuration file, add the setting `"model_load_path"` and provide a directory. See the example below:

```json
"model_load_path": <model_load_path>,
```

*NOTE: The path must have an existing pre-trained model, or it will fail to load. Also, to load a model using the `LBDDriver.py` script, you will need to specific a task which load models. The current supported tasks which load models are as follows:*

```
eval_x
eval_prediction_x
eval_ranking_x
refine_x
crichton_closed_discovery_refine_and_eval_x
closed_discovery_refine_and_eval_x
```


# How can I refine an existing model? <a name="how_to_refine_model"></a>

In your configuration file, change the task to `refine_x` (with x being the task number) and add a `model_load_path` with the path to your existing model. This should be the directory containing the following files:

```
model_config.json                    <- Saved model Keras configuration file
model_settings.cfg                   <- NNLBD model configuration file
model_token_id_key_data              <- Input/Output Term Mappings
model.h5                             <- The saved model
<name_of_configuration_file>.json    <- Copy of your configuration file
...
```

An example configuration file is shown below:

```json
{
    "global_settings": [
        {
            "_comment": "Global Variable Settings",
            "device_name": "/gpu:0",
            "number_of_iterations": 1
        }
    ],
    "refine_1": [
        {
            "_comment": "HOC1 Hinton - Closed Discovery",
            "network_model": "hinton",
            "model_type": "closed_discovery",
            "embedding_path": "../vectors/HOC/test_modified_cs1.embeddings.bin",
            "train_data_path": "../data/HOC/train_cs1_closed_discovery_without_aggregators_mod",
            "eval_data_path": "../data/HOC/test_cs1_closed_discovery_without_aggregators_mod",
            "model_load_path": "../saved_models/cs1_hinton_model",
            "model_save_path": "../saved_models/cs1_hinton_model_refined",
            "epochs": 400,
            "verbose": 2,
            "embedding_modification": "average",
            "gold_b_instance": "PR:000001754\tPR:000002307\tMESH:D000236"
        }
    ]
}
```

All refinement tasks include:

```
refine_x
crichton_closed_discovery_refine_and_eval_x
closed_discovery_refine_and_eval_x
```


# What exactly does the model save? <a name="what_does_the_model_save"></a>

When saving a model, the system saves a couple file at minimum. These are shown below. However, depending on the `Experimental Task` chosen, more files can be added to this including graph images of model training and evaluation performance, a TSV file containing evaluation metrics (for easier importing into a spreadsheet), and a copy of your configuration file.

```
model_config.json                    <- Saved model Keras configuration file
model_settings.cfg                   <- NNLBD model configuration file
model_token_id_key_data              <- Input/Output Term Mappings
model.h5                             <- The saved model
<name_of_configuration_file>.json    <- Copy of your configuration file
...
```

*NOTE: We do not recommend editing any saved configuration files. If you have done so, this will lead to unexpected performance issues when re-loading into the system for refinement and inference.*


# What's the basic structure of a configuration file? <a name="configuration_file_structure"></a>

We provide the basic structure of the configuration file below.

```json
{
    "global_settings": [
        {
            "_comment": "Global Variable Settings",
        }
    ],
    "train_1": [
        {
            "_comment": "<Comment About Task>",
        }
    ]
}
```

The `global_settings` section and the task (i.e. `train_1`) sections have their respective variables that can be included. We provide a list of all available settings and descriptions [here](./configuration_file.md).

*NOTE: In this instance we've specified the `task` as `train_x` (with x = 1). But you can change this to any task of your choosing.*

# Can you run multiple experiments in a single configuration file? <a name="multiple_experiment_configuration_file"></a>

You sure can! The system is setup such that you can chain as many tasks as you like within the JSON configuration file. The `_x` element within the task specification is the *task number*. All you need to do is increase this value with each added task and provide the parameters of each task. As each task is finished, the next will be executed sequentially. We provide an example of a configuration file which runs three tasks below.

```json
{
    "global_settings": [
        {
            "_comment": "Global Variable Settings",
            "device_name": "/gpu:0",
            "number_of_iterations": 1
        }
    ],
    "train_1": [
        {
            "_comment": "HOC1 Hinton - Closed Discovery",
            "network_model": "hinton",
            "model_type": "closed_discovery",
            "embedding_path": "../vectors/HOC/test_modified_cs1.embeddings.bin",
            "train_data_path": "../data/HOC/train_cs1_closed_discovery_without_aggregators_mod",
            "eval_data_path": "../data/HOC/test_cs1_closed_discovery_without_aggregators_mod",
            "model_save_path": "../saved_models/cs1_hinton_model",
            "epochs": 50,
            "verbose": 2,
            "embedding_modification": "average"
        }
    ],
    "refine_2": [
        {
            "_comment": "HOC1 Hinton - Closed Discovery",
            "network_model": "hinton",
            "model_type": "closed_discovery",
            "embedding_path": "../vectors/HOC/test_modified_cs1.embeddings.bin",
            "train_data_path": "../data/HOC/train_cs1_closed_discovery_without_aggregators_mod",
            "eval_data_path": "../data/HOC/test_cs1_closed_discovery_without_aggregators_mod",
            "model_load_path": "../saved_models/cs1_hinton_model",
            "model_save_path": "../saved_models/cs1_hinton_model_refined",
            "epochs": 50,
            "verbose": 2,
            "embedding_modification": "average"
        }
    ],
    "closed_discovery_train_and_eval_3": [
        {
            "_comment": "HOC1 Hinton - Closed Discovery",
            "network_model": "rumelhart",
            "model_type": "closed_discovery",
            "embedding_path": "../vectors/HOC/test_modified_cs1.embeddings.bin",
            "train_data_path": "../data/HOC/train_cs1_closed_discovery_without_aggregators_mod",
            "eval_data_path": "../data/HOC/test_cs1_closed_discovery_without_aggregators_mod",
            "model_save_path": "../saved_models/cs1_rumelhart_model",
            "epochs": 200,
            "verbose": 2,
            "learning_rate": 0.001,
            "batch_size": 256,
            "dropout": 0.1,
            "run_eval_number_epoch": 1,
            "embedding_modification": "concatenate",
            "gold_b_instance": "PR:000001754\tPR:000002307\tMESH:D000236",
            "feature_scale_value": 10.0
        }
    ]
}
```

*Brief description:* This configuration file instructs the system to train a `Hinton` model, using `averaged` inputs, and saves it after 50 epochs. Next, the system performs clean-up (removing the old model from memory), then loads the same `Hinton` model and refines the pre-trained model for another 50 epochs before saving to the new directory. Finally, the system performs clean-up and then creates a new `rumelhart` closed discovery model. It trains this model for 200 epochs and performs closed discovery evaluation after every epoch using the `gold_b_instance` and the `eval_data_path` data. In this example, all model use the same `embeddings_path`, `train_data_path`, and `eval_data_path`. However, these can be changed to your specification.

Another example of the benefits of the system is, if you wish to test the difference between input embeddings for the same model over the same datasets and embeddings. You can create three training+evaluation tasks and just change the `embedding_modification` between all three runs (i.e. `average`, `concatenate`, and `hadamard`). The system will the execute all tasks without futher intervention or having to execute multiple instances of the system.


# How can I reproduce your previous work? <a name="how_to_reproduce_previous_work"></a>

We provide these details [here](./getting_started.md#reduplicating-our-published-work).


# What models are supported when using the `closed_discovery_train_and_eval_x` or `closed_discovery_refine_and_eval_x` tasks? <a name="supported_closed_discovery_train_refine_eval_tasks_models"></a>

We have only tested the `hinton` and `rumelhart` models. All other remain untested. Thus, we cannot guarantee an accurate measure of performance when using other models.


# Why does the CD-2 model have it's own task specification? <a name="cd2_model_task_specification"></a>

This model's output differs from all others (i.e. it's output space is single-class classification vs our multi-label models). This requires a specific evaluation approach in comparison to our general approach. To separate these approaches, we created distinct tasks for the `CD-2` model.


# What's needed to generate the HOC datasets and embeddings with a Windows OS? <a name="a_priori_preprocessing_on_windows"></a>

When reduplicating our previous works for our [Base Multi-Label Models](./base_ml_model/README.md) over the [Cancer landmark discovery datasets](https://lbd.lionproject.net/downloads), you will need to use the [NN for LBD](https://github.com/cambridgeltl/nn_for_LBD) system to generate the training datasets, evaluation datasets, and word embeddings. (See [here](./reduplicating_work/dla_for_closed_lbd.md) for further details). To get around using a Linux operating system to use the `NN for LBD` package, you will need to take a couple of extra steps.

## Requirements

 - [Python 2.7](python.org)
 - Perl 5.x (Not necessary if you do not use our Perl script at Step #5)
 - [Cancer landmark discovery datasets](https://lbd.lionproject.net/downloads)
 - [Neural networks for open and closed Literature-based Discovery](https://github.com/cambridgeltl/nn_for_LBD) (NN for LBD)

## Instructions

1. Install [Python 2.7](https://www.python.org/).

    You will run into issues if another version of Python is installed. You can temporarily rename the `python.exe` (version 2.7) executable to `python27.exe`. This will give you access to calling both versions in CMD or Powershell. This can be reversed prior to uninstalling the application.

2. Download the [NN for LBD](https://github.com/cambridgeltl/nn_for_LBD) package and unarchive it somewhere. We will refer to this directory as `./nn_for_LBD` for the remainer of the tutorial.

3. Create a Python 2.7 virtual environment, preferably within the `nn_for_LBD` directory, and activate it. Update the following module: `pip`, `setuptools`, and `wheel`. Then install the `NN for LBD` package requirements.

    ```cmd
    pip install -U pip setuptools wheel
    pip install -r requirements.txt
    ```

4. The `NN for LBD` package depends on BASH scripts to generate the data and embeddings. We recommend using a BASH-style simulator to accomplish this. We recommend [git-for-windows](https://gitforwindows.org/) through CMD or using [CMDER](https://github.com/cmderdev/cmder).

    *NOTE: If you choose CMDER, then download the version which is pre-packaged with git-for-windows.*

    For either application, you will have access to an extensive suite of Linux-style utilities under Windows.

5. Download the five `Cancer landmark discovery datasets` from [here](https://lbd.lionproject.net/downloads). Then extract the files within each archive to the `./nn_for_LBD/data/` directory. It will resemble the directory structure below:

    ```cmd
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

    ```cmd
    ./nn_for_LBD/data/PR000001138_PR000006736/edges_with_scores.csv
                  .../PR000001754_MESHD000236/edges_with_scores.csv
                  .../PR000006066_MESHD013964/edges_with_scores.csv
                  .../PR000011170_MESHD010190/edges_with_scores.csv
                  .../PR000011331_PR000005308/edges_with_scores.csv
    ```

6. Now we need to edit the `experiment_batch_cases.sh` file, within the `./nn_for_LBD` main directory, before we can begin generating the data.

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

    Next, we need to change the following lines:

    <pre><code>LINE/<mark>linux</mark>/line -train "test_adj_mat_${embeddingsshortname}.line" \
            -output "test_${embeddingsshortname}-order1.embeddings.bin" -size 50 -order 1 \
            -samples 1000 -threads 10 #Halve so the combined vector can have the desired dimension</code>
    <code>
    LINE/<mark>linux</mark>/line -train "test_adj_mat_${embeddingsshortname}.line" \
            -output "test_${embeddingsshortname}-order2.embeddings.bin" -size 50 -order 2 \
            -samples 1000 -threads 10 #Halve so the combined vector can have the desired dimension</code></pre>

    to

    <pre><code>LINE/<mark>windows</mark>/line -train "test_adj_mat_${embeddingsshortname}.line" \
              -output "test_${embeddingsshortname}-order1.embeddings.bin" -size 50 -order 1 \
              -samples 1000 -threads 10 #Halve so the combined vector can have the desired dimension</code>
    <code>
    LINE/<mark>windows</mark>/line -train "test_adj_mat_${embeddingsshortname}.line" \
              -output "test_${embeddingsshortname}-order2.embeddings.bin" -size 50 -order 2 \
              -samples 1000 -threads 10 #Halve so the combined vector can have the desired dimension</code></pre>

7. Let's go back to our CMD session at the `./nn_for_LBD/` directory. Activate your virtual environment and run the `experiment_batch_cases.sh` file via the command below.

    ```bash
    bash experiment_batch_cases.sh
    ```

    This will take a while, so take a break, catch-up an episode of your favorite show for the moment. But don't forget to come back. See you later.

    After the script has finished, you will be left with the following files:

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

8. Now we can use the [CD-2 model](./cd2_redup_model/README.md). If you wish to use these datasets with our [Multi-Label Models](./base_ml_model/README.md), we need to perform two more modifications on these datasets. The training and testing files contain negative samples which are not utilized for the `Multi-Label Models`. To remove these samples, along with other unnecessary information, we recommend using our [convert_hoc_data_to_nnlbd_format_v2.py](/miscellaneous_scripts/convert_hoc_data_to_nnlbd_format_v2.py) script. Edit the variables `file_path` and `new_file_path` to make these changes. If you wish to perform this manually, omit the `label` column within each dataset and any instances with label `0.0` (e.g. these are negative sample instances). Also remove the header line (i.e. first line): '`node1 node2 node3 label`'.

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

    Now we are ready to begin LBD experimentation using the Multi-Label Models. You may remove the `./nn_for_LBD` directory and any associated files. **Please, keep the aforementioned files before removing the main `./nn_for_LBD` directory.**


# How can I run multiple iterations of the same experiment? <a name="how_to_run_multiple_experiment_iterations"></a>

When the system reads the configuration file, it looks for a setting named: `number_of_iterations`. The default value is `1`. This determines how many times to run the tasks provided within the configuration file.

*WARNING: When using this function and saving models, you should set `"set_per_iteration_model_path": "True"` in your configuration file. This will append the iteration number onto the desired save directory. For example, if we use the following configuration file:*

```json
{
    "global_settings": [
        {
            "_comment": "Global Variable Settings",
            "device_name": "/gpu:0",
            "number_of_iterations": 5
        }
    ],
    "train_1": [
        {
            "_comment": "HOC1 Hinton - Closed Discovery",
            "network_model": "hinton",
            "model_type": "closed_discovery",
            "embedding_path": "../vectors/HOC/test_modified_cs1.embeddings.bin",
            "train_data_path": "../data/HOC/train_cs1_closed_discovery_without_aggregators_mod",
            "eval_data_path": "../data/HOC/test_cs1_closed_discovery_without_aggregators_mod",
            "model_save_path": "../saved_models/cs1_hinton_model",
            "epochs": 50,
            "verbose": 2,
            "embedding_modification": "average",
            "set_per_iteration_model_path": "True"
        }
    ]
}
```

This will save all 5 models to the following directories:

```cmd
../saved_models/cs1_hinton_model_1
../saved_models/cs1_hinton_model_2
../saved_models/cs1_hinton_model_3
../saved_models/cs1_hinton_model_4
../saved_models/cs1_hinton_model_5
```

Each will contain their respective saved files.

# What embedding formats are supported? <a name="supported_embedding_formats"></a>

Our system supports Word2vec-style plain text and binary embeddings. The system will automatically determine which embeddings you've specified when loading your experiments.

# Can I convert between Word2vec binary and plain text embeddings? <a name="convert_between_w2v_binary_and_plain_text"></a>

Yes, you can do this with the `DataLoader` class. See the example code below.

Binary-to-Plain Text Embeddings
```python
from NNLBD.DataLoader import DataLoader

binary_embedding_path = "./word2vec_binary_embeddings.bin"
dataloader            = DataLoader()
plain_text_embeddings = dataloader.Load_Embeddings( file_path = binary_embedding_path )

# Do With Them As You Wish
...
```

Plain Text-to-Binary Embeddings
```python
from NNLBD.DataLoader import DataLoader

embedding_path        = "./word2vec_text_embeddings.bin"    # Existing File
binary_embedding_path = "./word2vec_binary_embeddings.bin"  # New Converted File
dataloader            = DataLoader()
plain_text_embeddings = dataloader.Load_Embeddings( file_path = embedding_path )
if not dataloader.Save_Binary_Embeddings( embeddings = embeddings, save_file_path = binary_embedding_path ):
    print( "Error Converting Embeddings To Binary Format" )
```


# I see the global setting `enable_gpu_polling`. What does this do? <a name="what_is_gpu_polling"></a>

When executing the system to run tasks, it will poll all detected GPU devices to see how much memory is listed as available (or free). The first GPU with more than the specified `acceptable_available_memory` will be selected for running experiments. If no GPU has the desired amount of memory available, no task will be executed. The system will remain active and check every 10 seconds for 2 weeks until a GPU reports enough memory to start running tasks.

To use this function, include these two parameters within the `global settings` section of your configuration file:

```json
"enable_gpu_polling": "True",
"acceptable_available_memory": 4096
```

Also, tell the system how much `acceptable_available_memory` is desired to run your experiment.

*NOTE: If you have exclusive GPU access available to you, then this function is not useful to you.*


# When loading a model trained on a different GPU, my model runs on CPU. How can I fix this? <a name="gpu_issue_with_loaded_models"></a>

The system attempts to reduplicate the training environment of your model. This means it will attempt to use the same  GPU device you specified for model training. If you attempt to re-load the model on a different system from the training environment, you may encounter issues with the desired GPU device. e.g. If the environment the model was trained on contained multiple GPUs and the new environment contains less GPUs.

For example, if the system was trained using the seventh GPU (`/gpu:6`) in a multi-GPU server, but the new system only contains one GPU (`/gpu:0`). The system will attempt to load the model using the seventh GPU (`/gpu:6`) once again. This results in an error and it falls back to using the CPU (`/cpu:0`). To fix this, you will need to edit the model's saved configuration file.

We provide an example of all model saved files [here](#what_does_the_model_save).

Edit the setting `DeviceName<:>/desired_device:x` in the `model_settings.cfg` file, with your desired device. Your model should load on the GPU without issue.


# Why did you reduplicate an existing model if the authors released their code? <a name="why_reduplicate_cd2"></a>

We did this for many reasons.

1. The authors developed their package using Python 2.7 and we already developed our system using Python 3.6 (and above). Having to maintain two virtual environments and run experiments back and forth between the two system was tedious at best. We found it easier to reduplicate the model and make it backwards compatible with their data.
<!--2. Their instructions were not sufficient to deploy their system. Usage instructions were sparse and a lot of time was spent trying to figure out how the system works.
3. We noticed the authors did not normalize text when generating static embeddings. This led us to question the study and other potential noted issues.-->
2. We wanted to use their model as a baseline to test against future work. This is much easier as their model is integrated into the our system.


# Do you plan to add more models to the system? <a name="add_more_models"></a>

Your assumption is correct. We plan to add more as we develop more solutions (i.e. models) and tested their suitability for LBD. As time passes, the system will go through many changes.


# What models do you plan to add next? <a name="what_models_are_you_adding_next"></a>

This particular assignment is G-14 classified. We're not allowed to release further details at this point. :eyes:


# Why did you code x-y-z like a-b-c and not use o-p-q instead? <a name="your_coding_sucks"></a>

We make no claims of being astute software engineers. We're simply people who like to explore machine learning for NLP. Please do not try and hold us responsible if your CPU/GPU fans take a lunch break, and your computer decides it's had enough and files for divorce. Use at your own risk.


