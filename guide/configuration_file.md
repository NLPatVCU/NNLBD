Configuration File Details
==========================

To execute an experiment, we use JSON-formatted configuration files as an argument while executing the `LBDDriver.py` script. An example is shown below:

```cmd
python LBDDriver.py config.json
```

We provide the basic structure of the configuration file below:

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

The configuration file contains pertinent details including global settings, the type of experiment you wish to run, the model architecture you wish to test, LBD discovery type (i.e. open vs closed), and various hyperparameters. First, we list all global settings. Next, we provide a list of the experiment tasks the system can perform for each model. Finally, we list all model settings below along with their data types, default values, and provide a brief description of each setting.


Global Settings
===============

|           Setting           | Data Type | Default Value |                             Description                               |
|:---------------------------:|:---------:|:-------------:|:---------------------------------------------------------------------:|
| device_name                 | String    | "/gpu:0       | Set desired model device (CPU/GPU) (i.e. /cpu:0, /gpu:0, /gpu:5, etc) |
| enable_gpu_polling          | Boolean   | False         | Enables GPU polling                                                   |
| number_of_iterations        | Integer   | 1             | Number of times to execute experiment task in JSON configuration file |
| acceptable_available_memory | Integer   | 4096          | Amount of VRAM necessary for GPU polling execution                    |

Experiment Tasks
================

|                    Task                     |                                                     Description                                                       |
|:-------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------:|
| train_x                                     | Trains a model                                                                                                        |
| eval_x                                      | Runs model Evaluate() function on a pre-trained model in memory or given 'model_load_path' `(Not complete)`           |
| eval_prediction_x                           | Runs model Evaluate_Prediction() function on a pre-trained model in memory or given 'model_load_path' `(Not complete)`|
| eval_ranking_x                              | Runs model Evaluate_Ranking() function on a pre-trained model in memory or given 'model_load_path' `(Not complete)`   |
| train_and_eval_x                            | Trains a model and runs model Evaluate() function afterwards                                                          |
| train_and_eval_prediction_x                 | Trains a model and runs model Evaluate_Prediction() function afterwards                                               |
| train_and_eval_ranking_x                    | Trains a model and runs model Evaluate_Ranking() function afterwards                                                  |
| refine_x                                    | Refines an existing/pre-trained model in memory or given 'model_load_path'                                            |
| crichton_closed_discovery_train_and_eval_x  | Runs Crichton CD-2 reduplication model and ranks gold A-B-C instance within evaluation data                           |
| crichton_closed_discovery_refine_and_eval_x | Refines existing Crichton CD-2 reduplication model in memory or given 'model_load_path' and ranks gold A-B-C instance within evaluation data   |
| closed_discovery_train_and_eval_x           | Runs closed discovery training of 'network_model' and performs evaluation given gold A-B-C relation (i.e. 'gold_b_instance')  |
| closed_discovery_refine_and_eval_x          | Refies an existing/pre-trained model in memory or given 'model_load_path', runs closed discovery training and performs evaluation given gold A-B-C relation (i.e. 'gold_b_instance')  |

These tasks can be chained within a single JSON file. All tasks will run sequentially. We provide an example below.

*NOTE: The 'x' within each task description represents a integer value. See the complete JSON example below for further details.*


Model Settings
================

|            Setting            | Data Type |     Default Value     |                            Description                              |
|:-----------------------------:|:---------:|:---------------------:|:-------------------------------------------------------------------:|
| print_debug_log               | Boolean   | False                 | Prints debug log to the terminal/console                            |
| write_log_to_file             | Boolean   | False                 | Writes debug log to a text file                                     |
| per_epoch_saving              | Boolean   | False                 | Saves model after every epoch                                       |
| use_gpu                       | Boolean   | True                  | Enables GPU usage                                                   |
| skip_out_of_vocabulary_words  | Boolean   | True                  | Skips OOV word/terms, otherwise throws errors                       |
| use_csr_format                | Boolean   | True                  | Use Compressed Sparse Row/Matrix Format                             |
| trainable_weights             | Boolean   | False                 | Train encoding/embedding layer weights                              |
| shuffle                       | Boolean   | True                  | Shuffles data during model training                                 |
| enable_early_stopping         | Boolean   | False                 | Use early stopping                                                  |
| use_batch_normalization       | Boolean   | False                 | Use batch normalization                                             |
| set_per_iteration_model_path  | Boolean   | False                 | Attaches current iteration to model save path (Prevents path overwriting between epochs) |
| restrict_output               | Boolean   | False                 | Restricts output layer to B-terms for closed discovery and C-terms for open discovery    |
| save_best_model               | Boolean   | False                 | Saves the best model during training. (Based on reported training metrics)               |
| use_cosine_annealing          | Boolean   | False                 | Use cosine annealing with ADAM optimizer                            |
| network_model                 | String    | "rumelhart"           | Set desired deep learning architecture/model (i.e. hinton, rumelhart, cd2, mlp)          |
| model_type                    | String    | "open_discovery       | Set open or closed discovery (i.e. open_discovery, closed_discovery)                     |
| activation_function           | String    | "sigmoid"             | Set activation function of output layer (i.e. sigmoid & softplus)   |
| loss_function                 | String    | "binary_crossentropy" | Set loss function (i.e. binary_crossentropy, categorical_crossentropy, etc.)             |
| embedding_path                | String    | ""                    | Specify embedding path if used                                      |
| train_data_path               | String    | ""                    | Set training data path                                              |
| eval_data_path                | String    | ""                    | Set evaluation data path                                            |
| model_save_path               | String    | ""                    | Set model save path                                                 |
| model_load_path               | String    | ""                    | Set model load path                                                 |
| checkpoint_directory          | String    | ""                    | Set model checkpoint directory                                      |
| epochs                        | Integer   | 30                    | Set number of training epochs                                       |
| verbose                       | Integer   | 1                     | Set Keras/TensorFlow verbosity                                      |
| learning_rate                 | Float     | 5e-3                  | Set learning rate                                                   |
| learning_rate_decay           | Float     | 4e-3                  | Set learning rate decay value                                       |
| feature_scale_value           | Float     | 10                    | Set feature scaling value                                           |
| batch_size                    | Integer   | 32                    | Set batch size                                                      |
| optimizer                     | String    | "adam"                | Set training optimizer (SGD/ADAM)                                   |
| device_name                   | String    | "/gpu:0"              | Set desired model device (CPU/GPU) (i.e. /cpu:0, /gpu:0, /gpu:5, etc) NOTE: Overrides global setting |
| final_layer_type              | String    | "dense"               | Set final layer type (i.e. dense, cosface, arcface, sphereface)     |
| dropout                       | Float     | 0.1                   | Set dropout value                                                   |
| momentum                      | Float     | 0.05                  | Set momentum value (if using SGD)                                   |
| early_stopping_metric_monitor | String    | "loss"                | Set early stopping monitorring metric (i.e loss, precision, recall, f1_score)            |
| early_stopping_persistence    | Integer   | 3                     | Set early stopping persistence                                      |
| cosine_annealing_min          | Float     | 1e-6                  | Set cosine annealing minimum value                                  |
| cosine_annealing_max          | Float     | 2e-4                  | Set cosine annealing maximum value                                  |
| prediction_threshold          | Float     | 0.5                   | Set prediction threshold                                            |
| margin                        | Float     | 30.0                  | Set Cosface, Sphereface & ArcFace architecture margin value         |
| scale                         | Float     | 0.35                  | Set Cosface, Sphereface & ArcFace architecture scale value          |
| embedding_modification        | String    | "concatenate"         | Set embedding modification (i.e. average, concatenate, or hadamard) |
| run_eval_number_epoch         | Integer   | 1                     | Set number of epochs to train before evaluation is performed        |
| gold_b_instance               | String    | None                  | Set gold B-term instance (Used for CD evaluation)                   |


Example of a Configuration File
===============================

Here's a complete example of a configuration file.

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
            "_comment": "Rumelhart Model - CUI Mini Test Data-set For Closed Discovery",
            "print_debug_log": "False",
            "write_log_to_file": "False",
            "network_model": "rumelhart",
            "model_type": "closed_discovery",
            "per_epoch_saving": "False",
            "use_gpu": "True",
            "skip_out_of_vocabulary_words": "True",
            "activation_function": "sigmoid",
            "loss_function": "binary_crossentropy",
            "use_csr_format": "True",
            "trainable_weights": "False",
            "embedding_path": "../vectors/test/vectors_random_cui_mini",
            "train_data_path": "../data/test/cui_mini_closed_discovery",
            "model_save_path": "../saved_models/rumelhart",
            "set_per_iteration_model_path": "True",
            "epochs": 100,
            "verbose": 1,
            "learning_rate": 0.001,
            "dropout": 0.1,
            "batch_size": 32,
            "restrict_output": "True",
            "gold_b_instance": "C003\tTREATS\tC002",
            "feature_scale_value": 10.0
        }
    ],
    "closed_discovery_train_and_eval_2": [
        {
            "_comment": "Hinton Model - CUI Mini Test Data-set For Closed Discovery",
            "print_debug_log": "False",
            "write_log_to_file": "False",
            "network_model": "hinton",
            "model_type": "closed_discovery",
            "per_epoch_saving": "False",
            "use_gpu": "True",
            "skip_out_of_vocabulary_words": "True",
            "activation_function": "sigmoid",
            "loss_function": "binary_crossentropy",
            "use_csr_format": "True",
            "trainable_weights": "False",
            "embedding_path": "../vectors/test/vectors_random_cui_mini",
            "train_data_path": "../data/test/cui_mini_closed_discovery",
            "model_save_path": "../saved_models/hinton",
            "set_per_iteration_model_path": "True",
            "epochs": 100,
            "verbose": 1,
            "learning_rate": 0.001,
            "dropout": 0.1,
            "batch_size": 32,
            "restrict_output": "True",
            "gold_b_instance": "C003\tTREATS\tC002",
            "feature_scale_value": 10.0
        }
    ]
}
```

In this example, the system is instructed to perform training and evaluation of two models sequentially on `/gpu:0`. First, a `Rumelhart` model on the `cui_mini` dataset and its embeddings: `vectors_random_cui_mini`. Next, it trains a `Hinton` model using the same parameters as the previous model. Since the `number_of_iterations` is set to `5` in `global_settings`, all tasks within the configuration file are ran 5 times. (i.e. This runs 5 Rumelhart and 5 Hinton models within one call of the LBDDriver.py script while providing this configuration file).

The result of this is produces the following saved model directories:

```cmd
../saved_models/hinton_1
../saved_models/hinton_2
../saved_models/hinton_3
../saved_models/hinton_4
../saved_models/hinton_5
../saved_models/rumelhart_1
../saved_models/rumelhart_2
../saved_models/rumelhart_3
../saved_models/rumelhart_4
../saved_models/rumelhart_5
```

Each directory contains the following files:

```
model.h5                             <- The saved model
model_config.json                    <- Saved model Keras configuration file
model_metrics.txt                    <- TSV list of model reported evaluation metrics.
model_settings.cfg                   <- NNLBD model configuration file (Do Not Edit!)
model_token_id_key_data              <- Input/Output Term Mappings
<name_of_configuration_file>.json    <- Copy of your configuration file
...
```

*NOTE: The list saved files will be subject to change depending on the system-instructed task performed and if model saving is enabled.*

***NOTE: All tasks are not yet complete, so the system may not perform as expected for items listed as `(Not complete)`.***