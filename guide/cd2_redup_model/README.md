**C**losed **D**iscovery-2 Model Details
========================================

## Table of Contents
1. [Model Description](#model_description)
2. [Data Description](#data_description)
3. [Pre-processing](#pre-processing)
    1. [Term Vectorization](#term_vectorization)
4. [One-class Classification](#one_class_classification)
5. [Word Embeddings](#word_embeddings)
6. [Configuration File](#configuration_file)
7. [Regularization](#regularization)


# Model Description <a name="model_description"></a>

The following figure shows the architecture of the reduplicated closed discovery-2 (CD-2) multi-perceptron model as proposed by [Crichton, et al (2020)](https://doi.org/10.1371/journal.pone.0232891). We train the model to identify implicit relations for closed discovery. Given explicit *A-B-C* relationship term triplets, we input *A*, *C*, and *C* term embeddings into the model and train the model to predict a likelihood of the triplet forming a true relationship (i.e. link prediction).

![Alt text](./model.jpg)

**NOTE: This model is closed discovery only!**


# Data Description <a name="data_description"></a>

To train the system and perform inference, suitable data is needed. The system expects colloocated or co-occurrence term relationship data. An example is shown below. Each line within the table represents an *A-B-C* relation link and its associated Jaccard similarity coefficient score as the label.


|         A Term        |         B Term       |              C Term              | Label |
|:---------------------:|:--------------------:|:--------------------------------:|:-----:|
| NF-B (PR:000001754)   | Bcl-2 (PR:000002307) | Adenoma (MESH:D000236)           |   1   |
| NOTCH1 (PR:000011331) | senescence (HOC:42)  | C/EBP (PR:000005308)             |   1   |
| IL-17 (PR:000001138)  | p38 (PR:000003107)   | MKP-1 (PR:000006736)             |   1   |
| Nrf2 (PR:000011170)   | ROS (CHEBI:26523)    | pancreatic cancer (MESH:D010190) |   1   |
| CXCL12 (PR:000006066) | senescence (HOC:42)  | thyroid cancer (MESH:D013964)    |   1   |


Terms and labels within each line are separated by whitespace or tab characters as shown below:

```
entity_a entity_b entity_c label_1
entity_a entity_b entity_c label_2
...
entity_a entity_b entity_c label_3
```


# Pre-processing

The system parses data within the dataset and lowercases all text. Next, the system compiles all unique lists of entities for all A, B, and C terms. These lists are used to map terms to index values within the lists.

*(TODO):* To override this setting, set the variable `"lowercase" : False` in the JSON configuration file.

In order for the model to generalize relationship data between terms, we vectorize the model inputs and outputs. The following sub-sections discuss how we accomplish this.


## Term Vectorization <a name="term_vectorization"></a>

We train the neural network architecture to identify implicit relations between terms using the embedded semantics between explicit term relationships. To accomplish this, we map the term text representations to their real-valued vector representations (or embeddings), and feed these embeddings as input into the model. We use embedding layers and provide the one-hot encodings to select the appropriate embeddings which are forward propagated through the model. Given the unique list of term embeddings, the one-hot encodings represent the index of the term we wish to forward as input into the model.

For example, we have the following explicit *A-B-C* relationship:

`NF-B (PR:000001754)` $\xrightarrow{co-occurs\ with}$ `Bcl-2 (PR:000002307)` $\xrightarrow{co-occurs\ with}$ `Adenoma (MESH:D000236)` <=> Label: 1

The model uses all *A*, *B*, and *C* terms as input and predicts the label. The one-hot encoded representations of the input terms would be represented as follows:

```
A Term Embedding Index -> 0 1 2 3 4 5 6 7 8 ... n
NF-B (PR:000001754)    -> 0 1 0 0 0 0 0 0 0 ... 0

B Term Embedding Index -> 0 1 2 3 4 5 6 7 8 ... n
Bcl-2 (PR:000002307)   -> 1 0 0 0 0 0 0 0 0 ... 0

C Term Embedding Index -> 0 1 2 3 4 5 6 7 8 ... n
Adenoma (MESH:D000236) -> 0 0 0 0 0 0 1 0 0 ... 0
```

This selects the embeddings associated to the input terms.

NOTE: We generally perform synonym marginalization by employing the use of the United Medical Language System's (UMLS) concept unique identifiers (CUIs) as our term representations. This system accounts for lexical variations between terms and maps synonymous terms to their overarching concept within the UMLS hierarchy. This also reduces the input and output spaces by reducing the number of total unique terms across both spaces.


# One-class Classification <a name="one_class_classification"></a>

In representing the output of our model, we use the associated Jaccard similarity coefficient label score as the true label. We train the model to predict the label given the *A-B-C* link. This can be viewed as predicting links between input (i.e. concepts) within a knowledge graph. This is also known as *link prediction*. For each *A-B-C* relationship, the model learns to predict if the triplet forms a true relationship. To accomplish this, negative samples are necessary. This entails generating false *A-B-C* relations and assigning their label as 0. A 50-50% distribution of positive-to-negative ratio of instance is recommended.


# Word Embeddings <a name="word_embeddings"></a>

For each term representation, we use word embeddings to map term text to real-valued vector representations. These embeddings are pre-trained using various collocation or co-occurrence based algorithms such as Large-scale Network Embeddings (LINE), or Word2vec. These algorithms capture linguistic patterns obtained from co-occurring terms present within large text corpora given some constraint (e.g. windowing technique, etc), or among first or second-order vertex collocations within knowledge graphs. However, any embedding generation algorithm will work. These embeddings are provided by the `embedding_path` setting in the configuration file. We provide an example of embedding representations below.

*NOTE: We generate these embeddings using their Medical Subject Heading (MeSH) term identifers (i.e. the elements encapsulated in parantheses of the 'TERM' column) to marginalize term synonyms.*

| Index |         Term           |                         Embedding                        |
|:-----:|:----------------------:|:---------------------------------------------------------:|
|   0   | Bcl-2 (PR:000002307)   | 0.775657 0.357768 0.303839 0.860398 ... 0.678169 0.403577 |
|   1   | NF-B (PR:000001754)    | 0.745247 0.536593 0.690505 0.260826 ... 0.059225 0.584933 |
|   2   | NOTCH1 (PR:000011331)  | 0.296836 0.358458 0.653022 0.184876 ... 0.610497 0.975490 |
|   3   | senescence (HOC:42)    | 0.548290 0.283603 0.538493 0.087476 ... 0.405630 0.353436 |
|   4   | IL-17 (PR:000001138)   | 0.239888 0.297092 0.349567 0.055417 ... 0.100888 0.484676 |
|   5   | p38 (PR:000003107)     | 0.731098 0.332324 0.114800 0.648015 ... 0.385557 0.132474 |
|   6   | Adenoma (MESH:D000236) | 0.787830 0.642801 0.823766 0.846400 ... 0.197761 0.599930 |
|   7   | ROS (CHEBI:26523)      | 0.142695 0.207144 0.487873 0.613676 ... 0.757319 0.546313 |
|   8   | CXCL12 (PR:000006066)  | 0.518898 0.925204 0.728640 0.717582 ... 0.261286 0.858261 |
|  ...  |          ...           |                                     ...                   |
|   n   | Nrf2 (PR:000011170)    | 0.672570 0.067213 0.542269 0.897966 ... 0.787969 0.178193 |


# Configuration File <a name="configuration_file"></a>

To execute an experiment, we use JSON-formatted configuration files as an argument while executing the `LBDDriver.py` script. An example is shown below:

```cmd
python LBDDriver.py config.json
```

- Minimum configuration file requirements:

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
            }
        ]
    }
    ```
All remaining model parameters will use their default values.  We provide further configuration file details [here](./../configuration_file.md).


# Regularization

The authors of this model report its ability to successfully reduplicate their tested relationships among all datasets. Further details of their approach and results can be observed within their manuscript [here](https://doi.org/10.1371/journal.pone.0232891).

We also found this model successfully captures implicit relations for closed discovery when evaluating over the Hallmarks of Cancer datasets (i.e. cancer landmark discovery set) as decribed [here](https://lbd.lionproject.net/downloads). We provide the necessary steps to reduplicate the method using our system [here](./../reduplicating_work/cd2_reduplication.md).