**N**eural **N**etwork Architectures For Open and Closed **L**iterature-**B**ased **D**iscovery
===============================================================================================

In Natural Language Processing, Literature-based Discovery (LBD) is a form of knowledge extraction which aims to identify implicit relations by leveraging existing knowledge. Simply stated, LBD accomplishes the following task:

* What:

    * Connects two pieces of knowledge previously thought unrelated.

* How:

    * Identify implicit relationships between entities within disjoint texts.

Since it's inception in the 1980's by Dr. Don R. Swanson, many statistical methods to unearthing undiscovered relationships have been identified. These relationships include treatments for Parkinson’s Disease and multiple sclerosis, to understanding potential treatments for cancer, discovering new health benefits of curcumin, and treatments for migraine headaches, and identifying metabolites related to post-cardiac arrest. With the advent of deep learning, neural networks have achieved state of the art performance in various computer vision and NLP tasks. Our system Neural Network Architectures for Literature-based Discovery (NNLBD), integrates advances in deep learning for LBD.


Installation
============

NNLBD was developed and tested in Python version 3.6.x to 3.10.6. It also relies on the TensorFlow API. Versions 1.15.2 to 2.9.0 are supported. Tested operating systems to run our package include: Microsoft Windows 10 (64-bit) and Linux Mint (64-bit). The Microsoft Windows environment is not pre-package with Python. We recommend installing the appropriate Python version from [python.org](https://www.python.org/).

*NOTE: Further mentions to Python refer to the Python3 installation environment.*

Prior to instaling NNLBD, we recommend creating virtual environment.

- Depending on how Python is installed in your system, one of the following commands will be appropriate:

    ```cmd
    Linux:
            python -m venv <name_of_virtualenv>
            python3 -m venv <name_of_virtualenv>
    Windows:
            python -m venv <name_of_virtualenv>
    ```
- To verify which version of Python is installed, you can check via:

    ```cmd
    python --version
    python3 --version
    ```

Next, we activate your virtual environment and update your `pip`, `setuptools`, and `wheel` packages.

```cmd
Linux:
       source <name_of_virtualenv>/bin/activate
       pip install -U pip setuptools wheel

Windows:
       "./<name_of_virtualenv>/Scripts/activate.bat"
       pip install -U pip setuptools wheel
```

*NOTE: Powershell users will need to use the `activate.ps1` script with suitable permissions or call `cmd` within powershell to execute the  `activate.bat` script.*


Python Requirements
===================

After the setup of your virtual environment is complete, install the necessary NNLBD package requirements.

- Python 3.10.x and TensorFlow 2.9.0 - *(Recommended)*
    ```cmd
    pip install -r requirements_mini_py3.10_tf2.9.0.txt
    ```
- Python 3.6.x and TensorFlow 2.4.0
    ```cmd
    pip install -r requirements_mini_py3.6_tf2.4.0.txt
    ```
- Python 3.6.x and TensorFlow 1.15.2
    ```cmd
    pip install -r requirements_mini_py3.6_tf1.15.2.txt
    ```

To manually install the required packages, execute the following commands:

- Python v3.10.2 and TensorFlow v2.9.0 - *(Recommended)*
    ```cmd
    pip install -U h5py==3.7.0 Keras==2.9.0 matplotlib==3.5.2 numpy==1.22.4 scipy==1.9.0 sparse==0.13.0 tensorflow==2.9.0
    ```

- Python v3.6 and TensorFlow v2.4.0
    ```cmd
    pip install -U h5py==2.10.0 Keras==2.4.3 matplotlib==3.3.4 numpy==1.19.5 scipy==1.5.4 tensorflow==2.4.0
    ```

- Python v3.6 and TensorFlow v1.15.2
    ```cmd
    pip install -U h5py==2.10.0 Keras==2.3.1 matplotlib==3.3.3 numpy==1.19.5 scipy==1.5.4 tensorflow==1.15.2 tensorflow-gpu==1.15.2
    ```


Getting Started, System Description and Model Details
=====================================================

To test your NNLBD environment, we recommend running one of our test scripts. We provide more information [here](./guide/faq.md#how_to_test_nnlbd_system).

To run one of our models, we provide details for getting started [here](./guide/getting_started.md). This also includes a description of our system, integrated models and a guide to replicate previous works.

We also provide an FAQ [here](./guide/faq.md).


Reference
=========
```bibtex
@article{CUFFY2023104362,
   title = {Exploring a deep learning neural architecture for closed Literature-based discovery},
   journal = {Journal of Biomedical Informatics},
   volume = {143},
   pages = {104362},
   year = {2023},
   issn = {1532-0464},
   doi = {https://doi.org/10.1016/j.jbi.2023.104362},
   url = {https://www.sciencedirect.com/science/article/pii/S1532046423000837},
   author = {Clint Cuffy and Bridget T. McInnes},
   keywords = {Natural language processing, Literature-based discovery, Literature-related discovery, Neural networks, Deep learning, Knowledge discovery},
   abstract = {Scientific literature presents a wealth of information yet to be explored. As the number of researchers increase with each passing year and publications are released, this contributes to an era where specialized fields of research are becoming more prevalent. As this trend continues, this further propagates the separation of interdisciplinary publications and makes keeping up to date with literature a laborious task. Literature-based discovery (LBD) aims to mitigate these concerns by promoting information sharing among non-interacting literature while extracting potentially meaningful information. Furthermore, recent advances in neural network architectures and data representation techniques have fueled their respective research communities in achieving state-of-the-art performance in many downstream tasks. However, studies of neural network-based methods for LBD remain to be explored. We introduce and explore a deep learning neural network-based approach for LBD. Additionally, we investigate various approaches to represent terms as concepts and analyze the affect of feature scaling representations into our model. We compare the evaluation performance of our method on five hallmarks of cancer datasets utilized for closed discovery. Our results show the chosen representation as input into our model affects evaluation performance. We found feature scaling our input representations increases evaluation performance and decreases the necessary number of epochs needed to achieve model generalization. We also explore two approaches to represent model output. We found reducing the model’s output to capturing a subset of concepts improved evaluation performance at the cost of model generalizability. We also compare the efficacy of our method on the five hallmarks of cancer datasets to a set of randomly chosen relations between concepts. We found these experiments confirm our method’s suitability for LBD.}
}
```


License
=======
This package is licensed under the GNU General Public License.


Authors
=======
Current contributors: Clint Cuffy


Acknowledgments
===============
- [VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/) ![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")
