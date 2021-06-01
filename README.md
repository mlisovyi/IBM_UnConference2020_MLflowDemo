# IBM_UnConference2020_MLflowDemo

- [IBM_UnConference2020_MLflowDemo](#ibm_unconference2020_mlflowdemo)
  - [Installation](#installation)
  - [Demo](#demo)

A demo of ML experimentation cycle tracked by mlflow.

## Installation

Clone the git repository locally:

```bash
git clone git@github.com:mlisovyi/IBM_UnConference2020_MLflowDemo.git
```

Install a conda environment with all relevant packages and activate it:

```bash
conda env create -f conda.yaml
conda activate mlflow_demo
```

## Demo

There are two main branches used in the demo:

* __exp_vanilla__ demonstrates how to train a basic ML model;
* __exp_mlflow__ demonstrates how to add model tracking to the model training.
