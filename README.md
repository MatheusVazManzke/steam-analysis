I'm currently rewritting this so that it better reflects my current technical skills.
## Table of Contents

1. [Introduction](#Introduction)
2. [Scaffolding](#Scaffolding)
3. [Environments](#Environments)
4. [Pre-commit hooks](#pre-commit-hooks)
5. [Transformers and Pipelines](#Transformers-and-Pipelines)
6. [Statistical Tests](#Statistical-Tests)
7. [Hyperparameter tuning with Optuna](#Hyperparameter-tuning-with-Optuna)
8. [Model Calibration](#Model-Calibration)
9. [MlFlow model tracking](#MlFlow-model-tracking)
10. [Model Interpretation](#Model-Interpretation)
    
</small> Note: Overall, there are some unnecessary complexities for a project of this scale. I still hope that it serves as an example of my baseline workflow.</small>

==============================

# Introduction

This repository is based on a project I did for a gaming company. The goal is trying to find patterns in the way successful games are presented on the gaming store 'Steam', using both classical data analysis tools and also LLM methods. My main goal here is documenting my workflow, providing an easy reference for best practices and reusable code. I also hope it may be useful to the occasional lost wanderer who stumbles upon it.

# Scaffolding 

The first thing I needed to tackle was the lack of structure in my projects. I knew from the start I couldn't just have a giant jupyter notebook file. But from the infinite ways to scaffold a project, which one would be the most adequate? Was there a structure I could use generally? I found the answer with [cookiecutter](https://drivendata.github.io/cookiecutter-data-science/). You can find details about the file structure bellow. 


Surely there are other ways to scaffold a project. And individual tasks may require specific solutions. But that's the default I always fall back on. 




Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


# Environment 

A tight controlled environment allows for better maintainability and reproducibility of results. You and your colleagues should be able to easily reproduce the environment where some experiment was run or some model was trained in order to consistently find the same results. Also, the intricate network of dependencies that exists between packages is very prone to breaking. With environments, we can limit the number and version of packages to the strictly needed. A have a separate environment for each serious project. I use [Conda](https://docs.conda.io/en/latest/) for easy management of dependencies.
The [makefile](https://github.com/MatheusVazManzke/steam-analysis/blob/main/Makefile) in this repository allows us to recreate an environment anywhere with a sequence of simple commands. This file is automatically created by Cookiecutter and provides a clear and elegant structure. It’s well worth taking a look at it to understand how it works.

# Pre-commit hooks
One of my favorite discoveries is [Pre-commit](https://pre-commit.com/). With Pre-commit, we can ensure that all our commits adhere to a standard set by ourselves in a .yaml file. You can check the file for this repository [here](https://github.com/MatheusVazManzke/steam-analysis/blob/main/.pre-commit-config.yaml). For example, I am unable to commit any file unless it complies with the PEP8 style guide for Python. If it doesn't, committing will fail, the files will be formatted by calling [Black](https://github.com/psf/black), and I will have a chance to review the changes before trying to commit again. This applies to any instructions we define in our file. We can set up hooks to typecheck, lint, and even test our code.

# Transformers and Pipelines
Very early I faced the question of how to reliably pre-process and treat new data. The dataset in this repository has data up to this year. If in one year from now I want to redo all the analysis and retrain the models, how can I be sure that everything will be processed as intended and as fast as possible? I built all the data processing into transformers classes that are then chained inside a pipeline. You can check the transformer classes I've coded [here](https://github.com/MatheusVazManzke/steam-analysis/blob/main/src/features/transformer_classes.py). I always try to make them as dynamic as possible, so that little refactoring will be needed if there are small changes to how the data is structured. With a few keystrokes, we can ensure that the whole transformation process will be replicated even if there are different column names or if we want to change the way we build our target variable. I'm not sure that this is the most elegant way to deal with such simple replication steps, but I also hope to show familiarity with OOP concepts.

# Statistical Tests
It didn't take long for me to realize that when analyzing data, comparing the mean for two different distributions can amount to simply saying that a number is higher than the other and nothing more. What we are usually interested in - and its important to make it explicit - is to find out if there are significant differences between the underlying distribution for two different results. There is a myriad of statistical tests and we should be careful about their assumptions. I also started explicitly stating the null hypothesis in my notebooks to ensure that the kind of conclusions that can be made from such a test are more constrained. A last, important lesson is that finding statistically significant differences does not tell us about the size of the difference! For that, we have to consider a series of effect size tests. You can find some code examples in my [analysis file](https://github.com/MatheusVazManzke/steam-analysis/blob/main/notebooks/analysis/1.2-mvm-data-analysis.ipynb)

# Hyperparameter tuning with Optuna
Optuna is my framework of choiche for hyperparameter optimization. Optuna works by allowing us to define an objective function that then will be optimized through a process of Bayesian optimization. We have full control over the search space and what is being optimized for. In fact, Optuna doesn't have necessarily to do with machine learning hyperparemeter: It is more generally a [optimization framework](https://medium.com/@walter_sperat/using-optuna-the-wrong-way-e403f7c8e726). The objective function and the subsequent study in this repository can be found [here](https://github.com/MatheusVazManzke/steam-analysis/blob/main/notebooks/modeling/2.2-mvm-classification-models.ipynb).

# MlFlow model tracking
Since Data Science deals fundamentallynwith experimentation, we need to keep track of all the results in an organized fashion that can also be easily shared with our colleagues. I like doing this with MlFlow. It allow us to create Experiments and inside each experiment, to track multiple Runs, storing different metrics, parameters and artifacts. You will notice that I put MlFlow inside my Optuna objective function. This way, whenever I run a study, I'm able to keep all the results. MlFlow user interfarce then allow us to very easily compare results. 

# Model Calibration

When we train a classification model, to minimize our usual loss functions it is enough to attribute the highest probability to the right classes, but we don't need to attribute the _right_ probability to them. Take a look at the calibration curve in the [model notebook](https://github.com/MatheusVazManzke/steam-analysis/blob/main/notebooks/modeling/2.2-mvm-classification-models.ipynb) to see by how much our model inflates probabilities. In practice, this means we can't reliably measure the uncertainty of our model; If we group all predictions with a 0.95% probability, for example, we might end up right only 50% of the time. 

# Model interpretation
[General Pitfalls of Model-Agnostic Interpretation Methods for Machine Learning Models](https://arxiv.org/pdf/2007.04131) is a great reference for the interpretability of machine learning models. After reading it, I've adopted a simple routine for model interpretation: I use some sort of feature permutation importance to see what features are contributing the most to the model's _generalization_ and SHAP values to understand which features were most important for a specific trained model. This distinction is important: SHAP Values tell us how each feature altered a single prediction in relation to a baseline. Even if our model is really bad and/or overfitted, we will get SHAP values and they will be telling us nothing about real world relations! 
