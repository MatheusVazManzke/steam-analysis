UNDER CONSTRUCTION

## Table of Contents

1. [Introduction](#Introduction)
2. [Scaffolding](#Scaffolding)
3. [Environments](#Environments)
4. [Pre-commit hooks](#pre-commit-hooks)
5. [Transformers and Pipelines](#Transformers-and-Pipelines)
6. [Hyperparameter tuning with Optuna](#Hyperparameter-tuning-with-Optuna)
7. [API](#API)

</small> Note: If you want to see the workflow and the code, go straight to [Scaffolding](#Scaffolding). Overall, there are some unnecessary complexities for a project of this scale. I still hope that it serves as an example of my baseline workflow.</small>

==============================

# Introduction

My main goal here is documenting my workflow, so that I can have an easy source of reference. I do hope, though, that it may be useful to the occasional lost wanderer that may end up here. For in data science, we are all eternal learners.

"How can I actually use this 'model' of yours?" - asked a friend of mine. I had no idea. The mathematical core of Machine Learning is a thing of beauty and one can easily become enraptured by its promises. I thought I was some sort of meta-oracle, writting in arcane language and teaching machines how to tread the unknown. But 'If a tree falls in a forest and no one is around to hear it, does it make a sound?'; If my models exist only in the comforting cells of my notebooks, are they learning anything at all? Are they able to withstand reality? Or, like phosporus when exposed to the atmosphere, will they dissolve spectacularly? That question I was asked cast me out of the dreamy space of beautiful abstractions and heeded me to the answer. 

I decided to start by hunting the real datasets and the real demands myself. knocked on the doors of local businesses and offered my skills. It was not long before I got some data to work with. It was hell. A pile of directories and files, duplicated names and a towering monolith of an environment. Soon I reached a cognitive bottleneck. I could not keep track of things, I had to redo work; either because I could not easily understand what I had done earlier or because I had outright lost the files. This Babel was not touching heaven. It was built to fall.

I decided to make an intertemporal pact with myself. I assured my future self that I would adhere to a principled structure. I paid future ease of information handling with present learning curve. I had a clear goal in sight: I wanted all my projects to be easy to navigate, easy to read, and easy to reproduce. 

In the end, the process of learning was fun and I hope to share some gems here so that the next person can have an even shorter learning curve. 

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

A tight controlled environment allows for better maintainability and reproducibility of results. You and your colleagues should be able to easily reproduce the environment where some experiment was run or some model was trained in order to consistently find the same results. Also, the intricate network of dependencies that exists between packages is very prone to breaking. With environments, we can limit the number and version of packages to the strictly needed. A have a separate environment for each serious project. I use [Conda]{https://docs.conda.io/en/latest/} for easy management of dependencies.
The makefile - [makefile](/blob/main/Makefile) in this repository allows us to recreate an environment anywhere with a sequence of simple commands. This file is automatically created by Cookiecutter and provides a clear and elegant structure. It’s well worth taking a look at it to understand how it works.

# PyTest

# Pre-commit hooks
One of my favorite discoveries is [Pre-commit](https://pre-commit.com/). With Pre-commit, we can ensure that all our commits adhere to a standard set by ourselves in a .yaml file. You can check mine [here](/blob/main/.pre-commit-config.yaml). For example, I am unable to commit any file unless it complies with the PEP8 style guide for Python. If it doesn't, committing will fail, the files will be formatted by calling [Black](https://github.com/psf/black), and I will have a chance to review the changes before trying to commit again. This applies to any instructions we define in our file. We can set up hooks to typecheck, lint, and even test our code.

# Transformers and Pipelines
Very early I faced the question of how to reliably pre-process and treat new data. The dataset in this repository has data up to this year. If in one year from now I want to redo all the analysis and retrain the models, how can I be sure that everything will be processed as intended and as fast as possible? I built all the data processing into transformers classes that are then chained inside a pipeline. You can check the transformer classes I've coded [here](https://github.com/{username}/{repository}/blob/main/src/features/transformer_classes.py). I always try to make them as dynamic as possible, so that little refactoring will be needed if there are small changes to how the data is structured. With a few keystrokes, we can ensure that the whole transformation process will be replicated even if there are different column names or if we want to change the way we build our target variable. I'm not sure that this is the most elegant way to deal with such simple replication steps, but I also hope to show familiarity with OOP concepts. 

# Hyperparameter tuning with Optuna

This is less meta than all the other entries in this short article. Still, [Optuna]{https://optuna.org/} is worth sharing. It made my life easier. When tuning a model, my go to algorithm was scikit-learn's GridSearchCV. It worked well enough. --to be continued-----

# API

This take us back to the initial question? How can my model actually be used? 
