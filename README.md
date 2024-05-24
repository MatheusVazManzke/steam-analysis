UNDER CONSTRUCTION

steam-analysis
==============================

My main goal here is documenting my workflow, so that I can have an easy source of reference. I write mainly to myself, expecting that the naivety of this early writing flourish into the assured confidence of a master. I do hope, though, that it may be useful to the occasional lost wanderer that may end up here. For in data science, we are all eternal learners. 

"How can I actually use this 'model' of yours?" - asked a friend of mine. I had no idea. The mathematical core of Machine Learning is a thing of beauty and one can easily become enraptured by its promises. I thought I was some sort of meta-oracle, writting in arcane language and teaching machines how to tread the unknown. But 'If a tree falls in a forest and no one is around to hear it, does it make a sound?'; If my models exist only in the comforting cells of my notebooks, are they learning anything at all? Are they able to withstand reality? Or, like phosporus when exposed to the atmosphere, will they dissolve spectacularly? That question I was asked cast me out of the dreamy space of beautiful abstractions and heeded me to the answer. 

I decided to start by hunting the real datasets and the real demands myself. knocked on the doors of local businesses and offered my skills. It was not long before I got some data to work with. It was hell. A pile of directories and files, duplicated names and a towering monolith of an environment. Soon I reached a cognitive bottleneck. I could not keep track of things, I had to redo work; either because I could not easily understand what I had done earlier or because I had outright lost the files. This Babel was not touching heaven. It was built to fall and so it fell. 

I decided to make an intertemporal pact with myself. I assured my future self that I would adhere to a principled structure. I paid future ease of information handling with present learning curve. I had a clear goal in sight: I wanted all my projects to be easy to navigate, easy to read, and easy to reproduce. 

In the end, the process of learning was fun and I hope to share some gems here so that the next person can have an even shorter learning curve. 

# Scaffolding 

The first thing I needed to tackle was the lack of structure in my projects. I knew from the start I couldn't just have a giant jupyter notebook file. But from the infinite ways to scaffold a project, which one would be the most adequate? Was there a structure I could use generally? I found the answer with [cookiecutter intro]


Surely there are other ways to scaffold a project. And individual tasks may require specific solutions. But that's the default I always fall 




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


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


# Environment 

I used to have one environment with all the packages I could possibly need. That's inelegant, I know now, I knew then. For every project that gets past conception and early data exploration, I create a virtual environment with Conda. I only install the packages I will be using. There's an aesthetic quality to this, but it also allows for easy of reproducibility. Conda will sort out all dependencies and, the less clutter we have, lesser is the probability of conflict. Also, a tight controlled environment allows for better control over the package versions. Packages are updated all the time and, considering the intricate network of dependencies between them, it is not a rare thing for something to stop working whenever you update them. 


# Version Control

Git

# PyTest

# Pre-commit hooks; testing, linting, formatting, typechecking.

Now, that's very cool. 

# Interlude 1 - Transformers and Pipelines

# Interlude 2 - Hyperparameter tuning with Optuna

