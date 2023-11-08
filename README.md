This repo contains a the code for weight analysis method for TrinitySRITrojAI submissions to the object-detection-feb2023 round of the [TrojAI leaderboard](https://pages.nist.gov/trojai/). 

# Dependencies

First install the dependencies required by https://github.com/usnistgov/trojai-example/tree/object-detection-feb2023.

Additional dependencies
```
pip install pandas
pip install scipy
pip install scikit-learn
pip install hyperopt
```

# Usage

Clone code into `<root>/trojai-fuzzing-vision`

Download and extract training set into `<root>/trojai-datasets/object-detection-feb2023`

`cd` into `<root>/trojai-fuzzing-vision` and run the following commands

First run feature extraction on the training set.

```
python trinity.py
```

This will produce feature files under `<root>/trojai-fuzzing-vision/fvs_weight`.

Then run cross-validation hyperparameter search using the feature file and a pre-defined detector architecture

```
python crossval.py --data fvs_weight --arch arch.mlp_set
```

This will produce an ensemble of learned detector parameters at `<root>/trojai-cyber-pdf/sessions/0000000/model.pt`. The best performing model will be kept and updated throughout hyperparameter search. Hyperparameter search can take hours to days. We typicall end at 200 runs and not wait all the way till the end. 

Finally test run detector on a new object detector model.
```
python run.py --detector sessions/0000000/model.pt --model ../trojai-datasets/object-detection-feb2023/models/id-00000001/model.pt 
```


# Hacking

Write your own `helper_r13_v0.py` to load your models. This includes a `def root()` function that returns where training data is located, and a `class engine` that loads the model weights.

`trinity.py` has utilities for feature extraction and running a detector ensemble.

`crossval.py` has utilities for crossval hyperparameter search.

`arch/mlp_set.py` is a simple Deep Sets architecture that converts a set of weight histograms across multiple layers to a Trojan prediction.