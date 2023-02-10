This repo contains a the code for TrinitySRITrojAI submission to the image-classificiation-sep2022 round of the [TrojAI leaderboard](https://pages.nist.gov/trojai/). 

# Dependencies

First install the dependencies required by https://github.com/usnistgov/trojai-example.

Additional dependencies
```
pip install pandas
pip install scipy
pip install scikit-learn
```

# Usage

Clone code into `<root>/trojai-fuzzing-vision`

Download and extract training set into `<root>/trojai-datasets/round11-train-dataset`

`cd` into `<root>/trojai-fuzzing-vision` and run the following commands

First run feature extraction on the training set.

```
python trinity.py
```

This will produce feature files under `<root>/trojai-fuzzing-vision/data_r11_trinity_v0`.

Then run cross-validation hyperparameter search using the feature file and a pre-defined detector architecture

```
python crossval.py
```

This will produce a set of learned detector parameters at `<root>/trojai-fuzzing-vision/sessions/0000000/`. 

Finally copy the detector parameters into a `learned_parameters` folder and build the singularity container, update meta file jsons, and build the container.
```
cp -r ./sessions/0000000/ ./learned_parameters
python metafile_generator.py
./build.sh
```

The script will test inference functionalities and build a container at `image-classification-sep2022_sts_SRI_trinity_v0.simg`.


