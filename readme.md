# Winning Percentage Predictions
This repository builds machine learning models to predict a MLB team's winning 
percentage at the end of the season. It's able to produce predictions at any point
in the season. The model uses two predictor variables for every team: 
the current winning percentage and the percentage of games played in the season. The
target is the team's season-end winning percentage. Every row represents a combination
of year, team, and game. 

To start, clone the repo:

```console
$ git clone git@github.com:micahmelling/winning-pct-predictions.git
```

You can then run generate_environment.sh to set up your environment. This will 
install packages in a virtual environment and set up pre-commit.

```console
$ chmod 755 generate_environment.sh
$ ./generate_environment.sh
```

Now, you can start executing scripts.

## Data
First, you need to generate the raw data. The below script uses the pybaseball
library to pull the data described above.

```console
$ python3 data/generate_raw_data.py
```

After you have the raw data, you can create a pickled file that will be suitable for 
modeling.

```console
$ python3 data/generate_modeling_data.py
```

You can toggle settings for pulling data with data/config.py.

## Modeling
After you have your modeling data, you can kick off training models.

```console
$ python3 modeling/train.py
```

The training script makes use of the following modules in the modeling directory:
- config: various config values for training the model; if you want to change behavior,
start here
- evaluate: functions to evaluate models
- explain: functions to explain models
- model: functions to optimize models
- pipeline: function(s) to generate a full modeling pipeline

To note, a custom time-series cross validation is implemented to prevent feature 
leakage.

## Utilities
The utilities directory houses three scripts that can be run for post-hoc model
analysis.
- model_analysis.py: analyzes errors and predictions
- produce_shap_waterfall_plots: makes SHAP waterfall plots
- shap_analysis: produces different SHAP explanation plots

## Helpers
The helpers directory has various functions to aid in modeling and analysis.
