# Predicting Flight Mode

This repo contains the code for the prediction of flight modes such as automatic, manual, or guided flights. This work is a continuation on the drone mode research explained here: https://github.com/cdy3870/UAV_ML. In this research, different models are explored in addition to image-based approaches. 

## Requirements

To reproduce the results from this paper, you first need to set up a virtual environment with the correct dependencies. For simplicity, the conda environment used has been exported and stored as environment.yml. 
Run the following after cloning the repo:

```
conda env create -f environment.yml
conda activate drone_stuff
```

## Source code

1. data_preprocessing.py: contains functions that process the flight review data
2. flight_mode_preprocess.py: contains functions that performs more specific preprocessing on the aggregated data such as handling the chunking involved with the different modes
3. flight_mode_models.py: contains the different models that were explored, the model categories include nn-based, classical ml, and ensemble models
4. experiment_images/flight_mode_images_64.py: contains an exploration of CNN image classification of trajectory plots

## Reproducing results

### Data

### Experiments



