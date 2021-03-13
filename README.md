# ECE143- Group 22 Project

## Dataset
New York City Airbnb Open Data: https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data

## File Structure
 ```
 .
 ├── dataset                 # Dataset
 │   ├── AB_NYC_2019.csv     # Data cvs file
 │   ├── ... 
 ├── code                    # Source files
 │   ├── visualization.py    # Function for dataset visualization
 │   ├── prediction.py       # Code for Prediction
 ├── save                    # Save visualization plot
 ├── demo.ipynb              # All the visualizations generated for presentation
 ├── price_prediction.ipynb  # This curates all the results of our prediction and analysis.
 ├── requirement.txt         # Packages requirement
 ├── .gitignore
 └── README.md
 
 ```

## Requirements
All packages are listed in `requirements.txt`.

## Installation
1. Clone this repository
2. Create a conda virtual environment with required packages
    ```bash
    conda create --name <env> --file requirements.txt
    ```

## Getting Started
* [demo_1.ipynb](demo_1.ipynb) It shows all the visualizations generated for our presentation.
* [visualization.py](code/visualization.py) It contains functions for data visualization. To generate data visualization
plots, please run
    ```bash
    cd code
    python visualization.py 
    ```
* [prediction.py](code/prediction.py) It contains functions for data prediction. To generate predictions,
please run
    ```bash
    cd code
    python prediction.py 
    ```
