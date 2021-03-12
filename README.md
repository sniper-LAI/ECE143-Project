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
 │   ├── ...                 # Add here if you need
 ├── save                    # Save visualization plot
 ├── demo.ipynb              # All the visualizations generated for presentation
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
* [demo.ipynb](demo.ipynb) It shows all the visualizations generated for our presentation.
* [visualization.py](code/visualization.py) It contains functions for data visualization. To generate data visualization
plots, please run
    ```bash
    cd code
    python visualization.py 
    ```