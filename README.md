# Multiplayer Dueling Bandit Algorithms

This code contains the implementation of the multiplayer dueling bandit algorithms mentioned in the manuscript. The code is organized and well-documented to facilitate understanding and usage.

## Contents

- **Basic_functions.py**: Contains basic utility functions used across different algorithms.
- **FYLR_algorithms.py**: Implementation of Follow Your Leader algorithms with experiments from the paper.
- **MPRUCB.py**: Contains the implementation of the Message Passing RUCB algorithm.
- **RMED.py**: Contains the implementation of the single-player RMED algorithm.
- **RUCB.py**: Contains the implementation of the single-player RUCB algorithm.
- **Versatile_DB.py**: Contains the implementation of the single-player VDB algorithm.
- **main.py**: The main script to run the experiments with command-line arguments.
- **default_parameters.json**: Default parameters configuration file.
- **override_parameters.json**: Override parameters configuration file.
- **plot_parameters.json**: Parameters configuration file for plotting results.
- **Irish_data.txt**, **Sushi_data.txt**: Example datasets used in experiments.
- **irish.pkl**, **sushi.pkl**, **six_arms.pkl**, **arithmetic.pkl**: Pickle files containing processed preference matrix for experiments.

## Adjustable Parameters
The following parameters can be configured when running code.

| Parameter             | Type              | Description                                                                             | Default Value                     |
|-----------------------|-------------------|-----------------------------------------------------------------------------------------|-----------------------------------|
| `num_of_runs`         | `int`             | Number of runs for each experiment.                                                     | `10`                              |
| `num_of_workers`      | `int`             | Number of parallel CPU cores to use.                                                    | `10`                              |
| `plot_flag`           | `bool`            | True to show the figure at the end (GUI required).                                      | `true`                            |
| `save_flag`           | `bool`            | True to save the parameters, regret and plot in a sub folder.                           | `false`                           |
| `new_folder_name`     | `str`             | Name of the folder to save results.                                                     | `"saved_results"`                 |
| `function_to_run`     | `list` of `str`   | Algorithms to run from `MP_RUCB`, `FYLRUCB`, `FYLRMED`.                                 | `["MP_RUCB","FYLRUCB","FYLRMED"]` |
| `dataset`             | `str`             | Dataset to use from `Sushi`, `Irish`, `Six_rankers`.                                    | `"Sushi"`                         |
| `T`                   | `int`             | Number of rounds for each experiment.                                                   | `30000`                           |
| `graph`               | `list` of `str`   | Communication graph from `complete`, `cycle`, `star`, `path`.                           | `["complete"]`                    |
| `M`                   | `list` of `int`   | Number of players.                                                                      | `[10]`                            |
| `gamma`               | `list` of `int`   | Decay parameter: -1 to use the diameter.                                                | `[-1]`                            |
| `lr_coeff`            | `list` of `int`   | VDB learning rate coefficient, where the learning rate at round t is `lr_coeff`/sqrt(t) | `[2]`                             |
| `alpha`               | `list` of `float` | Exploration parameter for RUCB and RMED.                                                | `[3]`                             |
| `RMED_flag`           | `list` of `bool`  | True to use RMED2FH, False for RMED1.                                                   | `[true]`                          |
| `rec_flag`            | `list` of `bool`  | True to use CW recommendations for MPRUCB, Fasle otherwise.                             | `[true]`                          |
| `average_regret_flag` | `list` of `bool`  | True to display the average regret over players instead of the sum.                     | `[false]`                         |
| `single_player_flag`  | `list` of `bool`  | True to use a single-agent with `M` decisions per round.                                | `[false]`                         |
| `config_override`     | `str`             | Name of a .json file to override the ones in default_parameters.json.                   | `default_parameters.json`         |                           

There are two kinds of parameters:
- **list parameters**: Indicated by having the type `list` in the table (should be wrapped with `[]` in the .json file but not in the command line, see examples below and in the .json file). Each value in the list corresponds to a separate experiment, and each of these parameters can be either a list with one element, or a list with n elements for some n (**the number of experiments**) that is identical across parameters. For example, <br> `rec_flag = [True, False],M=[1,10],gamma=[3]` <br> is allowed, and in this case `gamma=[3]` is treated as `gamma=[3,3]`, but <br> `rec_flag = [True, False],M=[1,10,100],gamma=[3]` <br> is not allowed. 
- **non-list parameters**: The same value will be repeated across experiments, should **not** be wrapped with `[]` in the command line or .json file. 

The `default_parameters.json` file should hold default values for all the parameters above, and the file used for `config_override` can hold only part of them, in which case only they will override the default ones. 

## Parameters For Plotting
In addition, the file `plot_parameters.json` contains some parameter values for plotting the figure. Please change the values in this file in accordance with the experiments you run (**you cannot determine these as command line inputs or use an override file**).

| Parameter          | Type                 | Description                                                                                   |
|--------------------|----------------------|-----------------------------------------------------------------------------------------------|
| `title`            | `str`                | Figure title.                                                                                 |
| `xlabel`           | `str`                | Label for the horizontal axis.                                                                |
| `ylabel`           | `str`                | Label for the vertical axis.                                                                  |
| `legend`           | `list` of `str`      | Legend for different experiments (make sure there is a legend for each different experiment). | 
| `yaxis_log_flag`   | `bool`               | True to present the vertical axis on a log scale.                                             | 
| `lower_bound_flag` | `bool`               | True to plot the lower bound.                                                                 |
## Running Experiments
Please run the `main.py` script with parameters that determine what experiment to run. For example, for the default parameters type:
```bash
python main.py
```
Which will run the experiment in Figure 1c of the manuscript. When changing the default parameters, command line values have priority over the values defined in the .json file provided in `config_override`, which itself has priority over the values in `default_parameters.json`. For example, 
```bash
python main.py --M 100
```
will use `M=100` players by overriding the value `M=10`, but all other values will be as in `default_parameters.json`. For a `override_parameters.json` file that contains only `M=4`, the following 
```bash
python main.py --config_override override_parameters.json
```
will result in running an experiment with `M=4`, but 
```bash
python main.py --config_override override_parameters.json --M 100
```
will run an experiment with `M=100`. Note that we only include implementation of the single-player VDB algorithm, so regardless of the values of multiplayer parameters such as `M`, the code will run the single-player version.  Change the parameters to produce the different experiments found in the paper:

- **Figure1**:
```bash
python main.py --dataset Six_rankers --T 300000
python main.py
python main.py --dataset Irish --T 300000
```
- **Figure2**:
```bash
python main.py --function_to_run MP_RUCB --M 1 4 10 100
python main.py --function_to_run FYLRUCB --M 1 4 10 100
python main.py --function_to_run FYLRMED --M 1 4 10 100
```
- **Figure3**:
```bash
python main.py --function_to_run MP_RUCB MP_RUCB MP_RUCB VDB --M 10 1 10 1 --average_regret_flag True False False False --single_player_flag False False True False
python main.py --function_to_run FYLRUCB FYLRUCB FYLRUCB VDB --M 10 1 10 1 --average_regret_flag True False False False --single_player_flag False False True False
python main.py --function_to_run FYLRMED FYLRMED FYLRMED VDB --M 10 1 10 1 --average_regret_flag True False False False --single_player_flag False False True False
```
- **Figure4**:
```bash
python main.py --function_to_run MP_RUCB --M 100 --graph complete cycle star path
python main.py --function_to_run FYLRUCB --M 100 --graph complete cycle star path
python main.py --function_to_run FYLRMED --M 100 --graph complete cycle star path
```
- **Figure5**:
```bash
python main.py --function_to_run MP_RUCB --M 100 --graph cycle --gamma 12 25 50
```
- **Figure6**:
```bash
python main.py --function_to_run MP_RUCB --M 100 --graph star --gamma 1 --rec_flag False True
```
