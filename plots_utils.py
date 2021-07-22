"""Plotting utilities for convenience in Jupyter notebooks."""

# These were originally in misc-plots-1.ipynb.

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Helper functions


abbreviations = {
    'parameter_radius': 'B',
    'noise': 'σₙ²',
    'clients': 'n',
    'power': 'P',
    'data_per_client': 'dpc',
    'quantization_range': 'qr',
    'zero_bits_strategy': 'zbs',
    'power_ema_coefficient': 'ema',
    'power_update_period': 'pup',
    'power_quantile': 'pq',
    'power_factor': 'pf',
}


def get_args(directory):
    """`directory` must be a pathlib.Path object."""
    argsfile = directory / 'arguments.json'
    with open(argsfile) as f:
        content = json.load(f)
    return content['args']


def get_eval(directory):
    """`directory` must be a pathlib.Path object."""
    argsfile = directory / 'evaluation.json'
    with open(argsfile) as f:
        content = json.load(f)
    return content


def fits_spec(args, specs):
    """Returns True if the given `args` dict matches the given `specs` dict,
    for those entries in `specs`."""
    for key, value in specs.items():
        arg = args[key]
        if value == "__all__":  # magic value
            matches = True
        elif isinstance(value, list):
            matches = arg in value
        else:
            matches = (arg == value)
        if not matches:
            return False
    return True


def all_subdirectories(results_dir):
    """Returns a list with all subdirectories in `results_dir`.

    `results_dir` should be either a `pathlib.Path` or a list of `pathlib.Path`s.
    In the latter case, the generator goes through each directory in the list in
    turn.
    """
    if not isinstance(results_dir, list):
        results_dir = [results_dir]
    # We could implement this as a generator, but the code ends up being more unwieldy.
    directories = [e
                   for rd in results_dir
                   for d in rd.iterdir() if d.is_dir()
                   for e in d.iterdir() if e.is_dir()]
    return directories


def fits_all_specs(args, title_specs, fixed_specs, series_specs, ignore_specs):
    """Checks if the `args` satisfy all of the `specs`. An assertion fails if:
     - any argument key is not found in the specs or vice versa
     - the arguments do not fit the `title_specs` or `fixed_specs`
    If the assertions do not fail, then this returns True if the `args` satisfy
    `series_specs`, and False if not.
    """
    found_args = set(args.keys())
    specified_args = set(title_specs) | set(fixed_specs) | set(series_specs) | ignore_specs
    assert found_args <= specified_args, "found but not specified: " + str(found_args - specified_args)
    assert specified_args <= found_args, "specified but not found: " + str(specified_args - found_args)
    assert fits_spec(args, fixed_specs)
    assert fits_spec(args, title_specs)
    return fits_spec(args, series_specs)


def specs_string(specs):
    """Returns a string suitable for representing the keys and values in `specs`.
    `specs` should be something yielding 2-tuples `(key, value)`. If you want to
    pass in a `dict`, use the `dict.items()` method."""
    return ", ".join(f"{abbreviations.get(key, key)}={value}" for key, value in specs)


def plot_all_dataframes(dataframes: dict, title_specs: dict, xlabel: str):
    """Plots all dataframes in `dataframes`, which is expected to be a dict of
    `pandas.DataFrame` objects. `title_specs` is used to generate a suffix for
    the title."""
    plot_cols = 2 if len(dataframes) >= 2 else 1
    plot_rows = (len(dataframes) + 1) // 2  # round up
    figsize = (8 * plot_cols, 5 * plot_rows)
    fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize, squeeze=False, sharex=True)
    axs = axs.flatten()

    title_suffix = specs_string(title_specs.items())

    for ax, (field, dataframe) in zip(axs, dataframes.items()):
        dataframe.plot(ax=ax)
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(field)
        ax.set_title(field + "\n" + title_suffix)


# Main plotting functions

def plot_averaged_training_charts(results_dir: Path, fields: list, title_specs: dict,
                                  fixed_specs: dict, series_specs: dict, retval=False):
    """Plots training charts (i.e., metrics vs round number) from the results
    in `results_dir`, for each of the metrics specified in `fields`.

    `title_specs` and `fixed_specs` are dicts indicating arguments that should always
    match (an assertion fails if any do not match). `series_spec` is a similar dict,
    but the values should be lists or the special string `"__all__"`, and this dict is
    used to differentiate between series.

    All arguments must be in one of the three `specs` dicts. This is to protect against
    accidentally averaging mismatched data. An assertion fails if an argument is found
    in the results directories that is not in any of these dicts. If you want to skip
    certain values of an argument (e.g., only plot those with a noise level of 0.1),
    such values should be placed in `series_specs` as a list of one item.

    If `retval` is true, the data from which the plots are generated is returned.
    """

    # General strategy: Step through each directory, and for each one:
    #  - check that fixed specs match
    #  - determine which series it belongs to based on series specs
    #  - add it to a DataFrame for that series
    # then at the end, take the averages for each DataFrame, and put it in an overall DataFrame.
    # We actually do this for each metric in `fields`, so we track dicts of {field-name: DataFrame}.

    data = {}

    for directory in all_subdirectories(results_dir):
        args = get_args(directory)
        if not fits_all_specs(args, title_specs, fixed_specs, series_specs, {'cpu', 'repeat'}):
            continue

        series = tuple(args[key] for key in series_specs.keys())  # identifier for series
        if series not in data:  # don't use setdefault to avoid generating this every time
            data[series] = {field: pd.DataFrame() for field in fields}

        training = pd.read_csv(directory / "training.csv")
        for field in fields:
            data[series][field][directory] = training[field]

    reduced = {field: pd.DataFrame() for field in fields}

    # Take averages and put them in new DataFrames
    for series in sorted(data.keys()):  # sort tuples to get sensible series order
        series_name = specs_string(zip(series_specs, series))
        sample_size = data[series][field].shape[1]
        series_name += f" ({sample_size})"
        for field in fields:
            reduced[field][series_name] = data[series][field].mean(axis=1)

    plot_all_dataframes(reduced, title_specs, "round")
    if retval:
        return reduced


# function that plots final accuracy vs number of clients, but averaged over many iterations

def plot_evaluation_vs_clients(results_dir: Path, fields: list, title_specs: dict,
                               fixed_specs: dict, series_specs: dict, retval=False):
    """Plots metric vs number of clients from the results in `results_dir`, for each of
    the metrics specified in `fields`.

    The `title_specs`, `fixed_specs` and `series_specs` arguments are the same as in
    `plot_averaged_training_charts()` above."""

    # Similar strategy to plot_averaged_training_charts; there's actually some amount of nonideal
    # code duplication. The main difference is that here we're collecting and averaging for each
    # point, not on a per-series basis.

    clients_range = range(2, 31)
    data = {}
    ignore_specs = {'cpu', 'repeat', 'clients'}

    for directory in all_subdirectories(results_dir):
        args = get_args(directory)
        if not fits_all_specs(args, title_specs, fixed_specs, series_specs, ignore_specs):
            continue

        series = tuple(args[key] for key in series_specs.keys())  # identifier for series
        if series not in data:  # don't use setdefault to avoid generating this every time
            data[series] = {(field, c): [] for field in fields for c in clients_range}

        evaluation = get_eval(directory)
        for field in fields:
            data[series][(field, args['clients'])].append(evaluation[field])

    reduced = {field: pd.DataFrame() for field in fields}

    # Take averages and put them in DataFrames
    for series in sorted(data.keys()):  # sort tuples to get sensible series order
        series_name = specs_string(zip(series_specs, series))
        for field in fields:
            for c in clients_range:
                samples = np.array(data[series][(field, c)])
                reduced[field].loc[c, series_name] = samples.mean()

    plot_all_dataframes(reduced, title_specs, "number of clients")
    if retval:
        return reduced
