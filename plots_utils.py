"""Plotting utilities for convenience in Jupyter notebooks."""

# These were originally in misc-plots-1.ipynb.

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import datetime
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, Markdown


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
    'rounding_method': 'rdm',
    'lr_client': 'lr',
    'momentum_client': 'mom',
}


def get_args_file(directory):
    """`directory` must be a pathlib.Path object."""
    argsfile = directory / 'arguments.json'
    with open(argsfile) as f:
        content = json.load(f)
    return content


def get_args(directory):
    """`directory` must be a pathlib.Path object."""
    args = get_args_file(directory)['args']

    # Ignore parameters that don't affect plots
    if 'save_models' in args:
        del args['save_models']
    if 'cpu' in args:
        del args['cpu']
    if 'repeat' in args:
        del args['repeat']

    # We removed the --small argument when adding support for other datasets,
    # if the option is there, replace it with the new option.
    if 'small' in args:
        args['dataset'] = 'epsilon-small' if args['small'] else 'epsilon'
        del args['small']

    # We added the --momentum-client argument late too.
    if 'momentum_client' not in args:
        args['momentum_client'] = 0.0

    return args


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
        arg = args.get(key, "__missing__")
        if value == "__all__":  # magic value
            matches = True
        elif isinstance(value, list):
            matches = arg in value
        else:
            matches = (arg == value)
        if not matches:
            return False
    return True


def _duration_str(times):
    start, finish = times
    duration = finish - start
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if duration.days > 0:
        return f"{duration.days} days, {hours:02} h {minutes:02} min {seconds:02} s"
    elif hours > 0:
        return f"{hours} h {minutes:02} min {seconds:02} s"
    elif minutes > 0:
        return f"{minutes} min {seconds:02} s"
    else:
        return f"{seconds} s"


def show_timestamp_info(results_dir):
    """Shows timestamp info of the earliest and latest experiments in the given directory."""
    times = []
    isofmt = '%Y-%m-%dT%H:%M:%S.%f'

    for directory in all_subsubdirectories(results_dir):
        started_str = get_args_file(directory)['started']
        started = datetime.datetime.strptime(started_str, isofmt)
        try:
            finished_str = get_eval(directory)['finished']
        except FileNotFoundError:
            warnings.warn(f"No evaluation.json file in {directory}")
            continue
        finished = datetime.datetime.strptime(finished_str, isofmt)
        if finished < started:
            warnings.warn(f"Finished before it started: {directory}")
        times.append((started, finished))

    times_of_interest = {  # ((start, finish), column_to_bold)
        'first to start': (min(times), 1),
        'last to finish': (max(times, key=lambda x: (x[1], x[0])), 2),
        'shortest': (min(times, key=lambda x: x[1] - x[0]), 3),
        'longest': (max(times, key=lambda x: x[1] - x[0]), 3),
    }

    printout = "| experiments in this directory | started at | finished at | duration |\n"
    printout += "|--:|:-:|:-:|--:|\n"
    fmt = '%d %b %Y, %H:%M:%S'
    for name, (times, column_to_bold) in times_of_interest.items():
        cells = [name, times[0].strftime(fmt), times[1].strftime(fmt), _duration_str(times)]
        cells[column_to_bold] = "**" + cells[column_to_bold] + "**"
        printout += "| " + " | ".join(cells) + " |\n"

    display(Markdown(printout))


def all_subsubdirectories(results_dir):
    """Returns a list with all subsubdirectories in `results_dir`, as in, all
    directories two levels down.

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

    # older directories are not necessarily composite
    if len(directories) == 0:
        directories = [d for rd in results_dir for d in rd.iterdir() if d.is_dir()]

    return directories


def fits_all_specs(args, title_specs, fixed_specs, series_specs, ignore_specs=set()):
    """Checks if the `args` satisfy all of the `specs`. An assertion fails if:
     - any argument key is not found in the specs or vice versa
     - the arguments do not fit `fixed_specs`
    If the assertions do not fail, then this returns True if the `args` satisfy
    `series_specs`, and False if not.
    """
    found_args = set(args.keys()) - ignore_specs
    optional_args = {key for key, value in fixed_specs.items()
                     if isinstance(value, list) and '__missing__' in value}
    specified_args = (set(title_specs) | set(fixed_specs) | set(series_specs)) - ignore_specs
    if found_args > specified_args:
        raise AssertionError("found but not specified: " + str(found_args - specified_args))
    if specified_args - optional_args > found_args:
        raise AssertionError("specified but not found: " + str(specified_args - found_args))
    assert fits_spec(args, fixed_specs), str(args)
    return fits_spec(args, title_specs) and fits_spec(args, series_specs)


def specs_string(specs):
    """Returns a string suitable for representing the keys and values in `specs`.
    `specs` should be something yielding 2-tuples `(key, value)`. If you want to
    pass in a `dict`, use the `dict.items()` method."""
    return ", ".join(f"{abbreviations.get(key, key)}={value}" for key, value in specs)


def make_axes(n):
    """Makes and returns handles to `n` axes in subplots."""
    plot_cols = min(3, n)
    plot_rows = (n + 2) // 3  # round up
    figsize = (8 * plot_cols, 5 * plot_rows)
    fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize, squeeze=False, sharex=True)
    axs = axs.flatten()
    return axs


def plot_all_dataframes(dataframes: dict, title_specs=None, xlabel=None, axs=None, **kwargs):
    """Plots all dataframes in `dataframes`, which is expected to be a dict of
    `pandas.DataFrame` objects.

    `title_specs` is used to generate a suffix for the title. `xlabel` is the
    x-axis label. (The y-axis label is the corresponding key in `dataframes`.)
    If neither of these arguments are provided, the legend, axis labels and
    title will not be set.

    If `axs` is provided, it must be a list of `matplotlib.axes.Axes` object,
    and the plots will be drawn on these axes rather than creating new ones.

    Other keyword arguments (if any) are passed to `dataframe.plot()`.
    """

    if axs is None:
        axs = make_axes(len(dataframes))
    elif len(axs) < len(dataframes):
        raise ValueError(f"Not enough axes ({len(axs)}) for {len(dataframes)} series")

    if title_specs:
        title_suffix = specs_string(title_specs.items())

    for ax, (field, dataframe) in zip(axs, dataframes.items()):
        ax.set_prop_cycle(None)
        dataframe.plot(ax=ax, **kwargs)

        if title_specs or xlabel:
            ax.legend()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(field)
            ax.set_title(field + "\n" + title_suffix)


# Main plotting functions

def collect_all_training_data(results_dir: Path, fields: list, title_specs: dict,
                              fixed_specs: dict, series_specs: dict):
    """Returns a `dict` of `dict`s of `DataFrame` objects, representing all
    training data relevant to the specifications in `title_specs`, `fixed_specs`
    and `series_specs`. For example:
        {
            (5, 1.0): {
                'accuracy': <DataFrame object>,
                'test_loss': <DataFrame object>
            },
            (20, 1.0): {
                'accuracy': <DataFrame object>,
                'test_loss': <DataFrame object>
            }
        }

    The keys of the top-level dict are values corresponding to the
    specifications in `series_specs`. In the example above, say that
    `series_specs=['clients', 'noise']` was passed in, then the first item
    corresponds to 5 clients and a noise level of 1.0.

    The keys of the inner dicts are the field names in `fields`.

    `title_specs`, `fixed_specs` and `series_specs` are as described in
    `plot_averaged_training_charts()`.
    """
    data = {}

    for directory in all_subsubdirectories(results_dir):
        args = get_args(directory)
        if not fits_all_specs(args, title_specs, fixed_specs, series_specs):
            continue

        series = tuple(args[key] for key in series_specs.keys())  # identifier for series
        if series not in data:  # don't use setdefault to avoid generating this every time
            data[series] = {field: pd.DataFrame() for field in fields}

        training = pd.read_csv(directory / "training.csv")
        for field in fields:
            data[series][field][directory] = training[field]

    return data


def aggregate_training_chart_data(data: dict, fields: list, series_labels: list, reduce_fn=np.mean):
    """Returns a dict` mapping the field names in `fields` to `DataFrame`
    objects containing the (typically) average of that field in the training
    data in `data`.

    `data` is meant to be a dict

    Most arguments are as described in `plot_averaged_training_charts()`.

    If `reduce` is provided, that is the function used to aggregate training
    chart data. This will be called with the `axis=1` argument, so it should
    normally be a numpy function like `np.max` or `np.mean`. By default, the
    mean is taken.
    """
    reduced = {field: pd.DataFrame() for field in fields}

    for series in sorted(data.keys()):  # sort tuples to get sensible series order
        series_name = specs_string(zip(series_labels, series))
        sample_size = data[series][fields[0]].shape[1]
        series_name += f" ({sample_size})"

        for field in fields:
            reduced[field][series_name] = reduce_fn(data[series][field], axis=1)

    return reduced


def plot_averaged_training_charts(results_dir: Path, fields: list, title_specs: dict,
                                  fixed_specs: dict, series_specs: dict, axs=None,
                                  plot_range=False):
    """Plots training charts (i.e., metrics vs round number) from the results
    in `results_dir`, for each of the metrics specified in `fields`.

    `fixed_specs` is a dict indicating arguments that should always match (an
    assertion fails if any do not match).

    `title_specs` is a similar dict, but matching is not mandatory (it skips
    non-matching directories) and this dict is used to generate thetitle.

    `series_spec` is a similar dict, but the values should be lists or the
    special string `"__all__"`, and this dict is used to differentiate between
    series.

    All arguments must be in at least one of the three `specs` dicts. This is to
    protect against accidentally averaging mismatched data. An assertion fails
    if an argument is found in the results directories that is not in any of
    these dicts. If you want to skip certain values of an argument (e.g., only
    plot those with a noise level of 0.1), such values should be placed in
    `series_specs` as a list of one item.

    If `axs` is provided, it must be a list of `matplotlib.axes.Axes` object,
    and the plots will be drawn on these axes rather than creating new ones.

    If `plot_range` is True, it also plots the minimum and maximum for each
    series on the same plot. This can get messy, so don't do this if you have a
    lot of series.
    """

    # General strategy: Step through each directory, and for each one:
    #  - check that fixed specs match
    #  - determine which series it belongs to based on series specs
    #  - add it to a DataFrame for that series
    # then at the end, take the averages for each DataFrame, and put it in an
    # overall DataFrame. We actually do this for each metric in `fields`, so we
    # track dicts of {field-name: DataFrame}.

    data = collect_all_training_data(results_dir, fields, title_specs, fixed_specs, series_specs)
    averages = aggregate_training_chart_data(data, fields, series_specs.keys())
    if axs is None:
        axs = make_axes(len(fields))
    plot_all_dataframes(averages, title_specs, "round", axs=axs)

    if plot_range:
        minima = aggregate_training_chart_data(data, fields, series_specs.keys(), reduce_fn=np.min)
        maxima = aggregate_training_chart_data(data, fields, series_specs.keys(), reduce_fn=np.max)
        plot_all_dataframes(minima, axs=axs, linewidth=0.5, legend=False)
        plot_all_dataframes(maxima, axs=axs, linewidth=0.5, legend=False)


# function that plots final accuracy vs number of clients, but averaged over many iterations

def plot_evaluation_vs_clients(results_dir: Path, fields: list, title_specs: dict,
                               fixed_specs: dict, series_specs: dict, axs=None):
    """Plots metric vs number of clients from the results in `results_dir`, for
    each of the metrics specified in `fields`.

    The `title_specs`, `fixed_specs` and `series_specs` arguments are the same
    as in `plot_averaged_training_charts()` above.

    If `axs` is provided, it must be a list of `matplotlib.axes.Axes` object,
    and the plots will be drawn on these axes rather than creating new ones.
    """

    # Similar strategy to plot_averaged_training_charts; there's actually some
    # amount of nonideal code duplication. The main difference is that here
    # we're collecting and averaging for each point, not on a per-series basis.
    # This makes the data structures quite different, so it's not easily
    # deduplicable.

    clients_range = range(2, 31)
    data = {}
    for directory in all_subsubdirectories(results_dir):
        args = get_args(directory)
        if not fits_all_specs(args, title_specs, fixed_specs, series_specs, {'clients'}):
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

    plot_all_dataframes(reduced, title_specs, "number of clients", axs=axs)


# function that plots analog vs digital plots

def plot_comparison(field, analog_path, digital_path, all_analog_specs, all_digital_specs):
    """Plots analog and digital (or any other two otherwise incompatible
    collections of series) on the same plot.

    `all_analog_specs` and `all_digital_specs` should both be 3-tuples
        (title_specs, fixed_specs, series_specs)
    """

    plt.figure(figsize=(8, 5))
    ax = plt.axes()

    series_specs = all_analog_specs[2]
    analog_data = collect_all_training_data(analog_path, [field], *all_analog_specs)
    analog_averages = aggregate_training_chart_data(analog_data, [field], series_specs.keys())[field]

    title_specs, _, series_specs = all_digital_specs
    digital_data = collect_all_training_data(digital_path, [field], *all_digital_specs)
    digital_averages = aggregate_training_chart_data(digital_data, [field], series_specs.keys())[field]

    all_averages = analog_averages.join(digital_averages, lsuffix=' analog', rsuffix=' digital')
    all_averages.plot(ax=ax)

    title = "analog vs digital\n" + specs_string(title_specs.items())
    plt.title(title)
    plt.xlabel("round")
    plt.ylabel(field)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
