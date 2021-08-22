"""Plotting functions.

This is the second generation of this file, and it supersedes the utilities in
plot_utils.py.

Argument specification dictionaries
-----------------------------------
Callers specify which experiments they want to include using dictionaries that
specify what to include and exclude. This is sort of like a mini-language of
sorts, though it relies a little more on magic values than would ordinarily be
advisable.

The keys are the argument names, e.g. `'clients'`, `'noise'`, `'lr_client'`.
Every argument that may be encountered must be specified explicitly. This is to
avoid accidentally averaging values that should be treated as different cases.

There is one special key, `'script'`, which (as you'd expect) specifies the
script. The main thing that makes this key special is that all of the other
arguments depend on it, since different scripts take different arguments. For
this reason, the value attached to `'script'` should just be a single string,
the name of the script.

The values of all other items, i.e. of all items relating to arguments, are
sequences comprising at least two items. The first item is a string, one of:

- `'expect'`: Raise an error if any experiment doesn't match this specification.
- `'filter'`: Skip experiments that don't match this specification. The use of
  this is discouraged, except for the `'script'` argument.
- `'title'`:  Same as `'filter'`, but also include this spec in the title.
- `'series'`: Make each different value of this its own series. If multiple
  arguments have this option, then each unique combination of values gets its
  own series.
- `'series-expect'`: Like `'expect'`, but raises an error if anything doesn't
  match.

The second item is a value or a list of values that the argument should match.
If it's a list, it counts if it matches any value. It can also be the magic
string '__all__', which will always match --- this should only be used with
`'series'`.

The third item is optional and, if provided, is a dict with extra processing
options. Currently, the only options here specify what to do if the argument
isn't present. This normally happens with experiments that are run
before the argument was added to the jadeite scripts. There are two valid keys
associated with this, here's what they mean:

- `'missing-action'`: Can contain any of these values:
  - `'error'`: This is the default. If the argument isn't present, it raises an
  error.
  - `'skip'`: Skip this experiment, i.e. treat it as if it didn't exist.
- `'missing-treat-as'`: The value should be a default argument value, and if the
  argument is missing, the script will treat it as if it were this value. This
  is mostly useful for specifying what the "old" default was, before the option
  was added. To treat missing argument as their own value, use this option, and
  specify a value of the correct type, but that you know won't be used by any of
  the data.

The `'missing-action'` and `'missing-treat-as'` keys are mutually exclusive
(if you specify one, you can't specify the other).

"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# August 2021


import datetime
import json
import os
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, Markdown


_action_values = [
    'expect',
    'filter',
    'title',
    'series',
    'series-expect',
]
_missing_action_values = ['error', 'skip']
_action_series_values = ['series', 'series-expect']
_action_expect_values = ['expect', 'series-expect']
_meta_keys = {
    'missing-action',
    'missing-treat-as',
}

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
    'learning_rate': 'lr',
    'momentum': 'mom',
    'weight_decay': 'wd',
    'parameter_schedule': 'parsch',
    'optimizer': 'opt',
    'lr_scheduler': 'lrsch',
}


def get_args_file(directory: Path):
    argsfile = directory / 'arguments.json'
    with open(argsfile) as f:
        content = json.load(f)
    return content


def get_args(directory: Path):
    """"Returns a dict representing the arguments used in the experiment whose
    results are in `directory`. This also performs some (slightly hacky)
    pre-processing for legacy compatibility.
    """
    contents = get_args_file(directory)
    args = contents['args']
    args['script'] = contents['script']

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

    # We changed this option to be consistently a str later, to avoid
    # str-to-NoneType comparisons.
    if 'lr_scheduler' in args and args['lr_scheduler'] is None:
        args['lr_scheduler'] = 'none'

    return args


def get_eval(directory: Path):
    evalfile = directory / 'evaluation.json'
    with open(evalfile) as f:
        content = json.load(f)
    return content


def all_experiment_directories(paths):
    """Returns a list with all experiment directories in `paths` that have
    completed. A directory is considered to have finished if it has an
    "evaluation.json" file. This will go
    """
    directories = [
        Path(dirpath)
        for path in paths
        for (dirpath, dirnames, filenames) in os.walk(path)
        if "evaluation.json" in filenames
    ]
    return directories


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


def show_timestamp_info(paths, specs=None):
    """Shows timestamp info of the earliest and latest experiments in the given directory."""
    times = []
    isofmt = '%Y-%m-%dT%H:%M:%S.%f'

    for directory in all_experiment_directories(paths):

        if specs is not None:
            args = get_args(directory)
            if args['script'] != specs['script']:
                continue
            if not fits_all_specs(args, specs):
                continue

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


def _split_spec(spec: Sequence):
    """Returns the 'meta' options dict of the given spec value (as defined in
    "Argument specification dictionaries" above), or an empty dict if the third
    element of the tuple isn't given.
    """
    if len(spec) not in [2, 3]:
        raise ValueError(f"Spec tuple should have 2 or 3 items: {spec}")
    elif len(spec) == 2:
        return *spec, {}
    else:
        return spec


def iterspecs(specs):
    """Convenience function iterating through an argument specification
    dictionary (as described above). This does some preprocessing, namely:
    - It skips the special "script" key. Callers should handle this separately.
    - If the "meta" dict isn't given, it fills it in with an empty dict.
    - It raises errors if anything looks wrong.

    Usage:
    ```
    for key, action, spec_value, meta in iterspecs(specs):
        # do things here
    ```
    """

    for key, spec in specs.items():
        if key == 'script':
            continue

        if len(spec) == 2:
            action, spec_value = spec
            meta = {}
        elif len(spec) == 3:
            action, spec_value, meta = spec
        else:
            raise ValueError(f"spec tuple should have 2 or 3 items: {spec!r}")

        # some combinations aren't allowed
        if action not in _action_values:
            raise ValueError(f"unrecognized action: {action!r}")
        if not isinstance(meta, dict):
            raise ValueError(f"third item in spec must be a dict: {meta!r}")
        if 'missing-treat-as' in meta and 'missing-action' in meta:
            raise ValueError(f"'missing-treat-as' and 'missing-action' both specified: {meta!r}")
        missing_action = meta.get('missing-action', 'error')
        if missing_action not in _missing_action_values:
            raise ValueError(f"unrecognized missing action: {missing_action!r}")
        if not set(meta.keys()) < _meta_keys:
            raise ValueError(f"unrecognized meta keys: {set(meta.keys()) - _meta_keys}")

        # some combinations as strongly inadvisable
        if action in ["filter", "expect", "title"]:
            if spec_value == "__all__":
                warnings.warn(f"spec for {key} has action {action} and value __all__, "
                              "are you sure about this wasn't meant to be series?")
            if isinstance(spec_value, list) or isinstance(spec_value, tuple):
                warnings.warn(f"spec for {key} has action {action} and a list of values, "
                              "are you sure about this wasn't meant to be series?")

        yield key, action, spec_value, meta


def fits_all_specs(args: dict, specs: dict, ignore=set()):
    """Checks if the `args` (as returned by `get_args()`) satisfy the `specs`.

    Raises `ValueError` if:
    - the script doesn't match, or
    - a spec with `'expect'` specified doesn't match, or
    - a spec with `missing-action: error` wasn't found.

    Otherwise, returns False if this experiment should be skipped, or True if it
    should be included.
    """

    # First, check the script.
    if args['script'] != specs['script']:
        raise ValueError(f"Script didn't match: found {args['script']}, specified {specs['script']}")

    # Second, check that every key in 'args' is specified

    not_specified_args = set(args.keys()) - set(specs.keys()) - ignore - {'script'}
    if not_specified_args:
        raise ValueError(f"Arguments found but not specified: {not_specified_args}")

    # Third, check that values match
    nonmatching_fatal = {}  # {key: (arg_value, spec_value)}, keep track of these

    for key, action, spec_value, meta in iterspecs(specs):

        if key in args:
            arg_value = args[key]

        # cases where argument is missing (a little cumbersome to sift through options)
        elif 'missing-treat-as' in meta:
            arg_value = meta['missing-treat-as']
        elif meta.get('missing-action', 'error') == 'error':
            nonmatching_fatal[key] = ('__MISSING__', spec_value)
            continue
        else:
            assert meta['missing-action'] == 'skip'  # only remaining possibility
            return False

        if spec_value == '__all__':
            match = True
        elif isinstance(spec_value, list):
            match = arg_value in spec_value
        else:
            match = (arg_value == spec_value)

        if not match:
            if action in _action_expect_values:
                nonmatching_fatal[key] = (arg_value, spec_value)
                continue
            else:
                return False

    if nonmatching_fatal:
        print("Non-matching arguments:")
        for key, (arg_value, spec_value) in nonmatching_fatal.items():
            print(f"{key}: found {arg_value!r}, specified {spec_value!r}")
        raise ValueError("One or more expected arguments didn't match, see above")

    return True


def specs_string(specs, chunk_size=3):
    """Returns a string suitable for representing the keys and values in `specs`.
    `specs` should be something yielding 2-tuples `(key, value)`. If you want to
    pass in a `dict`, use the `dict.items()` method."""
    parts = [f"{abbreviations.get(key, key)}={value}" for key, value in specs]
    if chunk_size is None:
        return ", ".join(parts)
    else:
        lines = [", ".join(parts[i:i + chunk_size]) for i in range(0, len(parts), chunk_size)]
        return "\n".join(lines)


def get_series_keys(specs) -> tuple:
    """Extracts the keys from `specs` that are marked as "series"."""
    return tuple(
        key
        for key, action, _, _ in iterspecs(specs)
        if action in _action_series_values
    )


def get_series_values(args: dict, specs: dict) -> tuple:
    """Extracts the values from `args` that are relevant to series (according to
    `specs`, as defined in "Argument specification dictionaries" above). Returns
    a tuple. This is intended to be used where a hashable type is needed, like
    dict keys.
    """
    arg_values = []

    for key, action, spec_value, meta in iterspecs(specs):
        if action not in ['series', 'series_expect']:
            continue
        elif key not in args:
            arg_value = meta.get('missing-treat-as', '__MISSING__')
        else:
            arg_value = args[key]
        arg_values.append(arg_value)

    return tuple(arg_values)


def get_title_string(specs, chunk_size=3):
    """Extracts the title string from `specs`."""
    title_specs = [
        (key, spec_value)
        for key, action, spec_value, _ in iterspecs(specs)
        if action == 'title'
    ]
    return specs_string(title_specs, chunk_size)


def make_axes(n, ncols=3, axsize=(8, 5)):
    """Makes and returns handles to `n` axes in subplots."""
    plot_cols = min(ncols, n)
    plot_rows = (n + ncols - 1) // ncols  # round up
    figsize = (axsize[0] * plot_cols, axsize[1] * plot_rows)
    fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize, squeeze=False, sharex=True)
    axs = axs.flatten()
    return axs


# Main plotting functions


def plot_all_dataframes(dataframes: dict, title_suffix=None, xlabel=None, axs=None,
                        nolabel=False, **kwargs):
    """Plots all dataframes in `dataframes`, which is expected to be a dict of
    `pandas.DataFrame` objects.

    If `title_suffix` is provided, it is used to generate a suffix for the
    title, after the field name. A title is only generated if `title_suffix` is
    not None. `xlabel` is the x-axis label. The y-axis label is the
    corresponding key in `dataframes`.

    If `axs` is provided, it must be a list of `matplotlib.axes.Axes` object,
    and the plots will be drawn on these axes rather than creating new ones.

    If `nolabel` is True, it assigns no label to the plot series (so it won't
    appear in the legend).

    Other keyword arguments (if any) are passed to `dataframe.plot()`.
    """

    if axs is None:
        axs = make_axes(len(dataframes))
    elif len(axs) < len(dataframes):
        raise ValueError(f"Not enough axes ({len(axs)}) for {len(dataframes)} series")

    for ax, (field, dataframe) in zip(axs, dataframes.items()):
        ax.set_prop_cycle(None)

        for name, series in dataframe.iteritems():
            if nolabel:
                label = ''
            elif 'sample_size' in series.attrs:
                label = f"{name} ({series.attrs['sample_size']})"
            else:
                label = name
            series.plot(ax=ax, label=label, **kwargs)

        ax.legend()
        ax.set_ylabel(field)
        if xlabel:
            ax.set_xlabel(xlabel)
        if title_suffix:
            ax.set_title(field + "\n" + title_suffix)


def collect_all_training_data(paths: Sequence[Path], fields: Sequence[str], specs, quiet=False):
    """Returns a dict of dicts of `DataFrame` objects, representing all training
    data relevant to the specifications in `specs`. For example:

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

    The keys of the top-level dict are values corresponding to series
    specifications in `specs`. In the example above, say that
    `series_specs=['clients', 'noise']` was passed in, then the first item
    corresponds to 5 clients and a noise level of 1.0.

    The keys of the inner dicts are the field names in `fields`.

    `specs` is as described in "Argument specification dictionaries" above.
    """

    data = {}
    skipped_script = 0
    skipped_series = set()

    for directory in all_experiment_directories(paths):
        args = get_args(directory)
        if args['script'] != specs['script']:
            skipped_script += 1
            continue

        series = get_series_values(args, specs)

        if not fits_all_specs(args, specs):
            skipped_series.add(series)
            continue

        if series not in data:  # don't use setdefault to avoid generating this every time
            data[series] = {field: pd.DataFrame() for field in fields}

        training = pd.read_csv(directory / "training.csv")
        for field in fields:
            data[series][field][directory] = training[field]

    if skipped_script and not quiet:
        display(Markdown(f"- Skipping {skipped_script} experiments using a different script"))
    if skipped_series and not quiet:
        display(Markdown(f"- Skipping {len(skipped_series)} series that don't match specs"))

    return data


def aggregate_training_chart_data(data: dict, fields: list, series_keys: list, reduce_fn=np.mean):
    """Returns a dict` mapping the field names in `fields` to `DataFrame`
    objects containing the (typically) average of that field in the training
    data in `data`.

    `data` is meant to be a dict that is returned by `collect_all_training_data()`.

    If `reduce_fn` is provided, that is the function used to aggregate training
    chart data. This will be called with the `axis=1` argument, so it should
    normally be a numpy function like `np.max` or `np.mean`. By default, the
    mean is taken.
    """
    reduced = {field: pd.DataFrame() for field in fields}

    for series in sorted(data.keys()):  # sort tuples to get sensible series order
        series_name = specs_string(zip(series_keys, series))
        sample_size = data[series][fields[0]].shape[1]
        for field in fields:
            reduced[field][series_name] = reduce_fn(data[series][field], axis=1)
            reduced[field][series_name].attrs['sample_size'] = sample_size

    return reduced


def plot_averaged_training_charts(
        paths: Sequence[Path], fields: Sequence[str], specs: dict,
        axs=None, add_lines=[], nolabel=False, linewidth=1.5, xlabel="round",
        **plot_kwargs):
    """Plots training charts (i.e., metrics vs round number) from results in `paths`,
    for each of the metrics specified in `fields`.

    `specs` is an argument specified dictionary as specified above.

    If `axs` is provided, it must be a list of `matplotlib.axes.Axes` object,
    and the plots will be drawn on these axes rather than creating new ones.

    `add_lines` can be a list containing some subset of `range`, `quartiles` and
    `confints`. These lines will be added to the plots as thinner lines.

    If `nolabel` is True, it assigns no label to the main plot series (so it
    won't appear in the legend).

    Other keyword arguments are passed through to the `DataFrame.plot()` function.
    """

    plot_kwargs['linewidth'] = linewidth

    data = collect_all_training_data(paths, fields, specs)
    series_keys = get_series_keys(specs)
    averages = aggregate_training_chart_data(data, fields, series_keys)
    if axs is None:
        axs = make_axes(len(fields))

    plot_all_dataframes(averages, title_suffix=get_title_string(specs), xlabel=xlabel,
                        axs=axs, nolabel=nolabel, **plot_kwargs)
