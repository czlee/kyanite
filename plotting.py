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

There is one special key, `'experiment'`, which (as you'd expect) specifies the
experiment class. The main thing that makes this key special is that all of the
other arguments depend on it, since different scripts take different arguments.
For this reason, the value attached to `'experiment'` should just be a single
string, the name of the experiment class, as specified in the subcommand to
`'run.py'`.

(The old special key is `'script'`, which is still supported for legacy reasons,
but isn't relevant to any script run using `run.py`.)

The values of all other items, i.e. of all items relating to arguments, are
sequences comprising at least two items. The first item is a string, one of:

- `'filter'`: Skip experiments that don't match this specification. The use of
  this is discouraged.
- `'title'`:  Same as `'filter'`, but also include this spec in the title.
- `'expect'`: Raise an error if any experiment doesn't match this specification.
- `'expect-if'`: Raise an error if any experiment passes the filter (including
  title), but doesn't match this specification.
- `'series'`: Make each different value of this its own series. If multiple
  arguments have this option, then each unique combination of values gets its
  own series.

Note on `'expect'`: Experiments that have a different experiment class are
skipped (filtered) and the other arguments won't check, so all specifications
using `'expect'` are read as "expect if the experiment class matches, ignore
otherwise".

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
from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from IPython.display import display, Markdown


verbosity = 1


_action_values = [
    'expect',
    'filter',
    'title',
    'series',
    'expect-if',
]
_missing_action_values = ['error', 'skip']
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
    'weight_decay_client': 'wd',
    'parameter_schedule': 'parsch',
    'optimizer': 'opt',
    'lr_scheduler': 'lrsch',
    'lr_scheduler_client': 'lrsch',
    'qrange_param_quantile': 'qpq',
    'qrange_client_quantile': 'qcq',
    'parameter_radius_initial': 'Bin',
    'qrange_initial': 'qin',
}


# ==============================================================================
# File management and parsing
# ==============================================================================

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

    if contents['script'] != 'run.py':
        assert 'experiment' not in args
        args['experiment'] = contents['script']

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
    "evaluation.json" file. This recurses over all subdirectories of the given
    paths.
    """

    # as a convenience to the user, check that the paths actually exist
    for path in paths:
        if not Path(path).exists():
            warnings.warn(f"{path} does not exist")
        elif not Path(path).is_dir():
            warnings.warn(f"{path} is not a directory")

    directories = [
        Path(dirpath)
        for path in paths
        for (dirpath, dirnames, filenames) in os.walk(path)
        if "evaluation.json" in filenames
    ]
    return directories


def _duration_str(start, finish):
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
    """Shows timestamp info of the earliest and latest experiments in the given
    directory.
    """
    times = []
    isofmt = '%Y-%m-%dT%H:%M:%S.%f'

    for directory in all_experiment_directories(paths):

        if specs is not None:
            args = get_args(directory)
            if args['experiment'] != specs['experiment'] or len(check_spec_match(args, specs)) > 0:
                continue

        start = datetime.datetime.strptime(get_args_file(directory)['started'], isofmt)
        finish = datetime.datetime.strptime(get_eval(directory)['finished'], isofmt)
        if finish < start:
            warnings.warn(f"Finished before it start: {directory}")
        times.append((start, finish))

    times_of_interest = (
        # (name, column to bold, (start, finish))
        ('first to start', 1, min(times)),
        ('last to finish', 2, max(times, key=lambda x: (x[1], x[0]))),
        ('shortest',       3, min(times, key=lambda x: x[1] - x[0])),  # noqa: E241
        ('longest',        3, max(times, key=lambda x: x[1] - x[0])),  # noqa: E241
    )

    # construct table
    printout = "| experiments | started at | finished at | duration |\n"
    printout += "|--:|:-:|:-:|--:|\n"
    fmt = '%d %b %Y, %H:%M:%S'
    for name, bold_col, (start, finish) in times_of_interest:
        cells = [name, start.strftime(fmt), finish.strftime(fmt), _duration_str(start, finish)]
        cells[bold_col] = "**" + cells[bold_col] + "**"
        printout += "| " + " | ".join(cells) + " |\n"

    display(Markdown(printout))


# ==============================================================================
# Argument specification dictionary handling
# ==============================================================================

def verify_specs(specs):
    """Show a useful warning if 'script' is in `specs` rather than 'experiment'.
    This is mostly to assist with the deprecation of the 'script' key in favor
    of 'experiment', which in turn was because of the refactoring of scripts to
    start from a single "run.py" script. Should be called from top-level
    functions. Operates in-place.
    """
    if 'script' in specs and 'experiment' not in specs:
        warnings.warn("The 'script' key in specs is deprecated, please change it to 'experiment'.")
        specs['experiment'] = specs['script']
        del specs['script']


def iterspecs(specs):
    """Convenience function iterating through an argument specification
    dictionary (as described above). This does some preprocessing, namely:
    - It skips the special "experiment" key. Callers should handle this
      separately.
    - If the "meta" dict isn't given, it fills it in with an empty dict.
    - It raises errors if anything looks wrong.

    Usage:
    ```
    for key, action, spec_value, meta in iterspecs(specs):
        # do things here
    ```
    """

    for key, spec in specs.items():
        if key == 'experiment':
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
        if action in ["filter", "expect", "title", "expect-if"]:
            if spec_value == "__all__":
                warnings.warn(f"spec for {key} has action {action} and value __all__, "
                              "are you sure about this wasn't meant to be series?")
            if isinstance(spec_value, list) or isinstance(spec_value, tuple):
                warnings.warn(f"spec for {key} has action {action} and a list of values, "
                              "are you sure about this wasn't meant to be series?")
            if callable(spec_value):
                warnings.warn(f"spec for {key} has action {action} and is callable, "
                              "are you sure about this wasn't meant to be series?")

        yield key, action, spec_value, meta


def check_spec_match(args: dict, specs: dict, ignore=set()):
    """Checks if the `args` (as returned by `get_args()`) satisfy the `specs`.

    Raises `ValueError` if:
    - the experiment class doesn't match, or
    - a spec with `'expect'` specified doesn't match, or
    - a spec with `'expect-if'` specified doesn't match and the args otherwise
      would match, or
    - a spec with `missing-action: error` wasn't found.

    Otherwise, returns a list of keys that don't match. Callers who just need
    a binary match/nonmatch result should use `not check_spec_match(...)`, since
    if it matches in full, the list of nonmatching keys will be empty.
    """

    # First, check the experiment class.
    if args['experiment'] != specs['experiment']:
        raise ValueError(f"Experiment class didn't match: found {args['experiment']}, "
                         f"specified {specs['experiment']}")

    # Second, check that every key in 'args' is specified

    not_specified_args = set(args.keys()) - set(specs.keys()) - ignore - {'experiment'}
    if not_specified_args:
        raise ValueError(f"Arguments found but not specified: {not_specified_args}")

    # Third, check that values match
    nonmatching_expect = {}     # {key: (arg_value, spec_value)}, keep track of these
    nonmatching_expect_if = {}  # as above
    nonmatching_filter = []

    for key, action, spec_value, meta in iterspecs(specs):

        if key in args:
            arg_value = args[key]

        # cases where argument is missing (a little cumbersome to sift through options)
        elif 'missing-treat-as' in meta:
            arg_value = meta['missing-treat-as']
        elif meta.get('missing-action', 'error') == 'error':
            nonmatching_expect[key] = ('__MISSING__', spec_value)
            continue
        else:
            assert meta['missing-action'] == 'skip'  # only remaining possibility
            nonmatching_filter.append(key)
            continue

        if spec_value == '__all__':
            match = True
        elif callable(spec_value):
            match = spec_value(arg_value)
        elif isinstance(spec_value, list):
            match = arg_value in spec_value
        else:
            match = (arg_value == spec_value)

        if not match:
            if action == 'expect':
                nonmatching_expect[key] = (arg_value, spec_value)
            elif action == 'expect-if':
                nonmatching_expect_if[key] = (arg_value, spec_value)
            else:
                nonmatching_filter.append(key)

    # if it passes the filter and 'expect-if' conditions failed, add them to the list
    if len(nonmatching_filter) == 0:
        nonmatching_expect.update(nonmatching_expect_if)

    if nonmatching_expect:
        print("Non-matching arguments:")
        for key, (arg_value, spec_value) in nonmatching_expect.items():
            print(f"{key}: found {arg_value!r}, specified {spec_value!r}")
        raise ValueError("One or more expected arguments didn't match, see above")

    return nonmatching_filter


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
        if action == 'series'
    )


def get_series_values(args: dict, specs: dict) -> tuple:
    """Extracts the values from `args` that are relevant to series (according to
    `specs`, as defined in "Argument specification dictionaries" above). Returns
    a tuple. This is intended to be used where a hashable type is needed, like
    dict keys.
    """
    arg_values = []

    for key, action, _, meta in iterspecs(specs):
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


# ==============================================================================
# Plotting
# ==============================================================================

def make_axes(n: int, ncols=3, axsize=(8, 5)):
    """Makes and returns handles to `n` axes in subplots."""
    plot_cols = min(ncols, n)
    plot_rows = (n + ncols - 1) // ncols  # round up
    figsize = (axsize[0] * plot_cols, axsize[1] * plot_rows)
    fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize, squeeze=False, sharex=True)
    axs = axs.flatten()
    return axs


def plot_all_dataframes(dataframes: dict, title_suffix=None, xlabel=None, axs=None,
                        nolabel=False, label='', **kwargs):
    """Plots all dataframes in `dataframes`, which is expected to be a dict of
    `pandas.DataFrame` objects.

    If `title_suffix` is provided, it is used to generate a suffix for the
    title, after the field name. A title is only generated if `title_suffix` is
    not None. `xlabel` is the x-axis label. The y-axis label is the
    corresponding key in `dataframes`.

    If `axs` is provided, it must be a list of `matplotlib.axes.Axes` object,
    and the plots will be drawn on these axes rather than creating new ones.

    If `label` is provided, it is prefixed to the label generated by this
    function. However, if `nolabel` is True, it assigns no label to the plot
    series (so it won't appear in the legend), and `label` is ignored.

    Other keyword arguments (if any) are passed to `dataframe.plot()`.
    """

    if axs is None:
        axs = make_axes(len(dataframes))
    elif len(axs) < len(dataframes):
        raise ValueError(f"Not enough axes ({len(axs)}) for {len(dataframes)} series")

    if label:
        label = label + ' '

    for ax, (field, dataframe) in zip(axs, dataframes.items()):
        ax.set_prop_cycle(None)

        for name, series in dataframe.iteritems():
            if nolabel:
                augmented_label = ''
            elif 'sample_size' in series.attrs:
                augmented_label = f"{label}{name} ({series.attrs['sample_size']})"
            else:
                augmented_label = f"{label}{name}"
            series.plot(ax=ax, label=augmented_label, **kwargs)

        ax.legend()
        ax.grid(True)
        ax.set_ylabel(field)
        if xlabel:
            ax.set_xlabel(xlabel)
        if title_suffix:
            ax.set_title(field + "\n" + title_suffix)


# ==============================================================================
# Data management
# ==============================================================================


def collect_all_training_data(paths: Sequence[Path], fields: Sequence[str], specs, quiet=False,
                              other_experiments=[]):
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
    verify_specs(specs)

    data = {}
    matched = 0
    skipped_experiments = Counter()
    skipped_keys = Counter()
    skipped_for_keys = 0

    for directory in all_experiment_directories(paths):
        args = get_args(directory)
        if args['experiment'] != specs['experiment']:
            if args['experiment'] not in other_experiments:
                skipped_experiments[args['experiment']] += 1
            continue

        series = get_series_values(args, specs)
        nonmatching_keys = check_spec_match(args, specs)

        if len(nonmatching_keys) > 0:
            skipped_keys.update(nonmatching_keys)
            skipped_for_keys += 1
            continue

        if series not in data:  # don't use setdefault to avoid generating this every time
            data[series] = {field: pd.DataFrame() for field in fields}

        training = pd.read_csv(directory / "training.csv")
        for field in fields:
            data[series][field][directory] = training[field]

        matched += 1

    if verbosity > 0 and not quiet:
        print_skipped(matched, skipped_experiments, skipped_keys, skipped_for_keys)

    return data


def print_skipped(matched, skipped_experiments, skipped_keys, skipped_for_keys):
    print(f"- Matched {matched} runs")

    if skipped_experiments:
        skipped_list = ", ".join(f"{s} ({c})" for s, c in skipped_experiments.most_common())
        print(f"- Skipping {sum(skipped_experiments.values())} runs using "
              f"{len(skipped_experiments)} other experiment classes: {skipped_list}")

    if skipped_for_keys or skipped_keys:
        skipped_list = ", ".join(f"{k} ({c})" for k, c in skipped_keys.most_common())
        print(f"- Skipping {skipped_for_keys} runs that don't match "
              f"on {len(skipped_keys)} keys: {skipped_list}")


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


# ==============================================================================
# Statistical functions
# ==============================================================================
# Currently, these are always run with `axis=1`, but making `axis` an argument allows us to avoid
# having to create helper functions for things like `np.min(x, axis=1)`.

# Confidence interval code:
# https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data

def confint_lower(x, axis):
    return st.t.ppf(0.025, x.shape[axis] - 1, loc=np.mean(x, axis=axis), scale=st.sem(x, axis=axis))


def confint_upper(x, axis):
    return st.t.ppf(0.975, x.shape[axis] - 1, loc=np.mean(x, axis=axis), scale=st.sem(x, axis=axis))


def quartile_lower(x, axis):
    return np.quantile(x, 0.25, axis=axis)


def quartile_upper(x, axis):
    return np.quantile(x, 0.75, axis=axis)


# Get confidence interval maximum width for easier reporting (useful sometimes)

def confint_width(x, axis):
    return st.t.ppf(0.975, x.shape[axis] - 1, loc=0, scale=st.sem(x, axis=axis))


def get_confint_max_widths(field: str, paths: Sequence[Path], specs: dict):
    data = collect_all_training_data(paths, [field], specs, quiet=True)
    reduced = aggregate_training_chart_data(data, [field], get_series_keys(specs), confint_width)[field]
    return reduced.max()


extra_line_specs = {
    'range': ([np.min, np.max], 1 / 5),
    'quartiles': ([quartile_lower, quartile_upper], 1 / 3),
    'confints': ([confint_lower, confint_upper], 1 / 5),
}


def get_extra_line_spec(extra):
    """Returns a tuple `reduce_fns, thin_factor`, where `reduce_fns` is a list
    of statistical functions to run on data (e.g. mean, max, upper quartile),
    and `thin_factor` is a factor to multiply the original line width by. The
    input `extra` is intended to come from the user and should be either a tuple
    comparising a string and a `thin_factor`, or just a string, in which case a
    default `thin_factor` is returned.
    """
    if isinstance(extra, tuple):  # user override for thin factor
        assert len(extra) == 2, "extra line specs must have two elements (or be just a string)"
        extra, thin_factor = extra
        reduce_fns, _ = extra_line_specs[extra]
        return reduce_fns, thin_factor
    else:
        return extra_line_specs[extra]


# ==============================================================================
# Top-level functions
# ==============================================================================

def plot_averaged_training_charts(
        paths: Sequence[Path], fields: Sequence[str], specs: dict,
        axs=None, extra_lines=[], nolabel=False, linewidth=1.5, xlabel="round", quiet=False,
        ylims=None, **plot_kwargs):
    """Plots training charts (i.e., metrics vs round number) from results in `paths`,
    for each of the metrics specified in `fields`.

    `specs` is an argument specified dictionary as specified above.

    If `axs` is provided, it must be a list of `matplotlib.axes.Axes` object,
    and the plots will be drawn on these axes rather than creating new ones.

    `extra_lines` can be a list containing some subset of `range`, `quartiles` and
    `confints`. These lines will be added to the plots as thinner lines.

    If `nolabel` is True, it assigns no label to the main plot series (so it
    won't appear in the legend).

    If `ylims` is provided, it must be a list of 2-tuples, and each will be passed
    to the corresponding `ax.set_ylim()`. This is just a convenience to avoid
    having to pass in `axs` just to set ylims.

    Other keyword arguments are passed through to the `DataFrame.plot()` function.
    """
    verify_specs(specs)

    data = collect_all_training_data(paths, fields, specs, quiet=quiet)
    series_keys = get_series_keys(specs)
    averages = aggregate_training_chart_data(data, fields, series_keys)
    if axs is None:
        axs = make_axes(len(fields))

    plot_all_dataframes(averages, title_suffix=get_title_string(specs), xlabel=xlabel,
                        axs=axs, nolabel=nolabel, linewidth=linewidth, **plot_kwargs)

    for extra in extra_lines:
        reduce_fns, thin_factor = get_extra_line_spec(extra)
        for reduce_fn in reduce_fns:
            reduced = aggregate_training_chart_data(data, fields, series_keys, reduce_fn=reduce_fn)
            plot_all_dataframes(reduced, axs=axs, nolabel=True, linewidth=linewidth * thin_factor,
                                **plot_kwargs)

    if ylims is not None:
        for ax, ylim in zip(axs, ylims):
            ax.set_ylim(ylim)


def plot_comparison(
        field: str, paths: Sequence[Path], analog_specs: dict, digital_specs: dict,
        ax=None, extra_lines=[], figsize=(8, 5), xlabel="round", linewidth=1.5, quiet=False,
        both_legends=None, label='', **plot_kwargs):
    """Plots a comparison between two groups of plots, analog (solid lines) and
    digital (dash lines).

    `analog_specs` and `digital_specs` are argument specification dictionaries,
    as specified above.

    If `ax` is provided, it must be a `matplotlib.axes.Axes` object, and it is
    used instead of creating a new one.

    `extra_lines` can be a list containing some subset of `range`, `quartiles` and
    `confints`. These lines will be added to the plots as thinner lines.

    `both_legends` indicates whether or not to list analog and digital entries
    in the legend separately. If not provided, the function will choose
    something sensible based on what is being plotted.

    Other keyword arguments are passed through to the `DataFrame.plot()`
    function. This includes `linewidth` and `label`, which are sometimes
    modified before being passed through.
    """
    verify_specs(analog_specs)
    verify_specs(digital_specs)

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.axes()

    digital_linestyle = (0, (4, 2, 1, 2))

    ana_data = collect_all_training_data(paths, [field], analog_specs,
                                         quiet=quiet, other_experiments=digital_specs['experiment'])
    dig_data = collect_all_training_data(paths, [field], digital_specs,
                                         quiet=quiet, other_experiments=analog_specs['experiment'])
    ana_series_keys = get_series_keys(analog_specs)
    dig_series_keys = get_series_keys(digital_specs)
    ana_averages = aggregate_training_chart_data(ana_data, [field], ana_series_keys)
    dig_averages = aggregate_training_chart_data(dig_data, [field], dig_series_keys)

    if both_legends is None:
        # show both legends if they're not the same, or if using "analog" /
        # "digital" linestyle entries wouldn't actually make the legend shorter
        both_legends = (ana_data.keys() != dig_data.keys())
        both_legends = both_legends or (len(ana_data) + len(dig_data) <= 4)

    # modify the sample sizes to have both analog and digital
    if not both_legends:
        for (_, ana), (_, dig) in zip(ana_averages[field].items(), dig_averages[field].items()):
            ana.attrs['sample_size'] = f"{ana.attrs['sample_size']} / {dig.attrs['sample_size']}"

    if label and both_legends:
        label = label + ' '

    plot_all_dataframes(ana_averages, xlabel=xlabel, axs=[ax], linewidth=linewidth,
                        label=label + "analog" if both_legends else label,
                        **plot_kwargs)
    plot_all_dataframes(dig_averages, xlabel=xlabel, axs=[ax], linewidth=linewidth,
                        linestyle=digital_linestyle,
                        nolabel=not both_legends,
                        label=label + "digital" if both_legends else label,
                        **plot_kwargs)

    for extra in extra_lines:
        reduce_fns, thin_factor = get_extra_line_spec(extra)
        for reduce_fn in reduce_fns:
            ana_reduced = aggregate_training_chart_data(ana_data, [field], ana_series_keys, reduce_fn)
            dig_reduced = aggregate_training_chart_data(dig_data, [field], dig_series_keys, reduce_fn)
            plot_all_dataframes(ana_reduced, axs=[ax], linewidth=linewidth * thin_factor,
                                nolabel=True, **plot_kwargs)
            plot_all_dataframes(dig_reduced, axs=[ax], linewidth=linewidth * thin_factor,
                                nolabel=True, linestyle=digital_linestyle, **plot_kwargs)

    # add line type indicators for analog and digital
    if not both_legends:
        x, y = ax.get_children()[0].get_data()
        ax.plot([x[0]], [y[0]], color='black', label="analog",
                linewidth=linewidth, **plot_kwargs)
        ax.plot([x[0]], [y[0]], color='black', label="digital", linestyle=digital_linestyle,
                linewidth=linewidth, **plot_kwargs)
        ax.legend()

    title = "analog vs digital\n" + get_title_string(digital_specs)
    ax.set_title(title)
    ax.set_xlabel("round")
    ax.set_ylabel(field)
