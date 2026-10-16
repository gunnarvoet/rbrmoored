# History

## (unreleased)

<!-- ### Breaking changes -->

### New Features

- `solo.plot` takes `offset_decimals` to set the number of decimals of the
  applied time offset printed on the plot. The default of 3 shows the full
  millisecond resolution, now always printed with three decimals.
- `solo.plot` returns the figure.

<!-- ### Bug fixes -->

<!-- ### Documentation -->

<!-- ### Internal Changes -->

## v2026.10

### Breaking changes

- `solo.spike_removal` now detects the periodic RBRsolo³ spike before removing
  anything and leaves records without a detectable spike untouched. Records
  processed with the previous version should be re-processed.
- `solo.spike_removal` subtracts the periodic spike by default
  (`method="subtract"`). Before, it replaced the spike samples by linear
  interpolation, which is still available as `method="interpolate"`. `solo.proc`
  follows the new default unless `spike_removal_kwargs` says otherwise.

### New Features

- `solo.spike_removal` documents what it found in the attributes `spike period
  in s`, `spike phase in s`, `spike amplitude in K`, `spike detection snr` and
  `spikes removed`.
- `solo.spike_removal(method="subtract")` subtracts the periodic spike instead
  of interpolating over it, so the spike samples keep their measurement. The
  spike and `n_tail` samples after it are estimated per block of `block_s`
  seconds as the mean excess over the line through the flanking samples. In 73
  RBRsolo³ loggers sampling at 2 s for 13 months the spike measured +1.13 to
  +1.58 mK, its scatter was 0.95 to 1.07 times that of any other sample, and
  the two samples after it were high by about 0.030 and 0.004 mK. After
  subtraction with the default `n_tail=2` the residual at the spike sample and
  the one after it was below 0.001 mK in three loggers checked.
- `solo.spike_removal(models=...)` selects the instrument models the removal is
  applied to (`None` for any), and `period_s` together with `models=None`
  handles other periodic single-sample artifacts, such as the spike every 32
  samples seen in four of 24 older RBRsolo loggers.
- `solo.proc(spike_removal_kwargs=...)` passes keyword arguments on to
  `solo.spike_removal`.
- `solo.spike_removal` records the attribute `spike removal method`, and with
  `method="subtract"` also `spike template in K` and `spikes not removed`.

### Bug fixes

- `solo.spike_removal` assumed a sampling period of 2 s. At other sampling rates
  the minimum spacing between detected spikes came out wrong, so it removed
  several times more samples than there are spikes (at 0.5 s sampling) or missed
  real spikes (at 2 s sampling). The sampling period is now taken from the
  record.
- `solo.spike_removal` used a peak detector with a threshold relative to a local
  noise estimate, which always finds peaks. On records without the artifact it
  removed good data - 2.5% of all samples in one 0.5 s record. Detection is now
  based on folding the record onto the spike period and comparing against folds
  at detuned periods.
- `__version__` is read from the package metadata instead of a hard-coded string
  that had gone stale, so the `rbrmoored version` attribute written by `read` is
  correct.

### Documentation

### Internal Changes

- `solo.spike_removal` no longer needs dask, which it used but which was never
  declared as a dependency.


## v2026.03

<!-- ### Breaking changes -->

### New Features
- Add despiking procedure for v3 time series.

### Internal Changes
- Speed up time vector correction.
- Move to uv build system.
- Use patched version of pyrsktools to be able to use numpy>=2.0.0.


## v2024.09

### New Features
- Allow to provide sensor drift time to overwrite auto-generated sensor drift during data download in Ruskin.

### Internal Changes
- Switch from setup.py to pyproject.toml.


## v2024.05

### Breaking changes

- Upgrade to `pyrsktools` >= 1.0.0. Prior versions of `pyrsktools` will not work.

### New Features

- Add options to shift sensor time or recorded time offset prior to data processing.

### Documentation

- Add pdoc documentation

### Internal Changes

- Change code to use [pyrsktools](https://docs.rbr-global.com/pyrsktools) >=1.0.0.
- Remove `gvpy` dependency.


## 0.1.0

- Initial release.
