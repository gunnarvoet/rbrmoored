# History

<!-- ## (unreleased) -->

<!-- ### Breaking changes -->

<!-- ### New Features -->

<!-- ### Bug fixes -->

<!-- ### Documentation -->

<!-- ### Internal Changes -->

## (unreleased)

### Breaking changes

- `solo.spike_removal` now detects the periodic RBRsolo³ spike before removing
  anything and leaves records without a detectable spike untouched. Records
  processed with the previous version should be re-processed.

### New Features

- `solo.spike_removal` documents what it found in the attributes `spike period
  in s`, `spike phase in s`, `spike amplitude in K`, `spike detection snr` and
  `spikes removed`.

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
