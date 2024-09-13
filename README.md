rbrmoored
=========

Data processing for moored RBR sensors.
Currently the package only processes temperature channels as it is geared towards processing data from RBR Solo thermistors.
The package wraps RBRs [pyrsktools](https://docs.rbr-global.com/pyrsktools/index.html).


Features
--------
* Process RBR Solo thermistor data including conversion to netcdf with proper meta data.
* Apply time offset.
* Compare time against pre- and post-recovery clock calibration (warm water dip).


Examples
--------
An example jupyter notebook including a little test data file can be found in [notebooks](./notebooks).


Development
-----------
A conda environment for development and testing can be installed using `environment.yml`.
