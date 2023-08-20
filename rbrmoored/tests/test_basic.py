import pathlib
import xarray as xr
import rbrmoored as rbr


def test_success():
    assert True


# We defined rootdir as a fixture in conftest.py
# and can use it here as input now
def test_proc(rootdir, tmp_path):
    rskfile = rootdir / "data/076611_20201012_1352.rsk"
    # tmpdir = pathlib.Path(tmpdir)
    tmpdir = tmp_path
    assert type(rskfile) == pathlib.PosixPath
    print(rskfile)
    assert rskfile.exists()
    solo = rbr.solo.proc(rskfile, tmpdir, figure_out=None, cal_time=None, show_plot=False)

    # make sure we can write and read the data as netcdf
    p = pathlib.Path(tmpdir) / "testfile.nc"
    solo.to_netcdf(p)
    cx2 = xr.open_dataarray(p)
    assert type(cx2) == xr.core.dataarray.DataArray

    assert cx2.attrs["time drift in ms"] == solo.attrs["time drift in ms"]
