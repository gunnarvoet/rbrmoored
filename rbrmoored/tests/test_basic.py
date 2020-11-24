import pathlib

import xarray as xr

import rbrmoored as rbr


def test_success():
    assert True


# We defined rootdir as a fixture in conftest.py
# and can use it here as input now
def test_proc(rootdir, tmpdir):
    rskfile = rootdir / "data/076611_20201012_1352.rsk"
    tmpdir = pathlib.Path(tmpdir)
    assert type(rskfile) == pathlib.PosixPath
    print(rskfile)
    assert rskfile.exists()
    # c = ctd.io.CTD(hexfile)
    solo = rbr.solo.proc(rskfile, tmpdir, figure_out=None, cal_time=None, show_plot=False)

    # make sure we can write and read the data as netcdf
    # p = pathlib.Path(tmpdir) / "testfile.nc"
    # cx.to_netcdf(p)
    # cx2 = xr.open_dataset(p)
    # assert type(cx2) == xr.core.dataset.Dataset
