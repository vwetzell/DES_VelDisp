#!/usr/bin/env python

import astropy.io.fits as pf
from astropy.table import Table

def read_tbl_data(filename):
    # Takes fits catalog and returns the corresponding data table
    hdul = pf.open(filename)
    tbl_data = hdul[1].data
    hdul.close()
    tbl = Table(tbl_data)
    return tbl
