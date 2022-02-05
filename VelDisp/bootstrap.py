#!/usr/bin/env python

import numpy as np
from multiprocessing import Pool

def multithreader(func,lol,cores=1):
    with Pool(processes=cores) as pool:
        ls_out = pool.map(func,lol)
    pool.close()
    pool.join()
    return ls_out

def bootstrap(data, bootnum=100, samples=None, bootfunc=None,cores=1):
    if samples is None:
        samples = data.shape[0]

    # make sure the input is sane
    if samples < 1 or bootnum < 1:
        raise ValueError("neither 'samples' nor 'bootnum' can be less than 1.")

    if bootfunc is None:
        resultdims = (bootnum,) + (samples,) + data.shape[1:]
    else:
        # test number of outputs from bootfunc, avoid single outputs which are
        # array-like
        try:
            resultdims = (bootnum, len(bootfunc(data)))
        except TypeError:
            resultdims = (bootnum,)

    # create empty boot array
    boot = np.empty(resultdims)

    bootarr = np.random.randint(low=0, 
                                high=data.shape[0], 
                                size=(bootnum,samples))

    dataarr = data[bootarr]

    boot = multithreader(bootfunc,dataarr,cores)
    
    return np.array(boot)