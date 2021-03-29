# DEFINING FUNCTIONS FOR FINDING CLOSEST MODELS ------------------------
import bisect
from os import path
import os

def namefinder_import(filename):
    # filename: name of file containing name patterns, ascending order
    with open(filename) as infile:
        param, pattern = [], []
        for line in infile:
            lnlst = line.split()
            if '#' not in line and len(lnlst) != 0:
                param.append(float(lnlst[1])) # parameter variable to be sorted
                pattern.append(lnlst[0])      # filename pattern associated
        outarr = [param, pattern]
        return outarr

def sorter(paramlst, paramvalue, crit, patt_finder):
    ''' finds the closest value/key in a 2D list to the requested parameter (mass/met)
    :param paramlst: list of parameters, if patt_finder = True, a 2 element list of values and pattern
                        if patt_finder = False, a 1 element(!) list of the list of values
    :param paramvalue: the parameter for which to evaluate the closest criterion
    :param crit: critical value, if all values are farther, all models are rejected
    :param patt_finder: if True, returns a pattern; if False, returns only the closest value
    :return:
        val: value, patt: pattern
        rej: True if all models are rejected
    '''

    ind, val = min(enumerate(paramlst[0]), key=lambda x: abs(x[1] - paramvalue))
    rej = True if abs(val-paramvalue) > crit else False
    if patt_finder == True: # returns a pattern as well
        patt = paramlst[1][ind]
        return val, patt, rej
    else:                   # returns only the closest value
        return val, rej

def interval_find(paramlst, param, crit, patt_finder, rej_fname):
    # function to find all values in a given interval
    bound_up = round(param + crit/2, 3)
    bound_low = round(param - crit/2, 3)

    lower_bound_i = bisect.bisect_left(paramlst[0], bound_low)
    upper_bound_i = bisect.bisect_right(paramlst[0], bound_up, lo=lower_bound_i)
    param_inrange = paramlst[0][lower_bound_i:upper_bound_i]

    nfound = len(param_inrange)
    rej_close = False

    if patt_finder == True:
        patt_inrange = paramlst[1][lower_bound_i:upper_bound_i]
        if nfound == 0:  # if no masses are in the errored region, it will find the closest mass
            # with the sorter function and tell if it is in the tolerated region or not
            paramval_close, patt_close, rej_close = sorter(paramlst, param, crit, True)
            param_inrange.append(paramval_close) # makes it also a list
            patt_inrange.append(patt_close)
            rej_fname.write('no mass in error range, using closest one ({})\n'.format(paramval_close))
        return param_inrange, patt_inrange, rej_close

    else:
        if nfound == 0:  # if no masses are in the errored region, it will find the closest mass
            # with the sorter function and tell if it is in the tolerated region or not
            paramval_close, rej_close = sorter(paramlst, param, crit, False)
            param_inrange.append(paramval_close) # makes it also a list
            rej_fname.write('no mass in error range, using closest one ({})\n'.format(paramval_close))
        return param_inrange, rej_close