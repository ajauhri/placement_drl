#! /usr/bin/env python3
#./main.py -c cities.csv -i data/<fname> -d

from __future__ import division

import helpers
from sim import Sim
from dqn import DQN
from a2c_pd import A2C
from worker import Worker
from baseline import Baseline

import sys
import os
import argparse 
import pandas
import logging
import numpy as np
from collections import Counter 
import copy
import tensorflow as tf

try:
    import cPickle
except ImportError:
    import _pickle as cPickle

time_bin_width_mins = .5
time_bins_per_day = int(24*60/time_bin_width_mins)
time_bins_per_hour = int(60 / time_bin_width_mins)
cell_length_meters = 100
time_bin_width_secs = int(time_bin_width_mins * 60)
action_dim = 5 # 0 - left, 1 - down, 2 - right, 3 - up, 4 - NOP

def main():
    parser = argparse.ArgumentParser(description='Placement')
    parser.add_argument("-c", "--config", help="path to config file",  
        default="")
    parser.add_argument("-i", "--input", help="path to data file",  
        default="")
    parser.add_argument("-d", "--debug", help="debug mode",  
        action="store_const", dest="loglevel", const=logging.DEBUG, 
        default=logging.WARNING)
    parser.add_argument("-p", "--pickle", help="create pickle files",
            action="store_true", dest="create_pickle")
    parser.add_argument("-v", "--verbose", help="verbose mode",  
        action="store_const", dest="loglevel", const=logging.INFO)

    args = parser.parse_args(sys.argv[1:])
    logging.basicConfig(format='%(asctime)s main: %(message)s', 
            level=args.loglevel)
    
    
    config = pandas.read_csv(args.config, sep=',')
    X = pandas.read_csv(args.input, sep=',', 
            header=None).values.astype(np.float64)
    
    """
    discretize the space into cells
    """
    geo_utils = helpers.GeoUtilities(config['lat_min'].iloc[0], 
                config['lat_max'].iloc[0], 
                config['lng_min'].iloc[0], 
                config['lng_max'].iloc[0],
                cell_length_meters)
    geo_utils.set_grids()
    
    """
    segregate data based on time bins
    """
    train_time_utils = helpers.TimeUtilities(time_bin_width_secs)
    train_time_utils.set_bounds(X)
    logging.info("Loaded %d data points", len(X))
    
    sim = Sim(X, len(geo_utils.lng_grids), train_time_utils, geo_utils, 
            action_dim, time_bins_per_hour)
    
    """
    starting time-step for training and testing time bins (tb)
    """
    test_tb_starts = [time_bins_per_hour]
    train_tb_starts = [time_bins_per_day*4 + time_bins_per_hour,
            time_bins_per_day*5 + time_bins_per_hour,
            time_bins_per_day*6 + time_bins_per_hour]

    train_bins = range(time_bins_per_day*4 + time_bins_per_hour, 
            time_bins_per_day*7, time_bins_per_day)
    
    if args.create_pickle:
        helpers.load_create_pickle(sim,
                train_time_utils,
                geo_utils,
                X,
                train_tb_starts,
                test_tb_starts)

    with open(r"rrs.pickle", "rb") as input_file:
        sim.rrs = cPickle.load(input_file)
        sim.req_sizes = cPickle.load(input_file)
        sim.post_start_cars = cPickle.load(input_file);

    with tf.Session() as sess:
        worker = Worker('worker', sim, 10, train_bins, test_tb_starts, 
                sim.num_cells, sim.n_actions)
        
        sess.run(tf.global_variables_initializer())
        worker.train(sess)
    
    """
    model = A2C(sim, 10, 
                train_windows, test_window,
                sim.num_cells,
                sim.n_actions, hidden_units)
    model.train()
    """
    '''
    model = Baseline(sim)
    model.run()
    '''

if __name__ == "__main__":
    main()
