#! /usr/bin/env python3
#./main.py -c cities.csv -i data/<fname> -d

from __future__ import division

import helpers
from sim import Sim
from dqn import DQN
from a2c_pd import A2C
from baseline import Baseline

import sys
import os
import argparse 
import pandas
import logging
import numpy as np
from collections import Counter 
import copy
import _pickle as cPickle
#import cPickle

time_bin_width_mins = 3
cell_length_meters = 100
time_bin_width_secs = time_bin_width_mins * 60
action_dim = 5 # 0 - left, 1 - down, 2 - right, 3 - up, 4 - NOP
state_dim = 3

def main():
    parser = argparse.ArgumentParser(description='Placement')
    parser.add_argument("-c", "--config", help="path to config file",  
        default="")
    parser.add_argument("-i", "--input", help="path to data file",  
        default="")
    parser.add_argument("-d", "--debug", help="debug mode",  
        action="store_const", dest="loglevel", const=logging.DEBUG, 
        default=logging.WARNING)
    parser.add_argument("-v", "--verbose", help="verbose mode",  
        action="store_const", dest="loglevel", const=logging.INFO)

    args = parser.parse_args(sys.argv[1:])
    logging.basicConfig(format='%(asctime)s main: %(message)s', 
            level=args.loglevel)
    
    
    config = pandas.read_csv(args.config, sep=',')
    X = pandas.read_csv(args.input, sep=',', 
            header=None).values.astype(np.float64)
    
    geo_utils = helpers.GeoUtilities(config['lat_min'].iloc[0], 
                config['lat_max'].iloc[0], 
                config['lng_min'].iloc[0], 
                config['lng_max'].iloc[0],
                cell_length_meters)
    geo_utils.set_grids()
    
    """
    print(geo_utils.orthodromic_dist([config['lat_min'].iloc[0], 
        config['lng_min'].iloc[0]],
        [config['lat_min'].iloc[0],config['lng_max'].iloc[0] ]))

    print(geo_utils.orthodromic_dist([config['lat_min'].iloc[0], 
        config['lng_min'].iloc[0]],
        [config['lat_max'].iloc[0],config['lng_min'].iloc[0] ]))
    """

    # segregate train data based on time
    train_time_utils = helpers.TimeUtilities(time_bin_width_secs)
    train_time_utils.set_bounds(X)
    train_request_buckets = train_time_utils.get_buckets(X, 0)
    train_pickup_buckets  = train_time_utils.get_buckets(X, 1)
    train_dropoff_buckets = train_time_utils.get_buckets(X, 4)
    logging.info("Loaded training %d data points", len(X))
    
    # segregate data based on time
    city = config['city'].iloc[0]
    sim = Sim(X, len(geo_utils.lng_grids), train_time_utils, geo_utils, 
            action_dim, 
            train_dropoff_buckets)
    
    #max_t = 40
    #for k in sorted(train_dropoff_buckets.keys())[:max_t]:

    all_windows = [60, 5820, 7260, 8700]
    train_windows = range(1440*4 + 60, 1440*7, 1440)
    test_window = 60
    
    for w in all_windows:
        req_count = [0] * sim.classes;
        req_arr = [[]] * sim.classes;
        for i in range(len(req_arr)):
            req_arr[i] = [];
        for r_t in range(w, w+60):
            if r_t in train_request_buckets:
                for i in train_request_buckets[r_t]:
                    dropoff_node, d_lat_idx, d_lon_idx = \
                            geo_utils.get_node(X[i, 5:7])
                    pickup_node, p_lat_idx, p_lon_idx = \
                            geo_utils.get_node(X[i, 2:4])
                    if (pickup_node >= 0):
                         
                        d_t = train_time_utils.get_bucket(X[i, 4])
                        p_t = train_time_utils.get_bucket(X[i, 1])
                        travel_t = d_t - p_t;

                        req_arr[pickup_node].append([dropoff_node, travel_t, r_t]);
                        req_count[pickup_node] += 1;
                
                logging.info("Loaded map for time bin %d, hour of day %d" % (\
                        r_t, train_time_utils.get_hour_of_day(r_t)))
            sim.req_sizes[r_t] = copy.deepcopy(req_count);
        sim.rrs[w] = req_arr;
    with open(r"rrs.pickle", "wb") as out_file:
        cPickle.dump(sim.rrs, out_file)
        cPickle.dump(sim.req_sizes, out_file)
    sys.exit(0)

    with open(r"rrs.pickle", "rb") as input_file:
        sim.rrs = cPickle.load(input_file)
        sim.req_sizes = cPickle.load(input_file)

    hidden_units = 128;
    model = A2C(sim, 10, 
                train_windows, test_window,
                sim.classes,
                sim.n_actions, hidden_units)
    model.train()
    #model = Baseline(sim)
    #model.run()


if __name__ == "__main__":
    main()
