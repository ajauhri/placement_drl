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
            train_dropoff_buckets, time_bins_per_hour)
    
    #max_t = 40
    #for k in sorted(train_dropoff_buckets.keys())[:max_t]:

    all_windows = [time_bins_per_hour, 
            time_bins_per_day*4 + time_bins_per_hour,
            time_bins_per_day*5 + time_bins_per_hour,
            time_bins_per_day*6 + time_bins_per_hour
            ]
    train_windows = range(time_bins_per_day*4 + time_bins_per_hour, 
            time_bins_per_day*7, time_bins_per_day)
    test_window = time_bins_per_hour
    
    
    post_start_cars = {};
    pre_load = 5;
    for w in all_windows:
        req_count = [0] * sim.classes;
        req_arr = [[]] * sim.classes;
        for i in range(len(req_arr)):
            req_arr[i] = [];
        for r_t in range(w-pre_load, w + sim.episode_duration):
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
                        if (p_t >= w):
                            req_arr[pickup_node].append([dropoff_node, travel_t, max(r_t,w)]);
                            req_count[pickup_node] += 1;
            logging.info("Loaded Requests for time bin %d, hour of day %d" % (\
                    r_t, train_time_utils.get_hour_of_day(r_t)))
            sim.req_sizes[r_t] = copy.deepcopy(req_count);

        for d_t in range(w,w+sim.episode_duration):
            if (d_t not in post_start_cars):
                post_start_cars[d_t] = [];
            if d_t in train_dropoff_buckets:
                for i in train_dropoff_buckets[d_t]:
                    dropoff_node, d_lat_idx, d_lon_idx = \
                        geo_utils.get_node(X[i, 5:7])
                    if (dropoff_node >= 0):
                        r_t = train_time_utils.get_bucket(X[i,0]);
                        if (r_t < (w - pre_load) or p_t < w):
                            post_start_cars[d_t].append(dropoff_node);
            logging.info("Loaded Dropoffs for time bin %d, hour of day %d" % (\
                    d_t, train_time_utils.get_hour_of_day(d_t)))
        sim.rrs[w] = req_arr;
    with open(r"rrs.pickle", "wb") as out_file:
        cPickle.dump(sim.rrs, out_file)
        cPickle.dump(sim.req_sizes, out_file)
    with open(r"post_start_cars.pickle", "wb") as out_file:
        cPickle.dump(post_start_cars, out_file)
    sys.exit(0)
    

    with open(r"rrs.pickle", "rb") as input_file:
        sim.rrs = cPickle.load(input_file)
        sim.req_sizes = cPickle.load(input_file)
    with open(r"post_start_cars.pickle","rb") as input_file:
        sim.post_start_cars = cPickle.load(input_file);

    hidden_units = 2048;

    model = A2C(sim, 10, 
                train_windows, test_window,
                sim.classes,
                sim.n_actions, hidden_units)
    model.train()
    '''
    model = Baseline(sim)
    model.run()
    '''

if __name__ == "__main__":
    main()
