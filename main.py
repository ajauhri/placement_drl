#! /usr/bin/env python3
#./main.py -c cities.csv -i data/<fname> -t data/<fname> -d

from __future__ import division

import helpers
from sim import Sim
from dqn import DQN

import sys
import os
import argparse 
import pandas
import logging
import numpy as np
from collections import Counter 

time_bin_width_mins = 2
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
    parser.add_argument("-t", "--test", help="path to test file",  
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
    Y = pandas.read_csv(args.test, sep=',', 
            header=None).values.astype(np.float64)
    
    geo_utils = helpers.GeoUtilities(config['lat_min'].iloc[0], 
                config['lat_max'].iloc[0], 
                config['lng_min'].iloc[0], 
                config['lng_max'].iloc[0],
                cell_length_meters)
    geo_utils.set_grids()
    
    # segregate train data based on time
    train_time_utils = helpers.TimeUtilities(time_bin_width_secs)
    train_time_utils.set_bounds(X)
    train_pickup_buckets = train_time_utils.get_buckets(X, 0)
    train_dropoff_buckets = train_time_utils.get_buckets(X, 4)
    logging.info("Loaded training %d data points", len(X))
    
    # segregate test data based on time
    """
    test_time_utils = helpers.TimeUtilities(time_bin_width_secs)
    test_time_utils.set_bounds(Y)
    test_pickup_buckets = test_time_utils.get_buckets(Y, 0)
    test_dropoff_buckets = test_time_utils.get_buckets(Y, 4)
    logging.info("Loaded test %d data points", len(Y))
    """

    
    # segregate data based on time
    #city = config['city'].iloc[0]
    sim = Sim(len(geo_utils.lng_grids), train_time_utils, geo_utils, action_dim,
            train_dropoff_buckets)
    for k in sorted(train_dropoff_buckets.keys())[:720]:
        if len(train_dropoff_buckets[k]) and (k+1) in train_pickup_buckets:
            rrs = {}
            
            for i in train_pickup_buckets[k+1]:
                dropoff_node, d_lat_idx, d_lon_idx = \
                        geo_utils.get_node(X[i, 5:7])
                pickup_node, p_lat_idx, p_lon_idx = \
                        geo_utils.get_node(X[i, 2:4])
                 
                
                d_t = train_time_utils.get_bucket(X[i, 4])
                p_t = train_time_utils.get_bucket(X[i, 1])
                if pickup_node not in sim.rrs:
                    sim.rrs[pickup_node] = []
                sim.rrs[pickup_node].append(helpers.rr(i, k+1, p_t, d_t, 
                    dropoff_node, d_lat_idx, d_lon_idx, 
                    pickup_node, p_lat_idx, p_lon_idx))
            
            #sim.add_maps(k, dropoff_map, pickup_map)
            sim.add_rrs(k+1, rrs)
            logging.info("Loaded map for time bin %d, hour of day %d" % (k, 
                train_time_utils.get_hour_of_day(k)))
    """
    for k in sorted(test_dropoff_buckets.keys())[:61]:
        if len(test_dropoff_buckets[k]) and (k+1) in test_pickup_buckets:
            
            dropoff_map = Counter()
            pickup_map = Counter()

            for i in dropoff_buckets[k]:
                dropoff_node, lat_idx, lon_idx = geo_utils.get_node(Y[i, 5:7])
                dropoff_map[dropoff_node] += 1
            for i in pickup_buckets[k+1]:
                pickup_node, lat_idx, lon_idx = geo_utils.get_node(Y[i, 2:4])
                pickup_map[pickup_node] += 1
            sim.add_maps(k, dropoff_map, pickup_map, True)
            logging.info("Loaded test map for time bin %d, %d" % (k, 
                test_time_utils.get_hour_of_day(k)))
    alg = DQN(state_dim, action_dim, 10, sim, len(dropoff_buckets))
    alg.train()
    """

if __name__ == "__main__":
    main()