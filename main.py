#! /usr/bin/env python

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
cell_length_meters = 200
time_bin_width_secs = time_bin_width_mins * 60

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
    D = pandas.read_csv(args.input, sep=',', 
            header=None).values.astype(np.float64)
    geo_utils = helpers.GeoUtilities(config['lat_min'].iloc[0], 
                config['lat_max'].iloc[0], 
                config['lng_min'].iloc[0], 
                config['lng_max'].iloc[0],
                cell_length_meters)
    time_utils = helpers.TimeUtilities(time_bin_width_secs)
    
    geo_utils.set_grids()
    
    # segregate data based on time
    time_utils.set_bounds(D)
    pickup_buckets = time_utils.get_buckets(D, 0)
    dropoff_buckets = time_utils.get_buckets(D, 4)
    logging.info("Loaded %d data points", len(D))
    
    # segregate data based on time
    #city = config['city'].iloc[0]
    sim = Sim(len(geo_utils.lng_grids), time_utils)
    
    for k in sorted(dropoff_buckets.keys()):
        if len(dropoff_buckets[k]) and (k+1) in pickup_buckets:
            #print "dropoff day %d, hour %d, dropoffs %d, pickups %d" % (
            #        int(k/24), time_utils.get_hour_of_day(k), len(v), len(pickup_buckets[k+1]))
            
            dropoff_map = Counter()
            pickup_map = Counter()

            for i in dropoff_buckets[k]:
                dropoff_node, lat_idx, lon_idx = geo_utils.get_node(D[i, 5:7])
                dropoff_map[dropoff_node] += 1
            for i in pickup_buckets[k+1]:
                pickup_node, lat_idx, lon_idx = geo_utils.get_node(D[i, 2:4])
                pickup_map[pickup_node] += 1
            sim.add_maps(k, dropoff_map, pickup_map)
            logging.info("Loaded map for time bin %d" % k)
            if k == 1000:
                break
            
            """
            n, lat_idx, lng_idx = geo_utils.get_node(D[v[0], 2:4])
            print len(geo_utils.lng_grids), n, len(geo_utils.lat_grids)
            #geo_utils.get_centroid(n)
            for a, n in  geo_utils.get_action_space(lat_idx, 
                    lng_idx).iteritems():
                print a, n, geo_utils.get_centroid(n)
            """
    #print len(geo_utils.lng_grids)
    """
    print sim.step(dropoff_node, 32, -1), dropoff_node
    print sim.step(dropoff_node, 32, 0), dropoff_node
    print sim.step(dropoff_node, 32, 1), dropoff_node
    print sim.step(dropoff_node, 32, 2), dropoff_node
    print sim.step(dropoff_node, 32, 3), dropoff_node
    print sim.step(pickup_node, 32, -1), pickup_node
    print sim.step(pickup_node, 32, 0), pickup_node
    print sim.step(pickup_node, 32, 1), pickup_node
    print sim.step(pickup_node, 32, 2), pickup_node
    print sim.step(pickup_node, 32, 3), pickup_node
    """

    alg = DQN(3, 5, 10, sim, len(dropoff_buckets), geo_utils)
    alg.train()

if __name__ == "__main__":
    main()
