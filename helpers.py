from __future__ import division

import numpy as np
import logging
import copy
try:
    import cPickle
except ImportError:
    import _pickle as cPickle

try:
    import cPickle
except ImportError:
    import _pickle as cPickle

def load_create_pickle(sim, train_time_utils, geo_utils, X,
        train_tb_starts, test_tb_starts):

    # segregate data based on time-steps
    train_request_buckets = train_time_utils.get_buckets(X, 0)
    train_dropoff_buckets = train_time_utils.get_buckets(X, 4)

    post_start_cars = {}
    pre_load = 5
    for start in train_tb_starts + test_tb_starts:
        req_count = [0] * sim.num_cells
        req_arr = [[] for x in range(sim.num_cells)]
        
        # load requests picked up after the beginning of the simulation
        for r_t in range(start-pre_load, start + sim.episode_duration):
            if r_t in train_request_buckets:
                for i in train_request_buckets[r_t]:
                    dropoff_node, d_lat_idx, d_lon_idx = \
                            geo_utils.get_node(X[i, 5:7])
                    pickup_node, p_lat_idx, p_lon_idx = \
                            geo_utils.get_node(X[i, 2:4])

                    if (pickup_node >= 0):
                        d_t = train_time_utils.get_bucket(X[i, 4])
                        p_t = train_time_utils.get_bucket(X[i, 1])
                        travel_t = d_t - p_t
                        
                        if (p_t >= start):
                            req_arr[pickup_node].append([dropoff_node,
                                travel_t, max(r_t, start)])
                            req_count[pickup_node] += 1
            logging.info("Loaded Requests for time bin %d, hour of day %d" \
                    % (r_t, train_time_utils.get_hour_of_day(r_t)))
            sim.req_sizes[r_t] = copy.deepcopy(req_count)
        
        # add drop-offs for requests picked up before simulation starts
        for d_t in range(start, start + sim.episode_duration):
            if (d_t not in post_start_cars):
                post_start_cars[d_t] = []
            if d_t in train_dropoff_buckets:
                for i in train_dropoff_buckets[d_t]:
                    dropoff_node, d_lat_idx, d_lon_idx = \
                        geo_utils.get_node(X[i, 5:7])
                    if (dropoff_node >= 0):
                        r_t = train_time_utils.get_bucket(X[i, 0]);
                        if (r_t < (start - pre_load) or p_t < start):
                            post_start_cars[d_t].append(dropoff_node);
            logging.info("Loaded Dropoffs for time bin %d, hour of day %d" \
                    % (d_t, train_time_utils.get_hour_of_day(d_t)))
        sim.rrs[start] = req_arr
    
    with open(r"rrs.pickle", "wb") as out_file:
        cPickle.dump(sim.rrs, out_file)
        cPickle.dump(sim.req_sizes, out_file)
        cPickle.dump(post_start_cars, out_file)


class TimeUtilities:
    def __init__(self, time_bin_width_secs):
        self.time_bin_width_secs = time_bin_width_secs
        self.time_bin_bounds = []
        self.n_time_bins_per_day = int((24*60*60) / time_bin_width_secs)
        self.n_time_bins_per_hour = int((60*60) / time_bin_width_secs)

    def set_bounds(self, D):
        """
        Compute boundaries of time bins based on time ranges in the data file
        """
        # to ensure the last upper bound is greater than the actual max. time
        bias = self.time_bin_width_secs
        self.time_bin_bounds = np.array(range(int(np.min(D[:,0])), 
            int(np.max(D[:, 0])) + bias, 
            self.time_bin_width_secs))
    
    def get_hour_of_day(self, ts):
        return int( (ts % self.n_time_bins_per_day) / self.n_time_bins_per_hour)
    
    def get_bucket(self, time):
        for t in range(len(self.time_bin_bounds) - 1):
            idx = np.where(np.logical_and(
                time >= self.time_bin_bounds[t],
                time < self.time_bin_bounds[t+1]))[0]
            if len(idx):
                return t

    def get_buckets(self, D, ind):
        time_bins = {}
        for i in range(len(self.time_bin_bounds) - 1):
            time_bins[i] = np.where(np.logical_and(
                D[:, ind] >= self.time_bin_bounds[i],
                D[:, ind] < self.time_bin_bounds[i+1]))[0]
        logging.info('Loaded {0} time buckets'.format(
            len(time_bins)))
        return time_bins

class GeoUtilities:

    EARTH_RADIUS = 6371.009;

    def __init__(self, lat_min, lat_max, 
            lng_min, lng_max, cell_length_meters):
        self.cell_length_meters = cell_length_meters
        self.latitude_meters = 111.2 * 1000
        self.longitude_meters = 111 * 1000
        self.earth_radius_km = 6371.009 
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lng_min = lng_min
        self.lng_max = lng_max
        self.lat_grids = []
        self.lng_grids = []

    def _gte_lb(self, x, lb):
        return x >= lb
        
    def _lte_ub(self, x, ub):
        return x <= ub

    def _within_boundary(self, x, lb, ub):
        return self._gte_lb(x, lb) and self._lte_ub(x, ub)
    
    def set_grids(self):
        """
        Divides a spatial area into equally sized cells.
        """
        lat_steps = int(abs(self.lat_max - self.lat_min) 
            * self.latitude_meters / self.cell_length_meters)
        self.lat_grids = np.linspace(self.lat_min, self.lat_max, lat_steps)
        
        lng_steps = int(abs(self.lng_max - self.lng_min) 
            * self.longitude_meters / self.cell_length_meters)
        self.lng_grids = np.linspace(self.lng_min, self.lng_max, lng_steps)

    def get_node(self, p):
        lat_cell = np.argmax(p[0] < self.lat_grids) \
            if self._within_boundary(
                p[0], self.lat_grids[0], self.lat_grids[-1]) \
            else -1
        
        lng_cell = np.argmax(p[1] < self.lng_grids) \
            if self._within_boundary(
                p[1], self.lng_grids[0], self.lng_grids[-1]) \
            else -1
        
        node = (lat_cell - 1) * (len(self.lng_grids) - 1) + \
                (lng_cell - 1)
        
        if lat_cell < 0 or lng_cell < 0:
            return -1, -1, -1
        
        return node, lat_cell - 1, lng_cell - 1

        
    def get_centroid_v2(self, n, n_lng_grids):
        c2 = np.array(n % n_lng_grids).astype(int).tolist();
        c1 = np.array(n / n_lng_grids).astype(int).tolist();
        return [(self.lat_grids[c1+1] + self.lat_grids[c1]) / 2, 
            (self.lng_grids[c2+1] + self.lng_grids[c2]) / 2]


    def orthodromic_dist1d(a, b):
        lat1, lng1 = math.radians(a[:,0]), math.radians(a[:,1])
        lat2, lng2 = math.radians(b[0]), math.radians(b[1])
        
        sin_lat1, cos_lat1 = math.sin(lat1), math.cos(lat1)
        sin_lat2, cos_lat2 = math.sin(lat2), math.cos(lat2)

        delta_lng = lng2 - lng1
        cos_delta_lng, sin_delta_lng = math.cos(delta_lng), math.sin(delta_lng)

        d = math.atan2(math.sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                                 (cos_lat1 * sin_lat2 -
                                  sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
                       sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)
        return EARTH_RADIUS * d * 1000

    def get_num_steps(self, n, n_lng_grids, p_n):
        lng_idx = int(n % n_lng_grids);
        lat_idx = int(n / n_lng_grids);

        p_lng_idx = int(p_n % n_lng_grids);
        p_lat_idx = int(p_n / n_lng_grids);

        return np.abs(p_lng_idx - lng_idx) + np.abs(p_lat_idx - lat_idx);
        
        
