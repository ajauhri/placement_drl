from __future__ import division

import numpy as np
import logging
import copy
try:
    import cPickle
except ImportError:
    import _pickle as cPickle

def load_create_pickle(sim, train_time_utils, geo_utils, X,
        train_tb_starts, test_tb_starts):
    """
    """

    # segregate data based on time-steps
    train_request_buckets = train_time_utils.get_buckets(X, 0)
    train_dropoff_buckets = train_time_utils.get_buckets(X, 4)

    pre_sim_pickups = {}
    pre_load = 5
    for start in train_tb_starts + test_tb_starts:
        req_count = [0] * sim.num_cells
        req_arr = [[] for x in range(sim.num_cells)]
        
        """
        load requests for the duration of the episode, and few from before.
        requests are only added if pickup is after start irrespective or request
        time.
        """
        for r_t in range(start - pre_load, start + sim.episode_duration + 1):
            if r_t in train_request_buckets:
                c = []
                for i in train_request_buckets[r_t]:
                    dropoff_node, _, _ = geo_utils.get_node(X[i, 5:7])
                    pickup_node, _, _ = geo_utils.get_node(X[i, 2:4])
                    c.append([X[i, 1], X[i, 4], pickup_node, dropoff_node, 
                        max(r_t, start), -1])
        
                c = np.array(c)
                
                # check pickup-node should be non-negative
                c = c[c[:, 2] >= 0, :]
                if len(c) > 0:
                    # find discrete time steps for pickups
                    p_ts = train_time_utils.get_xx_buckets(c, 0, s_t=r_t)

                    # filter out pickup times should >= simulation start time
                    valid_reqs = np.where(p_ts >= start)[0]
                    if len(valid_reqs) > 0:
                        c = c[valid_reqs, :]
                        p_ts = p_ts[valid_reqs]

                        # find discrete time steps for drop-offs
                        d_ts = train_time_utils.get_xx_buckets(c, 1, s_t=r_t)

                        # compute travel_time
                        c[:, -1] = d_ts - p_ts
                        print(np.sum(c[:, -1] < 0))
                        c = c.astype(int)
                        for r in c:
                            pickup_node = int(r[2])
                            # list of dropoff_node, req_time_bin, drive_time_bin
                            req_arr[pickup_node].append([r[3], r[4], r[5]])
                            req_count[pickup_node] += 1
                        sim.n_reqs[r_t] = req_count[:]
    
        logging.info("Loaded requests for time bins %d to  %d" % 
                (start - pre_load, start + sim.episode_duration))
        
        # add drop-offs for requests picked up before simulation starts
        c = []
        for d_t in range(start, start + sim.episode_duration):
            if d_t in train_dropoff_buckets:
                if d_t not in pre_sim_pickups:
                    pre_sim_pickups[d_t] = []
                for i in train_dropoff_buckets[d_t]:
                    dropoff_node, _, _ = geo_utils.get_node(X[i, 5:7])
                    c.append([dropoff_node, d_t, X[i, 1]])
        
        c = np.array(c)

        # check dropoff-node should be non-negative
        c = c[c[:, 0] >= 0, :]
        p_ts = train_time_utils.get_xx_buckets(c, 2, 
                start + sim.episode_duration)
        c = c.astype(int)
        for i in range(len(c)):
            # pickup time should be before start of simulation to avoid any
            # double counting in the requests generated in the simulator
            if p_ts[i] < start:
                pre_sim_pickups[c[i, 1]].append(c[i, 0])

        logging.info("Loaded dropoffs for all time bins from %d to %d" %
                (start, start + sim.episode_duration))
        sim.rrs[start] = req_arr
    
    with open(r"rrs.pickle", "wb") as out_file:
        cPickle.dump(sim.rrs, out_file)
        cPickle.dump(sim.n_reqs, out_file)
        cPickle.dump(pre_sim_pickups, out_file)


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
        self.time_bin_bounds = np.array(range(int(np.min(D[:, 0])), 
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
 
    def get_xx_buckets(self, D, ind, s_t=0, e_t=-1):
        t = np.zeros((len(D)))
        if e_t == -1 or e_t == len(self.time_bin_bounds):
            e_t = len(self.time_bin_bounds) - 1
        for i in range(s_t, e_t):
            idx = np.where(np.logical_and(
                D[:, ind] >= self.time_bin_bounds[i],
                D[:, ind] < self.time_bin_bounds[i+1]))[0]
            if len(idx):
                t[idx] = i
        return t

    def get_buckets(self, D, ind):
        time_bins = {}
        for i in range(len(self.time_bin_bounds) - 1):
            time_bins[i] = np.where(np.logical_and(
                D[:, ind] >= self.time_bin_bounds[i],
                D[:, ind] < self.time_bin_bounds[i+1]))[0]
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
        
        
