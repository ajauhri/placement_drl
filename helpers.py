from __future__ import division

import numpy as np
import logging

class TimeUtilities:
    def __init__(self, time_bin_width_secs):
        self.time_bin_width_secs = time_bin_width_secs
        self.hours_per_day = 24
        self.time_bin_bounds = []

    def set_bounds(self, D):
        """
        Compute boundaries of time bins based on time ranges in the data file
        """
        # to ensure the last upper bound is greater than the actual max. time
        bias = self.time_bin_width_secs
        self.time_bin_bounds = np.array(range(int(np.min(D[:,0])), 
            int(np.max(D[:,0])) + bias, 
            self.time_bin_width_secs))
    
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
            else 1 if not self._gte_lb(p[0], self.lat_grids[0]) \
                else len(self.lat_grids) - 1
        
        lng_cell = np.argmax(p[1] < self.lng_grids) \
            if self._within_boundary(
                p[1], self.lng_grids[0], self.lng_grids[-1]) \
            else 1 if not self._gte_lb(p[1], self.lng_grids[0]) \
                else len(self.lng_grids) - 1
        
        node = (lat_cell - 1) * len(self.lng_grids) + \
                (lng_cell - 1)

        return node, lat_cell - 1, lng_cell - 1


    def get_action_space(self, lat_idx, lng_idx):
        """
        Given a pair of lat. and lng. index, this function returns the boundaries
        of the possible actions which can be taken. The return is of dtype in the 
        half-open interval i.e. [start_lat_idx, end_lat_idx) and so fort for 
        longitudes.
        """
        actions = {}
        if lat_idx - 1 < 0:
            actions[1] = [0, lng_idx]
        else:
            actions[1] = [lat_idx - 1, lng_idx]

        if lat_idx + 1 >= len(self.lat_grids):
            actions[3] = [len(self.lat_grids) - 1, lng_idx]
        else:
            actions[3] = [lat_idx + 1, lng_idx]

        if lng_idx - 1 < 0:
            actions[0] = [lat_idx, 0]
        else:
            actions[0] = [lat_idx, lng_idx - 1]

        if lng_idx + 1 >= len(self.lng_grids):
            actions[2] = [lat_idx, len(self.lng_grids) - 1]
        else:
            actions[2] = [lat_idx, lng_idx + 1]

        for k,v in actions.iteritems():
            node = v[0] * len(self.lng_grids) + v[1] 
            actions[k] = node

        return actions

    def get_centroid(self, n):
        c2 = n % len(self.lng_grids) + 1
        c1 = int((n - c2 + 1) / len(self.lng_grids)) + 1
        return [(self.lat_grids[c1 - 1] + self.lat_grids[c1]) / 2, 
            (self.lng_grids[c2 - 1] + self.lng_grids[c2]) / 2]
