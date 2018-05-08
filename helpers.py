from __future__ import division

import numpy as np
import logging

class rr:
    def __init__(self, id, r_t, p_t, d_t, dn, dlat, dlng, pn, plat, plng):
        self.id = id
        self.r_t = r_t
        self.p_t = p_t
        self.d_t = d_t
        self.dn = dn
        self.dlat = dlat
        self.dlng = dlng
        self.pn = pn
        self.plat = plat
        self.plng = plat
        self.picked = False

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

        
    def get_centroid(self, n):
        c2 = n % len(self.lng_grids) + 1
        c1 = int((n - c2 + 1) / len(self.lng_grids)) + 1
        print(c1, c2)
        print('t', len(self.lng_grids), len(self.lat_grids))
        return [(self.lat_grids[c1 - 1] + self.lat_grids[c1]) / 2, 
            (self.lng_grids[c2 - 1] + self.lng_grids[c2]) / 2]

    def get_centroid_v2(self, n, n_lng_grids):
        c2 = np.array(n % (n_lng_grids - 1)).astype(int).tolist();
        c1 = np.array(n / (n_lng_grids - 1)).astype(int).tolist();
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
        
        
