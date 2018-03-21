from __future__ import division
import numpy as np

class Sim:
    def __init__(self, n_lng_grids, time_utils):
        self.pickup_maps = {}
        self.dropoff_maps = {}
        self.n_lng_grids = n_lng_grids
        self.n_actions = 5 #[0,1,2,3,4]
        self.time_utils = time_utils

    def _update_map(self, src_t, src, dst_t, dst):
        self.dropoff_maps[src_t][src] -= 1
        self.pickup_maps[dst_t][dst] -= 1
 
    def get_random_node(self, ts):
        src_th = self.time_utils.get_hour_of_day(ts)
        return np.random.choice(self.dropoff_maps[src_th].keys())

    def add_maps(self, t, dropoff_map, pickup_map):
        dropoff_t = self.time_utils.get_hour_of_day(t)
        pickup_t = self.time_utils.get_hour_of_day(t+1)

        if dropoff_t in self.dropoff_maps:
            self.dropoff_maps[dropoff_t] += dropoff_map
        else:
            self.dropoff_maps[dropoff_t] = dropoff_map
        
        if pickup_t in self.pickup_maps:
            self.pickup_maps[pickup_t] += pickup_map
        else:
            self.pickup_maps[pickup_t] = pickup_map
    
    def get_next_node(self, curr_node, a):
        lng_idx = int(curr_node % self.n_lng_grids)
        lat_idx = int(curr_node / self.n_lng_grids)
        next_node = -1
        
        if a == 4:
            return curr_node

        elif a == 1:
            if lat_idx - 1 < 0:
                next_node = [0, lng_idx]
            else:
                next_node = [lat_idx - 1, lng_idx]
        
        elif a == 3:
            if lat_idx + 1 >= self.n_lng_grids:
                next_node = [self.n_lng_grids - 1, lng_idx]
            else:
                next_node = [lat_idx + 1, lng_idx]

        elif a == 0:
            if lng_idx - 1 < 0:
                next_node = [lat_idx, 0]
            else:
                next_node = [lat_idx, lng_idx - 1]

        elif a == 2:
            if lng_idx + 1 >= self.n_lng_grids:
                next_node = [lat_idx, self.n_lng_grids - 1]
            else:
                next_node = [lat_idx, lng_idx + 1]
        return next_node[0] * self.n_lng_grids + next_node[1]
    
    def sample_action_space(self):
        return np.random.randint(self.n_actions)

    def step(self, src, ts, a, update=False):
        """
        s: state depicting the centroid of the dropoff grid, hour of day
        a: left (0), down (1), right (2), up (3), NOP, (4)
        """
        dropoff_th = self.time_utils.get_hour_of_day(ts)
        pickup_th = self.time_utils.get_hour_of_day(ts+1)
        dst = self.get_next_node(src, a)
        if self.dropoff_maps[dropoff_th][src] > 0 and \
                self.pickup_maps[pickup_th][dst] > 0:
                    if update:
                        self._update_map(dropoff_th, src, pickup_th, dst)
                    #TODO use a different reward; may be cumulative?
                    return (dst, pickup_th, 1, dropoff_th)
        return (dst, pickup_th, 0, dropoff_th)
