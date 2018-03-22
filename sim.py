from __future__ import division
import numpy as np

class Sim:
    def __init__(self, n_lng_grids, time_utils, geo_utils, n_actions):
        self.pickup_maps = {}
        self.dropoff_maps = {}
        self.test_pickup_maps = {}
        self.test_dropoff_maps = {}
 
        self.n_lng_grids = n_lng_grids
        self.n_actions = n_actions
        self.time_utils = time_utils
        self.geo_utils = geo_utils

    def get_random_node(self, ts):
        if ts in self.dropoff_maps:
            return np.random.choice(self.dropoff_maps[ts].keys())
        else:
            return -1

    def get_random_nodes(self, ts, size=100):
        if ts not in self.dropoff_maps:
            return []
        if len(self.dropoff_maps[ts]) < size:
            return np.random.choice(self.dropoff_maps[ts].keys(), 
                    len(self.dropoff_maps[ts]), 
                    replace=False)
        else:
            return np.random.choice(self.dropoff_maps[ts].keys(), 
                    size, 
                    replace=False)
    
    def get_state_rep(self, node, ts):
        th = self.time_utils.get_hour_of_day(ts)
        return self.geo_utils.get_centroid(node) + [th]

    def add_maps(self, ts, dropoff_map, pickup_map, is_test=False):
        if not is_test:
            if ts in self.dropoff_maps:
                self.dropoff_maps[ts] += dropoff_map
            else:
                self.dropoff_maps[ts] = dropoff_map
            
            if ts+1 in self.pickup_maps:
                self.pickup_maps[ts+1] += pickup_map
            else:
                self.pickup_maps[ts+1] = pickup_map
        
        elif is_test:
            if ts in self.test_dropoff_maps:
                self.test_dropoff_maps[ts] += dropoff_map
            else:
                self.test_dropoff_maps[ts] = dropoff_map
            
            if ts+1 in self.test_pickup_maps:
                self.test_pickup_maps[ts+1] += pickup_map
            else:
                self.test_pickup_maps[ts+1] = pickup_map
     
    def get_next_node(self, curr_node, a):
        lng_idx = int(curr_node % self.n_lng_grids)
        lat_idx = int(curr_node / self.n_lng_grids)
        next_node = -1
        
        #NOP
        if a == 4:
            return curr_node
        
        #down
        elif a == 1:
            if lat_idx - 1 < 0:
                next_node = [0, lng_idx]
            else:
                next_node = [lat_idx - 1, lng_idx]
        #up 
        elif a == 3:
            if lat_idx + 1 >= self.n_lng_grids:
                next_node = [self.n_lng_grids - 1, lng_idx]
            else:
                next_node = [lat_idx + 1, lng_idx]
        #left
        elif a == 0:
            if lng_idx - 1 < 0:
                next_node = [lat_idx, 0]
            else:
                next_node = [lat_idx, lng_idx - 1]
        #right
        elif a == 2:
            if lng_idx + 1 >= self.n_lng_grids:
                next_node = [lat_idx, self.n_lng_grids - 1]
            else:
                next_node = [lat_idx, lng_idx + 1]
        
        return next_node[0] * self.n_lng_grids + next_node[1]
    
    def sample_action_space(self):
        return np.random.randint(self.n_actions)

    def step(self, src, ts, a):
        """
        s: state depicting the centroid of the dropoff grid, hour of day
        a: left (0), down (1), right (2), up (3), NOP, (4)
        """
        dst = self.get_next_node(src, a)
        if self.dropoff_maps[ts][src] > 0 and \
                self.pickup_maps[ts+1][dst] > 0:
                    #TODO use a different reward; may be cumulative?
                    return (dst, self.pickup_maps[ts+1][dst])
        return (dst, 0)
    

    def start_test(self):
        self.test_steps = []
    
    def reset_test(self):
        for i in self.test_steps:
            self.test_pickup_maps[i[1]][i[0]] += 1

    def step_test(self, src, ts, a):
        """
        s: state depicting the centroid of the dropoff grid, hour of day
        a: left (0), down (1), right (2), up (3), NOP, (4)
        """
        dst = self.get_next_node(src, a)
        if self.test_dropoff_maps[ts][src] > 0 and \
                self.test_pickup_maps[ts+1][dst] > 0:
                    self.test_pickup_maps[ts+1][dst] -= 1
                    self.test_steps.append([dst, ts+1])
                    return (dst, 1)
        return (dst, 0)
