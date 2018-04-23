from __future__ import division
import numpy as np

class Sim:
    def __init__(self, n_lng_grids, time_utils, geo_utils, n_actions,
            dropoff_buckets):
        self.pickup_maps = {}
        self.rrs = {}
        self.test_pickup_maps = {}
        self.test_dropoff_maps = {}
        self.dropoff_buckets = dropoff_buckets
 
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
    
    """ 
    def add_maps(self, ts, pickup_map, is_test=False):
        if not is_test:
            self.pickup_maps[ts+1] = pickup_map
        
        elif is_test:
            self.test_dropoff_maps[ts] = dropoff_map
            self.test_pickup_maps[ts+1] = pickup_map
    """ 

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
    
    def reset(self, t):
        self.start_t = t
        self.curr_t = t
        th = self.time_utils.get_hour_of_day(t)
        
        states = []
        nodes = []
        for idx in self.dropoff_buckets[t]:
            dropoff_node, d_lat_idx, d_lon_idx = \
                    geo_utils.get_node(X[idx, 5:7])
            centroid = self.geo_utils.get_centroid(dropoff_node) 
            states.append(centroid + [th])
            nodes.append(dropoff_node)
        
        self.curr_states = states
        self.curr_nodes = nodes
    
    def step(self, a):
        """
        s: state depicting the centroid of the dropoff grid, hour of day
        a: left (0), down (1), right (2), up (3), NOP, (4)
        """
        th = self.time_utils.get_hour_of_day(self.curr_t)
        next_th = self.time_utils.get_hour_of_day(self.curr_t + 1)
        next_nodes = []
        next_states = []
        rewards = []
        
        #1 check curr dropoffs which can be matched
        for i in range(len(self.curr_states)):
            r, next_node, next_centroid = self._in_rrs(self.curr_nodes[i], a)
            rewards.append(r)
            
            if r == 0:
                #next_dropoff_idxs.append(self.curr_idxs[i])
                next_nodes.append(next_node)
                next_centroid.append(next_centroid)
            
                next_states.append(next_centroid + [next_th])
        
        #2 check dropoffs from previous matched requets if can be matched further
        if self.curr_t in self.matched_dropoffs:
            for i in range(len(self.matched_dropoffs)):
                r, next_node, next_centroid = self._in_rrs(self.curr_nodes[i], a)
                rewards.append(r)
                        
                if r == 0:
                    #next_dropoff_idxs.append(self.curr_idxs[i])
                    next_nodes.append(next_node)
                    next_states.append(next_centroid + [next_th])
        
        # dropoffs from before beginning of episode
        if self.curr_t+1 in self.dropoff_buckets:
            for idx in self.dropoff_buckets[self.curr_t+1]:
                dropoff_node, d_lat_idx, d_lon_idx = \
                        geo_utils.get_node(X[idx, 5:7])

                p_t = train_time_utils.get_bucket(X[idx, 1])
                if p_t <= self.start_t:
                    #next_nodes.append(dropoff_node)
                    centroid = self.geo_utils.get_centroid(dropoff_node) 
                    next_states.append(centroid + [next_th])
        
        self.curr_states = next_states
        self.curr_t += 1

     def _in_rrs(self, dropoff_node, a):
        matched = False
        placmt_node = self.get_next_node(dropoff_node, a)
        placmt_centroid = self.geo_utils.get_centroid(placmt_node)
        
        if placmt_node in self.rrs:
            for i in range(len(self.rrs[placmt_node])):
                if (not self.rrs[placmt_node][i].picked) and
                    ((self.rrs[placmt_node][i].r_t < (self.curr_t + 1) and 
                        self.rrs[dist][i].p_t > (self.curr_t + 1))
                     or
                     self.rrs[placmt_node][i].r_t == self.curr_t + 1):
                        self.rrs[placmt_node][i].picked = True
                        
                        pickup_t = self.rrs[placmt_node][i].p_t
                        dropoff_t = self.rrs[placmt_node][i].d_t
                        drive_t = dropoff_t - pickup_t
                        self.rrs[placmt_node][i].p_t = self.curr_t
                        
                        new_dropoff_t = curr_t + drive_t
                        if new_dropoff_t <= self.end_t:
                            if new_dropoff_t not in self.matched_dropoffs:
                                self.matched_dropoffs = []
                            self.matched_dropoffs[new_dropoff_t].append(
                                    self.rrs[placmt_node][i].dn)
                        
                        matched = True
                        r = self.gamma ** ((self.curr_t + 1) - \
                                self.rrs[placmt_node][i].r_t)
                        return r, plcmt_node, placmt_centroid
        if not matched:
            return 0, plcmt_node, placmt_centroid


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