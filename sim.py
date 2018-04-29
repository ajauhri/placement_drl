from __future__ import division
import numpy as np

class Sim:
    def __init__(self, X, n_lng_grids, time_utils, geo_utils, n_actions,
            dropoff_buckets, episode_duration=20, gamma=0.9, past_t=5):
        self.X = X
        self.rrs = {}
        self.dropoff_buckets = dropoff_buckets
        self.episode_duration = episode_duration
        self.gamma = gamma
        self.past_t = past_t
 
        self.n_actions = n_actions
        self.time_utils = time_utils
        self.geo_utils = geo_utils
        self.n_lng_grids = len(self.geo_utils.lng_grids)
        self.n_lat_grids = len(self.geo_utils.lat_grids)
        self.classes = len(self.geo_utils.lat_grids) *\
                len(self.geo_utils.lng_grids)
    
    def get_state(self, node, th):
        return np.eye(self.classes)[node].tolist() + [th]
    
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
            if lat_idx + 1 >= self.n_lat_grids:
                next_node = [self.n_lat_grids - 1, lng_idx]
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
        self.pmr_dropoffs = {}
        self.pmr_states = {}
        self.pmr_ids = {}
        self.start_t = t
        self.end_t = t + self.episode_duration
        self.curr_t = t
        th = self.time_utils.get_hour_of_day(t)
        self.car_id_counter = 0
        self.requests = dict(self.rrs)
        
        self.curr_ids = []
        self.curr_states = []
        self.curr_nodes = []
        for idx in self.dropoff_buckets[t]:
            dropoff_node, d_lat_idx, d_lon_idx = \
                    self.geo_utils.get_node(self.X[idx, 5:7])

            if (d_lat_idx > 0 and d_lon_idx > 0):

                #centroid = self.geo_utils.get_centroid(dropoff_node) 
                self.curr_states.append(self.get_state(dropoff_node, th))
                #self.curr_states.append(centroid + [th])
                self.curr_nodes.append(dropoff_node)
                self.curr_ids.append(self.car_id_counter)
                self.car_id_counter += 1
        
    def step(self, a_t, pmr_a_t):
        """
        s: state depicting the centroid of the dropoff grid, hour of day
        a: left (0), down (1), right (2), up (3), NOP, (4)
        """
        if self.curr_t > self.end_t:
            return False

        th = self.time_utils.get_hour_of_day(self.curr_t)
        next_th = self.time_utils.get_hour_of_day(self.curr_t + 1)
        
        next_nodes = []
        next_states = []
        next_ids = []
        rewards = []
        
        #1 check curr dropoffs which can be matched
        for i in range(len(self.curr_states)):
            r, next_node = self._in_rrs(self.curr_nodes[i], 
                    a_t[i],
                    self.curr_ids[i],
                    next_th)
            rewards.append(r)
            
            if r == 0:
                next_nodes.append(next_node)
                next_states.append(self.get_state(next_node, next_th))
                #next_states.append(next_centroid + [next_th])
                next_ids.append(self.curr_ids[i])
        
        #2 check dropoffs from previous matched requets if can be matched further
        if self.curr_t in self.pmr_states:
            for i in range(len(self.pmr_states[self.curr_t])):
                r, next_node = \
                        self._in_rrs(self.pmr_dropoffs[self.curr_t][i], 
                                pmr_a_t[i],
                                self.pmr_ids[self.curr_t][i],
                                next_th)
                rewards.append(r)
                        
                if r == 0:
                    next_nodes.append(next_node)
                    next_states.append(self.get_state(next_node, next_th))
                    #next_states.append(next_centroid + [next_th])
                    next_ids.append(self.pmr_ids[self.curr_t][i])
        
        #3 add dropoff vehicles from before beginning of episode
        if self.curr_t+1 in self.dropoff_buckets:
            for idx in self.dropoff_buckets[self.curr_t+1]:
                dropoff_node, d_lat_idx, d_lon_idx = \
                        self.geo_utils.get_node(self.X[idx, 5:7])

                if (d_lat_idx > 0 and d_lon_idx > 0):

                    p_t = self.time_utils.get_bucket(self.X[idx, 1])
                    if p_t <= self.start_t:
                        self.car_id_counter += 1
                        #centroid = self.geo_utils.get_centroid(dropoff_node)
                        next_nodes.append(dropoff_node)
                        next_states.append(self.get_state(dropoff_node, next_th))
                        #next_states.append(centroid + [next_th])
                        next_ids.append(self.car_id_counter)
            
        self.curr_states = next_states
        self.curr_ids = next_ids
        self.curr_nodes = next_nodes
        self.curr_t += 1
        
        return rewards

    def _in_rrs(self, dropoff_node, a, car_id, next_th):
        matched = False
        placmt_node = self.get_next_node(dropoff_node, a)
        #placmt_centroid = self.geo_utils.get_centroid(placmt_node)
        
        if placmt_node in self.requests:
            for i in range(len(self.requests[placmt_node])):
                if (not self.requests[placmt_node][i].picked) and (\
                        (self.requests[placmt_node][i].r_t < (self.curr_t + 1) \
                        and 
                        self.requests[placmt_node][i].r_t > (self.start_t - \
                                self.past_t) and
                        self.requests[placmt_node][i].p_t > (self.curr_t + 1))
                     or
                     self.requests[placmt_node][i].r_t == self.curr_t + 1):
                        self.requests[placmt_node][i].picked = True
                        
                        pickup_t = self.requests[placmt_node][i].p_t
                        dropoff_t = self.requests[placmt_node][i].d_t
                        drive_t = dropoff_t - pickup_t
                        self.requests[placmt_node][i].p_t = self.curr_t + 1
                        
                        new_dropoff_t = (self.curr_t + 1) + drive_t
                        if new_dropoff_t < self.end_t:
                            if new_dropoff_t not in self.pmr_dropoffs:
                                self.pmr_dropoffs[new_dropoff_t] = []
                                self.pmr_states[new_dropoff_t] = []
                                self.pmr_ids[new_dropoff_t] = []

                            #dropoff_centroid = self.geo_utils.get_centroid(
                            #        self.requests[placmt_node][i].dn)
                            
                            # maintain all previously matched rides to be 
                            # considered for future cars
                            self.pmr_dropoffs[new_dropoff_t].append(
                                    self.requests[placmt_node][i].dn)
                            #self.pmr_states[new_dropoff_t].append(
                            #        dropoff_centroid + [next_th])
                            self.pmr_states[new_dropoff_t].append(\
                                    self.get_state(\
                                    self.requests[placmt_node][i].dn, next_th))
                            self.pmr_ids[new_dropoff_t].append(car_id)
                        
                        matched = True
                        r = self.gamma ** ((self.curr_t + 1) - \
                                self.requests[placmt_node][i].r_t)
                        return r, placmt_node
        if not matched:
            return 0, placmt_node
