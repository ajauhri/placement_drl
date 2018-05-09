from __future__ import division
import numpy as np
import copy
from collections import Counter
import sys

class Sim:
    def __init__(self, X, n_lng_grids, time_utils, geo_utils, n_actions,
            dropoff_buckets, episode_duration=20, gamma=0.9, past_t=5):
        self.X = X
        self.rrs = {}
        self.req_sizes = {}
        self.dropoff_buckets = dropoff_buckets
        self.episode_duration = episode_duration
        self.gamma = gamma
        self.past_t = past_t
 
        self.n_actions = n_actions
        self.time_utils = time_utils
        self.geo_utils = geo_utils
        self.n_lng_grids = len(self.geo_utils.lng_grids)
        self.n_lat_grids = len(self.geo_utils.lat_grids)
        self.classes = (len(self.geo_utils.lat_grids) - 1) *\
            (len(self.geo_utils.lng_grids) - 1)
    
    def get_states(self):
        return np.eye(self.classes)[self.curr_states[:self.curr_index]].tolist()

    def get_pmr_states(self,t):
        t_i = t - self.start_t;
        pmr_state = self.pmr_states[t_i][:self.pmr_index[t_i]];
        return np.eye(self.classes)[pmr_state].tolist();
    
    def get_next_node(self, curr_node, a):
        lng_idx = int(curr_node % (self.n_lng_grids - 1))
        lat_idx = int(curr_node / (self.n_lng_grids - 1))
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
            if lat_idx + 1 >= self.n_lat_grids - 1:
                next_node = [self.n_lat_grids - 2, lng_idx]
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
            if lng_idx + 1 >= self.n_lng_grids - 1:
                next_node = [lat_idx, self.n_lng_grids - 2]
            else:
                next_node = [lat_idx, lng_idx + 1]

        return next_node[0] * (self.n_lng_grids - 1) + next_node[1]
    
    def sample_action_space(self):
        return np.random.randint(self.n_actions)

    def reset(self, t):
        self.pmr_index = [0] * self.episode_duration; 
        # technically should be maximum number of cars, not classes
        self.pmr_dropoffs = [[]] * self.episode_duration;
        self.pmr_states = [[]] * self.episode_duration;
        self.pmr_ids = [[]] * self.episode_duration;
        for i in range(self.episode_duration):
            self.pmr_dropoffs[i] = [-1] * self.classes;
            self.pmr_states[i] = [-1] * self.classes;
            self.pmr_ids[i] = [-1] * self.classes;
        self.start_t = t
        self.end_t = t + self.episode_duration
        self.curr_t = t
        th = self.time_utils.get_hour_of_day(t)
        self.car_id_counter = 0

        self.requests = self.rrs[self.start_t];
        self.curr_req_size = self.req_sizes[self.start_t];
        self.curr_req_index = [0] * (self.classes);

        self.curr_index = 0
        self.curr_ids = [-1] * (self.classes);
        self.curr_states = [-1] * (self.classes);
        self.curr_nodes = [-1] * (self.classes);
 
        for idx in self.dropoff_buckets[t]:
            dropoff_node, d_lat_idx, d_lon_idx = \
                    self.geo_utils.get_node(self.X[idx, 5:7])

            if (d_lat_idx >=0 and d_lon_idx >= 0):
                self.curr_nodes[self.curr_index] = dropoff_node
                self.curr_states[self.curr_index] = dropoff_node
                self.curr_ids[self.curr_index] = self.car_id_counter
                self.curr_index += 1;
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
        if self.curr_t+1 in self.req_sizes:
            self.curr_req_size = self.req_sizes[self.curr_t+1]
        
        next_index = 0;
        # technically should be maximum number of cars, not classes
        next_nodes= [-1] * (self.classes);
        next_states = [-1] * (self.classes);
        next_ids = [-1] * (self.classes);
        rewards = [0] * (self.classes);
        r_index = 0;

        #1 check curr dropoffs which can be matched
        for i in range(self.curr_index):
            matched, r, next_node = self._in_rrs(self.curr_nodes[i], 
                                           a_t[i],
                                           self.curr_ids[i],
                                           next_th)
            rewards[r_index] = r;
            r_index += 1;

            if (matched and r == 0):
                print("matched, ", car_id);
            
            if r == 0:
                car_id = self.curr_ids[i];
                next_nodes[next_index] = next_node
                next_states[next_index] = next_node
                next_ids[next_index] = self.curr_ids[i]
                next_index += 1;

        #2 check dropoffs from previous matched requets if can be matched further
        for i in range(self.pmr_index[self.curr_t- self.start_t]):
            matched, r, next_node = self._in_rrs(self.pmr_dropoffs[self.curr_t - self.start_t][i], 
                                           pmr_a_t[i],
                                           self.pmr_ids[self.curr_t - self.start_t][i],
                                           next_th)
            rewards[r_index] = r;
            r_index += 1;

            if (matched and r == 0):
                print("matched, ", car_id);
                        
            if r == 0:
                car_id = self.pmr_ids[self.curr_t - self.start_t][i]
                next_nodes[next_index] = next_node
                next_states[next_index] = next_node
                next_ids[next_index] = self.pmr_ids[self.curr_t - self.start_t][i]
                next_index += 1;
#                self.car_id_counter += 1;

        #3 add dropoff vehicles from before beginning of episode
        if self.curr_t+1 in self.dropoff_buckets:
            for idx in self.dropoff_buckets[self.curr_t+1]:
                dropoff_node, d_lat_idx, d_lon_idx = \
                        self.geo_utils.get_node(self.X[idx, 5:7])

                if (d_lat_idx >= 0 and d_lon_idx >= 0):
                    p_t = self.time_utils.get_bucket(self.X[idx, 1])
                    if p_t < self.start_t:
                        next_nodes[next_index] = dropoff_node
                        next_states[next_index] = dropoff_node
                        next_ids[next_index] = self.car_id_counter
                        next_index += 1;
                        self.car_id_counter += 1        

        self.curr_index = next_index
        self.curr_states = next_states[:next_index]
        self.curr_ids = next_ids[:next_index]
        self.curr_nodes = next_nodes[:next_index]
        self.curr_t += 1
        
        return rewards[:r_index];

    def _in_rrs(self, dropoff_node, a, car_id, next_th):
        matched = False
        placmt_node = self.get_next_node(dropoff_node, a)
        
        if self.curr_req_size[placmt_node] > self.curr_req_index[placmt_node]:
            req = self.requests[placmt_node][self.curr_req_index[placmt_node]];
            self.curr_req_index[placmt_node] += 1;
            matched = True;
            
            dropoff_node = req[0];
            drive_t = req[1];
            r_t = req[2];
            new_dropoff_t = (self.curr_t + 1) + drive_t

            if new_dropoff_t < self.end_t and dropoff_node >= 0:
                time_index = new_dropoff_t - self.start_t;
#                n_steps = self.geo_utils.get_num_steps(placmt_node, self.n_lng_grids, dropoff_node)
#                print(drive_t,n_steps)
                # maintain all previously matched rides to be 
                # considered for future cars
                self.pmr_dropoffs[time_index][self.pmr_index[time_index]] = dropoff_node;
                self.pmr_states[time_index][self.pmr_index[time_index]] = dropoff_node; 
                self.pmr_ids[time_index][self.pmr_index[time_index]] = self.car_id_counter;#car_id;
                self.pmr_index[time_index] += 1;
                self.car_id_counter += 1;

            r = self.gamma ** ((self.curr_t + 1) - r_t)
            return matched, r, placmt_node

        return matched, 0, placmt_node
