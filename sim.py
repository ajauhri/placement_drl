from __future__ import division
import numpy as np
import copy
from collections import Counter
import sys
import time;

class Sim:
    def __init__(self, X, n_lng_grids, time_utils, geo_utils, n_actions,
            dropoff_buckets, episode_duration=20, gamma=0.9, past_t=5):
        self.X = X
        self.rrs = {}
        self.req_sizes = {}
        self.post_start_cars = {}
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

        self.next_index = 0;
        # technically should be maximum number of cars, not classes
        self.next_nodes= [-1] * (self.classes);
        self.next_states = [-1] * (self.classes);
        self.next_ids = [-1] * (self.classes);
        self.rewards = [0] * (self.classes);
        self.r_index = 0;
        
        self.curr_nodes = self.post_start_cars[self.start_t];
        self.curr_states = self.post_start_cars[self.start_t];
        self.curr_index = len(self.curr_nodes) + 1;
        self.curr_ids = range(self.curr_index);
        self.car_id_counter = self.curr_index;
        
    def step(self, a_t, pmr_a_t):
        """
        s: state depicting the centroid of the dropoff grid, hour of day
        a: left (0), down (1), right (2), up (3), NOP, (4)
        """
        strt = time.time()
        if self.curr_t > self.end_t:
            return False

        th = self.time_utils.get_hour_of_day(self.curr_t)
        next_th = self.time_utils.get_hour_of_day(self.curr_t + 1)
        if self.curr_t+1 in self.req_sizes:
            self.curr_req_size = self.req_sizes[self.curr_t+1]
        
        self.next_index = 0;
        # technically should be maximum number of cars, not classes
        self.next_nodes= [-1] * (self.classes);
        self.next_states = [-1] * (self.classes);
        self.next_ids = [-1] * (self.classes);
        self.rewards = [0] * (self.classes);
        self.r_index = 0;
        end = time.time()
        print("step - setup")
        print(end-strt)

        strt = time.time()

        #1 check curr dropoffs which can be matched
        for i in range(self.curr_index):
            matched, r, next_node = self._in_rrs(self.curr_nodes[i], 
                                           a_t[i],
                                           self.curr_ids[i],
                                           next_th)
            self.rewards[self.r_index] = r;
            self.r_index += 1;

            if (matched and r == 0):
                print("matched, ", car_id);
            
            if r == 0:
                car_id = self.curr_ids[i];
                self.next_nodes[self.next_index] = next_node
                self.next_states[self.next_index] = next_node
                self.next_ids[self.next_index] = self.curr_ids[i]
                self.next_index += 1;

        end = time.time()
        print("step - 1")
        print(end-strt)

        strt = time.time();

        #2 check dropoffs from previous matched requets if can be matched further
        for i in range(self.pmr_index[self.curr_t- self.start_t]):
            matched, r, next_node = self._in_rrs(self.pmr_dropoffs[self.curr_t - self.start_t][i], 
                                           pmr_a_t[i],
                                           self.pmr_ids[self.curr_t - self.start_t][i],
                                           next_th)
            self.rewards[self.r_index] = r;
            self.r_index += 1;

            if (matched and r == 0):
                print("matched, ", car_id);
                        
            if r == 0:
                car_id = self.pmr_ids[self.curr_t - self.start_t][i]
                self.next_nodes[self.next_index] = next_node
                self.next_states[self.next_index] = next_node
                self.next_ids[self.next_index] = self.pmr_ids[self.curr_t - self.start_t][i]
                self.next_index += 1;
#                self.car_id_counter += 1;

        end = time.time()
        print("step - 2")
        print(end-strt)

        strt = time.time();

        #3 add dropoff vehicles from before beginning of episode
        for dropoff_node in self.post_start_cars[self.curr_t]:
            self.next_nodes[self.next_index] = dropoff_node
            self.next_states[self.next_index] = dropoff_node
            self.next_ids[self.next_index] = self.car_id_counter
            self.next_index += 1;
            self.car_id_counter += 1        

        self.curr_index = self.next_index
        self.curr_states = self.next_states[:self.next_index]
        self.curr_ids = self.next_ids[:self.next_index]
        self.curr_nodes = self.next_nodes[:self.next_index]
        self.curr_t += 1

        end = time.time()
        print("step - 3")
        print(end-strt)

        return self.rewards[:self.r_index];

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
#            drive_t = self.geo_utils.get_num_steps(placmt_node, self.n_lng_grids, dropoff_node)
            new_dropoff_t = (self.curr_t + 1) + drive_t

            if new_dropoff_t < self.end_t and dropoff_node >= 0:
                time_index = new_dropoff_t - self.start_t;
                # maintain all previously matched rides to be 
                # considered for future cars
                self.pmr_dropoffs[time_index][self.pmr_index[time_index]] = dropoff_node;
                self.pmr_states[time_index][self.pmr_index[time_index]] = dropoff_node; 
                self.pmr_ids[time_index][self.pmr_index[time_index]] = self.car_id_counter;#car_id;
                self.pmr_index[time_index] += 1;
                self.car_id_counter += 1;

            r = 1;#self.gamma ** ((self.curr_t + 1) - r_t)
            return matched, r, placmt_node

        return matched, 0, placmt_node
