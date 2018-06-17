from __future__ import division
import numpy as np
import copy
from collections import Counter
import sys
import time;
import random

class Sim:
    def __init__(self, X, n_lng_grids, time_utils, geo_utils, n_actions,
                 episode_duration=20, gamma=0.9, past_t=5):
        self.X = X
        self.rrs = {}
        self.req_sizes = {}
        self.post_start_cars = {}
        self.episode_duration = episode_duration
        self.gamma = gamma
        self.past_t = past_t
 
        self.n_actions = n_actions
        self.time_utils = time_utils
        self.geo_utils = geo_utils
        self.n_lng_grids = len(self.geo_utils.lng_grids) - 1
        self.n_lat_grids = len(self.geo_utils.lat_grids) - 1
        self.num_cells = (len(self.geo_utils.lat_grids) - 1) *\
            (len(self.geo_utils.lng_grids) - 1)
        self.max_cars = self.num_cells*3
    
    def update_base_img(self):
        pmr_t = self.curr_t - self.start_t
        num_cars_per_node = [0] * self.num_cells
        for ni in self.curr_nodes[:self.curr_index]:
            num_cars_per_node[ni] += 1
        for ni in self.pmr_dropoffs[pmr_t][:self.pmr_index[pmr_t]]:
            num_cars_per_node[ni] += 1
        self.base_img[:, :, 0] = np.flipud(np.reshape(num_cars_per_node, 
            [self.n_lat_grids, self.n_lng_grids]))
        
        num_reqs = np.array(self.curr_req_size) \
                - np.array(self.curr_req_index)
        self.base_img[:, :, 1] = np.flipud(np.reshape(num_reqs, 
            [self.n_lat_grids, self.n_lng_grids]))
 
    def create_state_imgs(self):
        imgs = []
        for n in self.curr_nodes[:self.curr_index]:
            y = int(n % self.n_lng_grids)
            x = int(n / self.n_lng_grids)
            state = np.copy(self.base_img)
            state[-x, y, 2] = 255
            imgs.append(state)
        self.curr_imgs = np.array(imgs)
         
        pmr_t = self.curr_t - self.start_t
        if self.pmr_index[pmr_t] > 0:
            imgs = []
            pmr_nodes = self.pmr_states[pmr_t][:self.pmr_index[pmr_t]]
            for n in pmr_nodes:
                y = int(n % self.n_lng_grids)
                x = int(n / self.n_lng_grids)
                state = np.copy(self.base_img)
                state[-x, y, 2] = 255
                imgs.append(state)
            self.pmr_imgs = np.array(imgs) 

    def get_pmr_states(self, t):
        t_i = t - self.start_t;
        pmr_state = self.pmr_states[t_i][:self.pmr_index[t_i]];
        return np.eye(self.num_cells)[pmr_state].tolist();
    
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
        self.pmr_index = [0] * self.episode_duration; 
        
        # technically should be maximum number of cars, not classes
        self.pmr_dropoffs = [[-1] * self.max_cars \
                for i in range(self.episode_duration)]
        self.pmr_states = [[-1] * self.max_cars \
                for i in range(self.episode_duration)]
        self.pmr_ids = [[-1] * self.max_cars \
                for i in range(self.episode_duration)]

        self.start_t = t
        self.end_t = t + self.episode_duration
        self.curr_t = t

        self.requests = self.rrs[self.start_t]
        self.curr_req_size = self.req_sizes[self.start_t]
        self.curr_req_index = [0] * (self.num_cells)
        self.next_index = 0
        
        self.curr_nodes = self.post_start_cars[self.start_t]
        self.curr_index = len(self.curr_nodes)
        self.curr_ids = range(self.curr_index)
        self.car_id_counter = self.curr_index

        # create base image 
        self.base_img = np.zeros([self.n_lat_grids, self.n_lng_grids, 3],
                dtype=np.uint8)
               
    def step(self, a_t, pmr_a_t):
        if self.curr_t > self.end_t:
            return False
       
        th = self.time_utils.get_hour_of_day(self.curr_t)
        next_th = self.time_utils.get_hour_of_day(self.curr_t + 1)
        
        if self.curr_t+1 in self.req_sizes:
            self.curr_req_size = self.req_sizes[self.curr_t+1]
        
        self.next_index = 0
        self.next_nodes= [-1] * (self.max_cars)
        self.next_ids = [-1] * (self.max_cars)
        self.rewards = [0] * (self.max_cars)
        self.r_index = 0
        
        rndm_smple = random.sample(range(self.curr_index),self.curr_index);
        
        #1 check curr dropoffs which can be matched
        for i in range(self.curr_index):
            idx = rndm_smple[i];
            matched, r, next_node = self._in_rrs(self.curr_nodes[idx], 
                                           a_t[idx],
                                           self.curr_ids[idx],
                                           next_th)
            self.rewards[self.r_index] = r;
            self.r_index += 1;

            if r == 0:
                self.next_nodes[self.next_index] = next_node
                self.next_ids[self.next_index] = self.curr_ids[idx]
                self.next_index += 1;

        pmr_t = self.curr_t - self.start_t;
        pmr_idx = self.pmr_index[pmr_t];
        rndm_smple = random.sample(range(pmr_idx), pmr_idx)

        #2 check dropoffs from previous matched requets if can be matched further
        for i in range(pmr_idx):
            idx = rndm_smple[i]
            matched, r, next_node = self._in_rrs(
                    self.pmr_dropoffs[pmr_t][idx], 
                    pmr_a_t[idx],
                    self.pmr_ids[pmr_t][idx],
                    next_th)
            self.rewards[self.r_index] = r
            self.r_index += 1

            if r == 0:
                self.next_nodes[self.next_index] = next_node
                self.next_ids[self.next_index] = \
                        self.pmr_ids[pmr_t][idx]
                self.next_index += 1

        #3 add dropoff vehicles from before beginning of episode
        if (self.curr_t + 1 in self.post_start_cars):
            for dropoff_node in self.post_start_cars[self.curr_t+1]:
                self.next_nodes[self.next_index] = dropoff_node
                self.next_ids[self.next_index] = self.car_id_counter
                self.next_index += 1
                self.car_id_counter += 1        
        
        old_ids = self.curr_ids
        self.curr_index = self.next_index
        self.curr_nodes = self.next_nodes[:self.next_index]
        self.curr_ids = self.next_ids[:self.next_index]
        self.curr_t += 1
        
        return self.rewards[:self.r_index], old_ids

    def _in_rrs(self, dropoff_node, a, car_id, next_th):
        matched = False
        placmt_node = self.get_next_node(dropoff_node, a)
        
        if self.curr_req_size[placmt_node] > self.curr_req_index[placmt_node]:
            req = self.requests[placmt_node][self.curr_req_index[placmt_node]]
            self.curr_req_index[placmt_node] += 1
            matched = True
            
            dropoff_node = req[0];
            drive_t = req[1];
            r_t = req[2];
#            drive_t = self.geo_utils.get_num_steps(placmt_node, self.n_lng_grids, dropoff_node)
            new_dropoff_t = (self.curr_t + 1) + drive_t

            if new_dropoff_t < self.end_t and dropoff_node >= 0:
                time_index = new_dropoff_t - self.start_t;
                # maintain all previously matched rides to be 
                # considered for future cars
                self.pmr_dropoffs[time_index][self.pmr_index[time_index]] = dropoff_node
                self.pmr_states[time_index][self.pmr_index[time_index]] = dropoff_node
                self.pmr_ids[time_index][self.pmr_index[time_index]] = self.car_id_counter
                self.pmr_index[time_index] += 1
                self.car_id_counter += 1

            r = 1;#self.gamma ** ((self.curr_t + 1) - r_t)
            return matched, r, placmt_node

        return matched, 0, placmt_node
