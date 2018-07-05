from __future__ import division
import numpy as np
import copy
from collections import Counter
import sys
import time
import random

class Sim:
    def __init__(self, X, n_lng_grids, time_utils, geo_utils, n_actions,
                 episode_duration, gamma=0.9, past_t=5):
        self.X = X
        """
        rrs: indexed by only the start time of each episode. rrs[t] has all 
        requests indexed by the origin node. For example: rrs[t][n] returns 
        list of requests where each item of the list contains 
        [dropoff_node, travel_time, start_time].
        """
        self.rrs = {}

        """
        n_reqs: aggregate number of requests indexed by time-step. 
        n_reqs[1] will have aggregated requests for t=0 and t=1 for each cell.
        """
        self.n_reqs = {} 

        """
        pre_sim_pickups: indexed by dropoff time (d_t), returns a list of nodes 
        where drop-offs occurred at time d_t.
        """
        self.pre_sim_pickups = {}
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
        self.pmr_imgs = []
        self.curr_imgs = []
    
    def update_base_img(self):
        pmr_t = self.curr_t - self.start_t
        num_cars_per_node = [0] * self.num_cells
        
        for ni in self.curr_nodes:
            num_cars_per_node[ni] += 1
        
        for ni in self.pmr_states[pmr_t][:self.pmr_index[pmr_t]]:
            num_cars_per_node[ni] += 1
        self.base_img[:, :, 0] = np.flipud(np.reshape(num_cars_per_node, 
            [self.n_lat_grids, self.n_lng_grids]))
        
        n_reqs_rem = np.array(self.n_reqs[self.curr_t]) \
                - np.array(self.agg_good_placmts)
        self.base_img[:, :, 1] = np.flipud(np.reshape(n_reqs_rem, 
            [self.n_lat_grids, self.n_lng_grids]))
 
    def create_state_imgs(self):
        imgs = []
        for n in self.curr_nodes:
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
        self.pmr_states = [[-1] * self.max_cars \
                for i in range(self.episode_duration)]
        self.pmr_ids = [[-1] * self.max_cars \
                for i in range(self.episode_duration)]

        self.start_t = t
        self.end_t = t + self.episode_duration
        self.curr_t = t

        self.requests = self.rrs[self.start_t]
        self.agg_good_placmts = [0] * (self.num_cells)
        self.next_index = 0
        
        self.curr_nodes = self.pre_sim_pickups[self.start_t]
        self.curr_ids = range(len(self.curr_nodes))
        self.car_id_counter = len(self.curr_nodes)

        # create base image 
        self.base_img = np.zeros([self.n_lat_grids, self.n_lng_grids, 3],
                dtype=np.uint8)

        self.update_base_img()
        self.create_state_imgs()
    
    def get_states(self, t): 
        states = self.curr_imgs
        pmr_t = t - self.start_t
        if self.pmr_index[pmr_t] > 0:
            states = np.append(states, self.pmr_imgs, axis=0)
        return states 

    def step(self, a_t):
        if self.curr_t > self.end_t:
            return False
       
        th = self.time_utils.get_hour_of_day(self.curr_t)
        
        next_nodes = []
        next_ids = []
        rewards = []

        num_pickups = 0
        num_requests = 0
        num_cars = 0
        avg_reward = 0
        avg_wait_pickups = 0
        avg_wait_requests = 0
        
        rndm_smple = random.sample(range(len(self.curr_nodes)), 
                len(self.curr_nodes))
        
        #1 check curr dropoffs which can be matched
        assert len(self.curr_nodes) == len(self.curr_ids) == len(self.curr_imgs)
        for idx in rndm_smple:
            matched, r, next_node, w = self._in_rrs(self.curr_nodes[idx], 
                                           a_t[idx],
                                           self.curr_ids[idx])
            rewards.append(r)

            num_cars += 1

            if r == 0:
                next_nodes.append(next_node)
                next_ids.append(self.curr_ids[idx])
            else: 
                num_pickups += 1
                avg_reward += r
                avg_wait_pickups += w

        pmr_t = self.curr_t - self.start_t
        pmr_idx = self.pmr_index[pmr_t]
        rndm_smple = random.sample(range(pmr_idx), pmr_idx)

        #2 check dropoffs from previous matched requests if can be matched further
        for i in range(pmr_idx):
            idx = rndm_smple[i]
            matched, r, next_node, w = self._in_rrs(self.pmr_states[pmr_t][idx], 
                    a_t[idx + len(self.curr_nodes)],
                    self.pmr_ids[pmr_t][idx])
            rewards.append(r)

            num_cars += 1

            if r == 0:
                next_nodes.append(next_node)
                next_ids.append(self.pmr_ids[pmr_t][idx])
            else: 
                num_pickups += 1
                avg_reward += r
                avg_wait_pickups += w

        #3 add dropoff vehicles from before beginning of episode
        if self.curr_t + 1 in self.pre_sim_pickups:
            for dropoff_node in self.pre_sim_pickups[self.curr_t+1]:
                next_nodes.append(dropoff_node)
                next_ids.append(self.car_id_counter)
                self.car_id_counter += 1

        prev_ids = self.curr_ids
        self.curr_nodes = next_nodes
        self.curr_ids = next_ids
        self.curr_t += 1
        
        prev_imgs = self.curr_imgs
        prev_pmr_imgs = self.pmr_imgs

        if self.curr_t < self.end_t:
            self.update_base_img()
            self.create_state_imgs()
        """
        TODO: Not sure what stat. is being collected here.
        for i in range(self.num_cells):
            for j in range(self.agg_good_placmts[i],
                    self.n_reqs[self.curr_t+1][i]):
                req = self.requests[i][j]
                avg_wait_requests += req[1]
                num_requests += 1
        """
        avg_reward /= num_pickups
        avg_wait_pickups /= num_pickups
        #avg_wait_requests /= num_requests

        bundle = (num_pickups, num_cars, num_requests, avg_reward, 
                avg_wait_pickups)#, avg_wait_requests)

        return rewards, prev_ids, prev_imgs, prev_pmr_imgs, bundle

    def _in_rrs(self, dropoff_node, a, car_id):
        matched = False
        placmt_node = self.get_next_node(dropoff_node, a)
        
        if self.n_reqs[self.curr_t + 1][placmt_node] > \
                self.agg_good_placmts[placmt_node]:
            req = self.requests[placmt_node][self.agg_good_placmts[placmt_node]]
            self.agg_good_placmts[placmt_node] += 1
            matched = True
            
            dropoff_node = req[0]
            r_t = req[1]
            drive_t = req[2]
            new_dropoff_t = (self.curr_t + 1) + drive_t

            if new_dropoff_t < self.end_t and dropoff_node >= 0:
                time_index = new_dropoff_t - self.start_t;
                # maintain all previously matched rides to be 
                # considered for future cars
                self.pmr_states[time_index][self.pmr_index[time_index]] = dropoff_node
                self.pmr_ids[time_index][self.pmr_index[time_index]] = self.car_id_counter
                self.pmr_index[time_index] += 1
                self.car_id_counter += 1
               
            r = 1 #self.gamma ** ((self.curr_t + 1) - r_t)
            w = self.curr_t+1 - r_t # wait time
            return matched, r, placmt_node, w
        
        return matched, 0, placmt_node, 0
