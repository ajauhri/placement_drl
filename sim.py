from __future__ import division
import numpy as np
import copy

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
        self.classes = len(self.geo_utils.lat_grids) *\
                len(self.geo_utils.lng_grids)
    
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

<<<<<<< HEAD
=======
    def _add_all_dropoffs(self):
        th = self.time_utils.get_hour_of_day(self.curr_t)
        for t in range(self.start_t, self.end_t):
            if t in self.dropoff_buckets:
                    for idx in self.dropoff_buckets[t]:
                        dropoff_node, d_lat_idx, d_lon_idx = \
                                self.geo_utils.get_node(self.X[idx, 5:7])

                        if (d_lat_idx >= 0 and d_lon_idx >= 0):
                            p_t = self.time_utils.get_bucket(self.X[idx, 1])
                            if p_t < self.start_t:
                                self.curr_nodes[self.curr_index] = dropoff_node
                                self.curr_states[self.curr_index] = dropoff_node
                                self.curr_ids[self.curr_index] = self.car_id_counter
                                self.curr_index += 1;
                                self.car_id_counter += 1

>>>>>>> 8c93295d24747911acfe0fdfa176591071b5464e
    def reset(self, t):
        self.pmr_index = [0] * self.episode_duration; 
        # technically should be maximum number of cars, not classes
        self.pmr_dropoffs = [[-1] * (self.classes)] * self.episode_duration;
        self.pmr_states = [[-1] * (self.classes)] * self.episode_duration;
        self.pmr_ids = [[-1] * (self.classes)] * self.episode_duration;
        self.start_t = t
        self.end_t = t + self.episode_duration
        self.curr_t = t
        th = self.time_utils.get_hour_of_day(t)
        self.car_id_counter = 0
<<<<<<< HEAD
        self.requests = copy.deepcopy(self.rrs)
        
        self.curr_ids = []
        self.curr_states = []
        self.curr_nodes = []
        #self._add_all_dropoffs()
=======
        self.requests = self.rrs[self.start_t];
        self.curr_req_size = self.req_sizes[self.start_t];
        self.curr_req_index = [0] * (self.classes);

        self.curr_index = 0
        self.curr_ids = [-1] * (self.classes);
        self.curr_states = [-1] * (self.classes);
        self.curr_nodes = [-1] * (self.classes);
 
        self._add_all_dropoffs()
        '''
>>>>>>> 8c93295d24747911acfe0fdfa176591071b5464e
        for idx in self.dropoff_buckets[t]:
            dropoff_node, d_lat_idx, d_lon_idx = \
                    self.geo_utils.get_node(self.X[idx, 5:7])

            if (d_lat_idx >=0 and d_lon_idx >= 0):
<<<<<<< HEAD
                self.curr_states.append(self.get_state(dropoff_node, th))
                self.curr_nodes.append(dropoff_node)
                self.curr_ids.append(self.car_id_counter)
                self.car_id_counter += 1
        
=======
                self.curr_nodes[self.curr_index] = dropoff_node
                self.curr_states[self.curr_index] = dropoff_node
                self.curr_ids[self.curr_index] = self.car_id_counter
                self.curr_index += 1;
                self.car_id_counter += 1        
        '''

>>>>>>> 8c93295d24747911acfe0fdfa176591071b5464e
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
            r, next_node = self._in_rrs_v2(self.curr_nodes[i], 
                                           a_t[i],
                                           self.curr_ids[i],
                                           next_th)
            rewards[r_index] += r;
            r_index += 1;
            
            if r == 0:
                next_nodes[next_index] = next_node
                next_states[next_index] = next_node
                next_ids[next_index] = self.curr_ids[i]
                next_index += 1;
        
        #2 check dropoffs from previous matched requets if can be matched further
        for i in range(self.pmr_index[self.curr_t - self.start_t]):
            r, next_node = self._in_rrs_v2(self.pmr_dropoffs[self.curr_t - self.start_t][i], 
                                           pmr_a_t[i],
                                           self.pmr_ids[self.curr_t - self.start_t][i],
                                           next_th)
            rewards[r_index] += r;
            r_index += 1;
                        
<<<<<<< HEAD
                if r == 0:
                    next_nodes.append(next_node)
                    next_states.append(self.get_state(next_node, next_th))
                    next_ids.append(self.pmr_ids[self.curr_t][i])
        
=======
            if r == 0:
                next_nodes[next_index] = next_node
                next_states[next_index] = next_node
                next_ids[next_index] = self.pmr_ids[self.curr_t - self.start_t][i]
                next_index += 1;

        '''
>>>>>>> 8c93295d24747911acfe0fdfa176591071b5464e
        #3 add dropoff vehicles from before beginning of episode
        if self.curr_t+1 in self.dropoff_buckets:
            for idx in self.dropoff_buckets[self.curr_t+1]:
                dropoff_node, d_lat_idx, d_lon_idx = \
                        self.geo_utils.get_node(self.X[idx, 5:7])

                if (d_lat_idx >= 0 and d_lon_idx >= 0):
                    p_t = self.time_utils.get_bucket(self.X[idx, 1])
                    if p_t <= self.start_t:
<<<<<<< HEAD
                        next_nodes.append(dropoff_node)
                        next_states.append(self.get_state(dropoff_node, next_th))
                        next_ids.append(self.car_id_counter)
                        self.car_id_counter += 1
            
        self.curr_states = next_states
        self.curr_ids = next_ids
        self.curr_nodes = next_nodes
=======
                        next_nodes[next_index] = dropoff_node
                        next_states[next_index] = dropoff_node
                        next_ids[next_index] = self.car_id_counter
                        next_index += 1;
                        self.car_id_counter += 1        
        '''

        self.curr_index = next_index
        self.curr_states = next_states[:next_index]
        self.curr_ids = next_ids[:next_index]
        self.curr_nodes = next_nodes[:next_index]
>>>>>>> 8c93295d24747911acfe0fdfa176591071b5464e
        self.curr_t += 1
        
        return rewards[:r_index];

    def _in_rrs(self, dropoff_node, a, car_id, next_th):
        matched = False
        placmt_node = self.get_next_node(dropoff_node, a)
        
        if placmt_node in self.requests:
            for i in range(len(self.requests[placmt_node])):
                if (not self.requests[placmt_node][i].picked) and (\
                        (self.requests[placmt_node][i].r_t < (self.curr_t + 1) \
                        and 
                        #self.requests[placmt_node][i].r_t > (self.start_t - \
                        #        self.past_t) and
                        self.requests[placmt_node][i].p_t >= (self.start_t))
                     or
                        self.requests[placmt_node][i].r_t == self.curr_t + 1):

                        self.requests[placmt_node][i].picked = True
                        
                        pickup_t = self.requests[placmt_node][i].p_t
                        dropoff_t = self.requests[placmt_node][i].d_t
                        drive_t = dropoff_t - pickup_t
                        self.requests[placmt_node][i].p_t = self.curr_t + 1
                        
                        new_dropoff_t = (self.curr_t + 1) + drive_t
                        if new_dropoff_t < self.end_t:
                                                        
                            # maintain all previously matched rides to be 
                            # considered for future cars
                            if self.requests[placmt_node][i].dn >= 0:
                                if new_dropoff_t not in self.pmr_dropoffs:
                                    self.pmr_dropoffs[new_dropoff_t] = []
                                    self.pmr_states[new_dropoff_t] = []
                                    self.pmr_ids[new_dropoff_t] = []

                                self.pmr_dropoffs[new_dropoff_t].append(
                                        self.requests[placmt_node][i].dn)
                                self.pmr_states[new_dropoff_t].append(\
                                        self.get_state(\
                                        self.requests[placmt_node][i].dn, 
                                        next_th))
                                self.pmr_ids[new_dropoff_t].append(car_id)
                        
                        r = self.gamma ** ((self.curr_t + 1) - \
                                self.requests[placmt_node][i].r_t)
                        return r, placmt_node
        if not matched:
            return 0, placmt_node


    def _in_rrs_v2(self, dropoff_node, a, car_id, next_th):
        matched = False
        placmt_node = self.get_next_node(dropoff_node, a)
        
        if self.curr_req_size[placmt_node] > self.curr_req_index[placmt_node]:
            req = self.requests[placmt_node][self.curr_req_index[placmt_node]];
            self.curr_req_index[placmt_node] += 1;
            
            dropoff_node = req[0];
            drive_t = req[1];
            r_t = req[2];
            new_dropoff_t = (self.curr_t + 1) + drive_t
            if new_dropoff_t < self.end_t and dropoff_node >= 0:
                time_index = new_dropoff_t - self.start_t;
                # maintain all previously matched rides to be 
                # considered for future cars
                self.pmr_dropoffs[time_index][self.pmr_index[time_index]] = dropoff_node;
                self.pmr_states[time_index][self.pmr_index[time_index]] = dropoff_node; 
                self.pmr_ids[time_index][self.pmr_index[time_index]] = car_id;
                self.pmr_index[time_index] += 1;

            r = self.gamma ** ((self.curr_t + 1) - r_t)
            return r, placmt_node

        return 0, placmt_node
