from __future__ import division
import tensorflow as tf, numpy as np, sys
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.animation import FuncAnimation

class Baseline:
    def __init__(self, sim, max_epochs=20):
        self.sim = sim
        self.max_epochs = max_epochs
        self.action_dim = self.sim.n_actions
    
    def _aggregate(self, trajs, rewards, actions, times, states_t, \
            r_t, a_t, t_t, ids_t):
        for i in range(len(states_t)):
            if ids_t[i] not in trajs:
                trajs[ids_t[i]] = []
                rewards[ids_t[i]] = []
                actions[ids_t[i]] = []
                times[ids_t[i]] = [];
            trajs[ids_t[i]].append(states_t[i])
            rewards[ids_t[i]].append(r_t[i])
            actions[ids_t[i]].append(a_t[i])
            times[ids_t[i]].append(t_t);

    def run(self):
        rewards_arr = []
        out = open('baseline', 'w')
        for epoch in range(self.max_epochs):
            start_t = 20 #np.random.randint(self.sim.past_t, 
                    #self.max_t - self.sim.episode_duration)

            self.sim.reset(start_t)

            trajs = {}
            rewards = {}
            actions = {}
            times = {}

            #beginning of an episode run 
            for t in range(self.sim.start_t, self.sim.end_t):
                a_t = []
                for j in range(len(self.sim.curr_states)):
                    a = np.random.choice(self.action_dim, 1)[0]
                    a_t.append(a)
                # obtain actions for previously (p) matched (m) rides (r) 
                pmr_a_t = []
                if t in self.sim.pmr_states:
                    for j in range(len(self.sim.pmr_states[t])):
                        a = np.random.choice(self.action_dim, 1)[0]
                        pmr_a_t.append(a)
                
                states_t = self.sim.curr_states
                ids_t = self.sim.curr_ids
                num_ids = len(ids_t);
                if (t in self.sim.pmr_ids):
                    num_ids += len(self.sim.pmr_ids);
                print("ts %d, ids %d" % (t, num_ids))
                
                # step in the enviornment
                r_t = self.sim.step(a_t, pmr_a_t)

                # len of r_t should equal to current states (states_t) and 
                # states obtained from pmr
                assert (len(states_t) + len(pmr_a_t)) == len(r_t)  
                assert (len(ids_t) + len(pmr_a_t)) == len(r_t)  
                
                self._aggregate(trajs, rewards, actions, times, states_t, 
                        r_t, a_t, t, ids_t)

                if t in self.sim.pmr_ids:
                    self._aggregate(trajs, rewards, actions, times,
                            self.sim.pmr_states[t],
                            r_t[len(states_t):],
                            pmr_a_t,
                            t,
                            self.sim.pmr_ids[t])
            #end of an episode run and results aggregated

            temp_r = [];
            for car_id, r in rewards.items():
                temp_r.append(np.sum(r));

            print("reward %.2f" % np.sum(temp_r))
            out.write("%d, %.2f\n" % (epoch, np.sum(temp_r)))
            out.flush()

            rewards_arr.append(np.sum(temp_r));
        out.close()
