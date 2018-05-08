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
    
    def run(self):
        out = open('baseline', 'w')
        for epoch in range(self.max_epochs):
            start_t = 20
            self.sim.reset(start_t)
            rewards = [0] * (self.sim.end_t - self.sim.start_t);

            for t in range(self.sim.start_t, self.sim.end_t):

                pmr_t = t - self.sim.start_t;
                
                a_t = [-1] * self.sim.curr_index
                for j in range(self.sim.curr_index):
                    a_t[j] = np.random.choice(self.action_dim, 1)[0]
                
                pmr_a_t = [] 
                if self.sim.pmr_index[pmr_t] > 0:
                    pmr_a_t = [-1] * self.sim.pmr_index[pmr_t]
                    for j in range(self.sim.pmr_index[pmr_t]):
                        pmr_a_t[j] = np.random.choice(self.action_dim, 1)[0]


                ids_t = self.sim.curr_ids[:self.sim.curr_index]
                num_ids = len(ids_t);
                
                if self.sim.pmr_index[pmr_t] > 0:
                    num_ids += len(self.sim.pmr_ids[pmr_t][:self.sim.pmr_index[pmr_t]]);
                print("ts %d, ids %d" % (t, num_ids))
                
                r_t = self.sim.step(a_t, pmr_a_t)
                rewards[pmr_t] = np.sum(r_t)

            print("reward %.2f" % np.sum(rewards))
            out.write("%d, %.2f\n" % (epoch, np.sum(rewards)))
            out.flush()
        out.close()
