import os
import uuid 
import sys
import tensorflow as tf
import numpy as np

from pg.estimators import ActorEstimator, CriticEstimator

class Worker:
    def __init__(self, name, sim, n_time_bins, train_bins, 
            test_tb_starts,
            state_dim, 
            action_dim):
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sim = sim
        self.n_time_bins = n_time_bins
        self.gamma = 0.90
        self.epsilon = 0.5
        self.n = 15
        self.train_bins = train_bins
        self.test_tb_starts = test_tb_starts

        with tf.variable_scope(name):
            self.actor_net = ActorEstimator(sim.n_lat_grids, sim.n_lng_grids,
                    self.action_dim)
            self.critic_net = CriticEstimator(sim.n_lat_grids, sim.n_lng_grids,
                    reuse=True)
    
    def _get_actions(self, p):
        c = p.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        a = (u < c).argmax(axis=1)
        return a

    def _aggregate(self, trajs, rewards, actions, times, states_t, 
            r_t, a_t, t_t, ids_t, ids_idx):
        for i in range(len(states_t)):
            car_id = ids_t[i]
            idx = ids_idx[car_id]
            trajs[car_id][idx] = states_t[i]
            rewards[car_id][idx] = r_t[i]
            actions[car_id][idx] = a_t[i]
            times[car_id][idx] = t_t
            ids_idx[car_id] += 1

    def train(self, sess):
        with sess.as_default(), sess.graph.as_default():
            max_epochs = 60
            rewards_train = [0] * max_epochs
            rewards_test = [0] * max_epochs
            costs = [0] * max_epochs

            for epoch in range(max_epochs):
                start_t = np.random.choice(self.train_bins, 1)[0] 
                self.sim.reset(start_t)
                # classes should be total number of cars in simulation
                # this should be safe though
                
                ids_idx = [0] * self.sim.max_cars
                trajs = [[0]*self.sim.episode_duration 
                        for i in range(self.sim.max_cars)]
                rewards = [[0]*self.sim.episode_duration 
                        for i in range(self.sim.max_cars)]
                actions = [[0]*self.sim.episode_duration 
                        for i in range(self.sim.max_cars)]
                times = [[0]*self.sim.episode_duration 
                        for i in range(self.sim.max_cars)]
                
                #beginning of an episode run 
                for t in range(self.sim.start_t, self.sim.end_t):
                    pmr_t = self.sim.curr_t - self.sim.start_t
                    states = self.sim.get_states(t)
                    p_t = sess.run(self.actor_net.probs,
                            {self.actor_net.states: states})

                    a_t = self._get_actions(p_t)
                    
                    # step in the enviornment
                    r_t, ids_t, prev_imgs, prev_pmr_imgs, bundle = \
                            self.sim.step(a_t)

                    self._aggregate(trajs, rewards, actions, times, 
                            prev_imgs,
                            r_t, 
                            a_t, 
                            t, 
                            ids_t, 
                            ids_idx)
                    
                    if self.sim.pmr_index[pmr_t] > 0:
                        self._aggregate(trajs, rewards, actions, times,
                                prev_pmr_imgs,
                                r_t[len(prev_imgs):],
                                a_t[len(prev_imgs):],#pmr_a_t,
                                t,
                                self.sim.pmr_ids[pmr_t][:self.sim.pmr_index[pmr_t]],
                                ids_idx)
                #end of an episode run and results aggregated

                for car_id in range(self.sim.car_id_counter):
                    idx = ids_idx[car_id]
                    trajs[car_id] = trajs[car_id][:idx]
                    rewards[car_id] = rewards[car_id][:idx]
                    actions[car_id] = actions[car_id][:idx]
                    times[car_id] = times[car_id][:idx]
                
                print("Number of Cars: %d" % (self.sim.car_id_counter))
                num_reqs_tot = sum(self.sim.n_reqs[self.sim.curr_t])
                print("Number of Requests: %d" % (num_reqs_tot))
                num_rides_tot = sum(self.sim.agg_good_placmts)
                print("Number of Rides: %d" % (num_rides_tot))

                temp_c = [0] * len(rewards)
                temp_r = [0] * len(rewards)
                k = 0

                for car_id in range(self.sim.car_id_counter):
                    r = rewards[car_id]
                    V_omega = sess.run(self.critic_net.logits,
                            feed_dict={self.critic_net.states: trajs[car_id]})
                    
                    t_s = times[car_id]
                    R = [0]*len(r)
                    for i in range(len(t_s)):
                        ts = t_s[i:];
                        tp = [ts[0]] + ts[0:-1];
                        td = np.asarray(ts) - np.asarray(tp)
                        cum_td = np.cumsum(td)
                        cum_r = 0;
                        for j in range(len(ts)):
                            if cum_td[j] >= self.n:
                                cum_r += V_omega[i+j] * (self.gamma**cum_td[j]);
                                break;
                            cum_r += r[i+j] * (self.gamma**cum_td[j]);
                        R[i] = cum_r
                    
                    _, critic_loss = sess.run([self.critic_net.train_op, 
                        self.critic_net.loss], 
                        feed_dict={self.critic_net.states: trajs[car_id],
                            self.critic_net.targets: R})
                    
                    values = (R - V_omega)
                    a_s = actions[car_id]
                    #one_hot_values = np.zeros([len(a_s),self.action_dim]);
                    #for j in range(len(a_s)):
                    #    one_hot_values[j, a_s[j]] = values[j];

                    _, c = sess.run([self.actor_net.train_op,
                        self.actor_net.loss],
                        feed_dict={self.actor_net.states: trajs[car_id],
                            self.actor_net.targets: values,
                            self.actor_net.actions: a_s})
                        #self.actor_values: one_hot_values});
                    temp_c[k] = c
                    temp_r[k] = np.sum(r)
                    k += 1

                print("sum reward %.2f" % np.sum(temp_r))
                print("sum cost %.2f" % np.sum(temp_c))
                #print("sum abs cost %.2f" % np.sum(np.abs(temp_c)))

                costs[epoch] = np.mean(temp_c);
                rewards_train[epoch] = np.sum(temp_r);
                rewards_test[epoch] = self.test(sess)
                print('test rewards', rewards_test[epoch])

    def test(self, sess):
        start_t = self.test_tb_starts[0]
        self.sim.reset(start_t)
        rewards = [0] * (self.sim.end_t - self.sim.start_t);

        for t in range(self.sim.start_t, self.sim.end_t):
            pmr_t = t - self.sim.start_t
            states = self.sim.get_states(t)
            p_t = sess.run(self.actor_net.probs,
                    {self.actor_net.states: states})
            
            a_t = []
            for j in range(len(p_t)):
                a_t.append(np.random.choice(self.action_dim, 1, p=p_t[j])[0])
            r_t, ids_t, _, _, _ = self.sim.step(a_t)
            rewards[pmr_t] = np.sum(r_t)

        #print("Number of Cars: %f" % (self.sim.car_id_counter));
        #num_reqs_tot = sum(self.sim.curr_req_size)
        #print("Number of Requests: %f" % (num_reqs_tot));
        #num_rides_tot = sum(self.sim.curr_req_index)
        #print("Number of Rides: %f" % (num_rides_tot));
        return np.sum(rewards)
