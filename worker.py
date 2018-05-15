import os
import uuid 
import sys


class Worker:
    def __init__(self, sim, n_time_bins, train_windows, test_window,
            state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sim = sim
        self.n_time_bins = n_time_bins
        self.actor_alpha = 0.0005
        self.critic_alpha = 0.0005
        self.gamma = 0.90
        self.epsilon = 0.5
        self.n = 15;
        self.train_windows = train_windows
        self.test_window = test_window

        with tf.variable_scope(name):
            self.actor_net = ActorEstimator()
            self.critic_net = CriticEstimator()
    
    def _get_actions(self, p):
        c = p.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        a = (u < c).argmax(axis=1)
        return a

    def _aggregate(self, trajs, rewards, actions, times, states_t, \
            r_t, a_t, t_t, ids_t,ids_idx):
        for i in range(len(states_t)):
            car_id = ids_t[i]
            idx = ids_idx[car_id]
            trajs[car_id][idx] = states_t[i]
            rewards[car_id][idx] = r_t[i]
            actions[car_id][idx] = a_t[i]
            times[car_id][idx] = t_t
            ids_idx[car_id] += 1

    def train(self):
        max_epochs = 60
        rewards_train = [0] * max_epochs;
        rewards_test = [0] * max_epochs;
        test_num_rides = [0] * max_epochs;
        costs = [0] * max_epochs

        self._setup_output_dirs()
        
        for epoch in range(max_epochs):
            start_t = np.random.choice(self.train_windows, 1)[0] 
            self.sim.reset(start_t)
            break
            """
            num_ts = self.sim.end_t - self.sim.start_t;
            # classes should be total number of cars in simulation
            # this should be safe though
            num_cars = 0;
            ids_idx = [0] * self.sim.tot_cells;
            trajs = [[]] * self.sim.tot_cells;
            rewards = [[]] * self.sim.tot_cells;
            actions = [[]] * self.sim.tot_cells;
            times = [[]] * self.sim.tot_cells;
            for i in range(self.sim.tot_cells):
                trajs[i] = [0] * num_ts;
                rewards[i] = [0] * num_ts;
                actions[i] = [0] * num_ts;
                times[i] = [0] * num_ts;
            
            #episode_probs = self.actor_sess.run(self.actor_out_layer,
            #         feed_dict={self.actor_states: all_states});

            #beginning of an episode run 
            for t in range(self.sim.start_t, self.sim.end_t):
                pmr_t = t - self.sim.start_t
                state_nodes = self.sim.curr_states[:self.sim.curr_index]
                p_t = episode_probs[state_nodes]

                a_t = self._get_actions(p_t)

                # obtain actions for previously (p) matched (m) rides (r) 
                pmr_nodes = []
                if self.sim.pmr_index[pmr_t] > 0:
                    pmr_nodes = self.sim.pmr_states[pmr_t][:self.sim.pmr_index[pmr_t]]
                    #pmr_p_t = episode_probs[pmr_nodes]
                    pmr_a_t = self._get_actions(pmr_p_t)
                    
                states_t = self.sim.get_states();
                ids_t = self.sim.curr_ids[:self.sim.curr_index]
                num_cars = np.max([np.max(ids_t),num_cars]);

                print("begin episode")
                # step in the enviornment
                r_t = self.sim.step(a_t, pmr_a_t)
                end = time.time();
                print("step time")

                # len of r_t should equal to current states (states_t) and 
                # states obtained from pmr
                         
                self._aggregate(trajs, rewards, actions, times, states_t, 
                        r_t, a_t, t, ids_t, ids_idx)
                
                if self.sim.pmr_index[pmr_t] > 0:
                    self._aggregate(trajs, rewards, actions, times,
                            self.sim.get_pmr_states(t),
                            r_t[len(states_t):],
                            pmr_a_t,
                            t,
                            self.sim.pmr_ids[pmr_t][:self.sim.pmr_index[pmr_t]],
                            ids_idx)
                print("aggregate")

                                

            #end of an episode run and results aggregated

            for car_id in range(num_cars):
                idx = ids_idx[car_id];
                trajs[car_id] = trajs[car_id][:idx];
                rewards[car_id] = rewards[car_id][:idx];
                actions[car_id] = actions[car_id][:idx];
                times[car_id] = times[car_id][:idx];
            
            if do_animate:
                self.create_animation(imaging_data, epoch, self.sim.start_t, self.sim.end_t);

            print("Number of Cars: %f" % (num_cars));
            num_reqs_tot = sum(self.sim.curr_req_size)
            print("Number of Requests: %f" % (num_reqs_tot));
            num_rides_tot = sum(self.sim.curr_req_index)
            print("Number of Rides: %f" % (num_rides_tot));
            for car_id in range(num_cars):
                r = rewards[car_id]
                V_omega = self.critic_sess.run(self.critic_out_layer,
                        feed_dict={self.critic_states: trajs[car_id]}).flatten()
                
                t_s = times[car_id];
                R = [0]*len(r);
                for i in range(len(t_s)):
                    ts = t_s[i:];
                    tp = [ts[0]] + ts[0:-1];
                    td = (np.asarray(ts) - np.asarray(tp)).tolist();
                    cum_td = np.cumsum(td).tolist();
                    cum_r = 0;
                    for j in range(len(ts)):
                        if cum_td[j] >= self.n:
                            cum_r += V_omega[i+j] * (self.gamma**cum_td[j]);
                            break;
                        cum_r += r[i+j] * (self.gamma**cum_td[j]);
                    R[i] = cum_r
                
                _, c = self.critic_sess.run([self.critic_train_op, 
                    self.critic_loss_op], 
                    feed_dict={self.critic_states: trajs[car_id], 
                        self.critic_values: R})

            temp_c = [0] * len(rewards);
            temp_r = [0] * len(rewards);
            k = 0;
            for car_id in range(num_cars):
#                idx = ids_idx[car_id];
                r = rewards[car_id]#[:idx]
                V_omega = self.critic_sess.run(self.critic_out_layer,
                        feed_dict={self.critic_states: trajs[car_id]}).flatten()

                t_s = times[car_id];
                R = [0]*len(r)
                for i in range(len(t_s)):
                    ts = t_s[i:];
                    tp = [ts[0]] + ts[0:-1];
                    td = (np.asarray(ts) - np.asarray(tp)).tolist();
                    cum_td = np.cumsum(td).tolist();
                    cum_r = 0
                    for j in range(len(ts)):
                        if cum_td[j] >= self.n:
                            cum_r += V_omega[i+j] * (self.gamma**cum_td[j]);
                            break;
                        cum_r += r[i+j] * (self.gamma**cum_td[j]);
                    R[i] = cum_r
                
                values = (R - V_omega);
                a_s = actions[car_id];
                one_hot_values = np.zeros([len(a_s),self.action_dim]);
                
                for j in range(len(a_s)):
                    one_hot_values[j, a_s[j]] = values[j];

                _, c = self.actor_sess.run([self.actor_train_op,
                    self.actor_loss_op],
                    feed_dict={self.actor_states: trajs[car_id], 
                    self.actor_values: one_hot_values});
                temp_c[k] = c;
                temp_r[k] =np.sum(r);
                k += 1;
            
            print("sum reward %.2f" % np.sum(temp_r))
            print("sum cost %.2f" % np.sum(temp_c))
            #print("sum abs cost %.2f" % np.sum(np.abs(temp_c)))

            costs[epoch] = np.mean(temp_c);
            rewards_train[epoch] = np.sum(temp_r);
            rewards_test[epoch], test_num_rides[epoch] = self.test();
            print('test rewards', rewards_test[epoch])
            
            if save_training:
                train_out_str = "e, %d, %.2f, %.2f, %.2f\n" \
                    % (epoch, np.sum(temp_r), costs[-1], np.sum(np.abs(temp_c)))
                self.train_out.write(train_out_str)
                
                test_out_str = "%d, %.2f\n, %d\n" % (
                        epoch, rewards_test[epoch], test_num_rides[epoch])
                self.test_out.write(test_out_str)
            
            self.test_out.flush()
            self.train_out.flush()
        self.train_out.close()
        self.test_out.close()
        """




