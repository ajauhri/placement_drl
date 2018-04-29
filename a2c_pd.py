from __future__ import division
import tensorflow as tf, numpy as np, sys
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

class A2C:
    def __init__(self, sim, n_time_bins, state_dim=3, 
            action_dim=5, hidden_units=16):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.sim = sim
        self.n_time_bins = n_time_bins
        self.actor_alpha = 0.0001
        self.critic_alpha = 0.0001
        self.gamma = 0.99
        self.epsilon = 0.5
        self.n = 5

        num_mins_per_bin = 2
        num_bins_in_hour = 60 / num_mins_per_bin
        num_hours_per_day = 24
        self.data_for_day = num_bins_in_hour * num_hours_per_day

        self.setup_actor_critic()

    def setup_actor_critic(self):
        with tf.Graph().as_default() as actor:
            actor_weights = {
                'h1': tf.Variable(tf.random_uniform([self.state_dim, 
                    self.hidden_units])),
                'h2': tf.Variable(tf.random_uniform([self.hidden_units, 
                    self.hidden_units])),
                'h3': tf.Variable(tf.random_uniform([self.hidden_units, 
                    self.hidden_units])),
                'out': tf.Variable(tf.random_uniform([self.hidden_units, 
                    self.action_dim]))
                }

            actor_biases = {
                'b1': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'b2': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'b3': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'out': tf.Variable(tf.constant(0.0, shape=[self.action_dim]))
                }

            self.actor_states = tf.placeholder("float", [None, self.state_dim])
            self.actor_values = tf.placeholder("float", [None, self.action_dim])

            actor_l1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.actor_states, 
                actor_weights['h1']), actor_biases['b1']))
            actor_l2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(actor_l1, 
                actor_weights['h2']), actor_biases['b2']))
            actor_l3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(actor_l2, 
                actor_weights['h3']), actor_biases['b3']))
            self.actor_out_layer = tf.nn.softmax(tf.nn.bias_add(\
                    tf.matmul(actor_l3, actor_weights['out']), 
                    actor_biases['out']))

            self.actor_loss_op = tf.reduce_mean(tf.log(tf.clip_by_value(\
                    self.actor_out_layer,1E-15,0.99))*self.actor_values)
            self.actor_optimizer = tf.train.AdamOptimizer(self.actor_alpha)
            self.actor_train_op = self.actor_optimizer.minimize(\
                    self.actor_loss_op)
            init = tf.global_variables_initializer()
        
        self.actor_sess = tf.Session(graph=actor)
        self.actor_sess.run(init)

        with tf.Graph().as_default() as critic:
            critic_weights = {
                'h1': tf.Variable(tf.random_uniform([self.state_dim, 
                    self.hidden_units])),
                'h2': tf.Variable(tf.random_uniform([self.hidden_units, 
                    self.hidden_units])),
                'h3': tf.Variable(tf.random_uniform([self.hidden_units,
                    self.hidden_units])),
                'out': tf.Variable(tf.random_uniform([self.hidden_units, 1]))
                }
            
            critic_biases = {
                'b1': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'b2': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'b3': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'out': tf.Variable(tf.constant(0.0, shape=[1]))
                }
            
            self.critic_states = tf.placeholder("float", [None, self.state_dim])
            self.critic_values = tf.placeholder("float", [None])
           
            critic_l1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.critic_states, 
                critic_weights['h1']), critic_biases['b1']))
            critic_l2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(critic_l1, 
                critic_weights['h2']), critic_biases['b2']))
            critic_l3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(critic_l2, 
                critic_weights['h3']), critic_biases['b3']))
            self.critic_out_layer = tf.matmul(critic_l3, 
                    critic_weights['out']) + critic_biases['out']
            
            self.critic_loss_op = tf.reduce_mean(tf.square(\
                    tf.subtract(self.critic_values, self.critic_out_layer)));
            self.critic_optimizer = tf.train.AdamOptimizer(self.critic_alpha)
            self.critic_train_op = self.critic_optimizer.\
                    minimize(self.critic_loss_op)
            init = tf.global_variables_initializer()

        self.critic_sess = tf.Session(graph=critic)
        self.critic_sess.run(init)
        
        """
        a = self.actor_sess.run(actor_out_layer,
                        feed_dict={actor_states: [[4, 5, 4]]})
        c = self.critic_sess.run(critic_out_layer,
                        feed_dict={critic_states: [[4, 5, 4]]})
        print(a, np.sum(a))
        print(c)
        """
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

    def train(self):
        max_epochs = 10
        rewards_test = []
        costs = []
        for epoch in range(max_epochs):
            start_t = 6
            self.sim.reset(start_t)

            trajs = {}
            rewards = {}
            actions = {}
            times = {}

            fig = plt.figure(1)
            plt.ion();
            m = Basemap(projection = 'stere', 
                        llcrnrlat=37.765,urcrnrlat=37.81,
                        llcrnrlon=-122.445,urcrnrlon=-122.385,
                        resolution='f',
                        lat_0 = 37.78,lon_0 = -122.41)
            m.drawmapboundary(fill_color='aqua',zorder=1)
            m.fillcontinents(color='mediumseagreen',zorder=2)
            m.readshapefile("sf_road_shapefile/geo_export_bca4a474-0dad-4589-b7c2-f325f80f9119","sf_roads",zorder=3);
            plt.xlabel('lon')
            plt.ylabel('lat')
            plt.show()
            
            #beginning of an episode run 
            for t in range(self.sim.start_t, self.sim.end_t):
                p_t = self.actor_sess.run(self.actor_out_layer,
                        feed_dict={self.actor_states: self.sim.curr_states})

                a_t = []
                for j in range(len(p_t)):
                    a = np.random.choice(self.action_dim, 1, p=p_t[j])[0]
                    a_t.append(a)
                
                # obtain actions for previously (p) matched (m) rides (r) 
                pmr_a_t = []
                if t in self.sim.pmr_states:
                    pmr_rides = len(self.sim.pmr_states[t])
                    pmr_t = self.actor_sess.run(self.actor_out_layer, 
                            feed_dict={\
                                    self.actor_states: self.sim.pmr_states[t]})
                    
                    for j in range(len(pmr_t)):
                        a = np.random.choice(self.action_dim, 1, p=pmr_t[j])[0]
                        pmr_a_t.append(a)

                states_t = self.sim.curr_states
                ids_t = self.sim.curr_ids
                print("ids " + str(len(ids_t)))
                
                i = 0;
                lat = [];
                lon = [];
                for node in self.sim.curr_nodes:
                    loc = self.sim.geo_utils.get_centroid_v2(node,self.sim.n_lng_grids)
                    lat.append(loc[0]);
                    lon.append(loc[1]);
                loc = m(lon,lat)
                plot_points = plt.scatter(loc[0],loc[1],5,color='r',zorder=4);
                plt.draw();
                plt.pause(0.001);
                plot_points.remove();

                # step in the enviornment
                r_t = self.sim.step(a_t, pmr_a_t)

                # len of r_t should equal to current states (states_t) and 
                # states obtained from pmr
                assert (len(states_t) + len(pmr_a_t)) == len(r_t)  
                assert (len(ids_t) + len(pmr_a_t)) == len(r_t)  
                
#                print('iter %d' % t)
                
                self._aggregate(trajs, rewards, actions, times, states_t, 
                        r_t, a_t, t, ids_t)

                if t in self.sim.pmr_ids:
                    self._aggregate(trajs, rewards, actions, times,
                            self.sim.pmr_states[t],
                            r_t,
                            pmr_a_t,
                            t,
                            self.sim.pmr_ids[t])
            #end of an episode run and results aggregated
           
            
            for car_id, r in rewards.items():
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
#                    if (cum_td[-1] < self.n):
                        # train on next drop off location
                
                _, c = self.critic_sess.run([self.critic_train_op,
                    self.critic_loss_op], 
                    feed_dict={self.critic_states: trajs[car_id], 
                        self.critic_values: R})

            temp_c = [];
            temp_r = [];
            for car_id,r in rewards.items():
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
                    one_hot_values[j,a_s[j]] = values[j];

                prob_out = self.actor_sess.run(self.actor_out_layer, 
                        feed_dict={self.actor_states: trajs[car_id]})

                _, c = self.actor_sess.run([self.actor_train_op,
                    self.actor_loss_op],
                    feed_dict={self.actor_states: trajs[car_id], 
                    self.actor_values: one_hot_values});
                temp_c.append(c);
                temp_r.append(np.sum(r));

            print("reward " + str(np.sum(temp_r)))
            print("cost " + str(np.mean(temp_c)))

#           logging.debug("train: epoch %d, time hour %d, cost %.4f" % 
#           (epoch, self.sim.time_utils.get_hour_of_day(epoch), cost))
            costs.append(np.mean(temp_c))
            rewards_test.append(np.sum(temp_r));
#           rewards_test.append(self.test())

        fig = plt.figure(1)
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(costs, color='r', linewidth=1)
        plt.xlabel('epochs')
        plt.ylabel('cost')
        ax2 = ax1.twinx()
        ax2.plot(rewards_test, 'b--', linewidth=1)
        plt.ylabel('reward')
        plt.show()


    def test(self):
        self.sim.start_test()
        tot_r = 0
        n_dropoffs = 0
        
        for epoch in range(self.n_time_bins)[:60]:
            if epoch in self.sim.test_dropoff_maps and \
                    len(self.sim.test_dropoff_maps) > 1:
                states = []
                nodes = []
                for n in self.sim.test_dropoff_maps[epoch].keys():
                    states.append(self.sim.get_state_rep(n, epoch))
                    nodes.append(n)
                    n_dropoffs += self.sim.test_dropoff_maps[epoch][n]
                states = np.array(states)
                q = self.sess.run(self.out_layer, feed_dict={self.X: states})
                a = np.argmax(q, axis=1)
                for i in range(len(states)):
                    _, r = self.sim.step_test(nodes[i], epoch, a[i])
                    tot_r += r
        logging.debug('test: test reward %d, n_dropoffs %d' \
                % (tot_r, n_dropoffs))
        self.sim.reset_test()
        return tot_r
