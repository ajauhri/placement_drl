from __future__ import division
from collections import Counter
import copy
import tensorflow as tf, numpy as np, sys
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.animation import FuncAnimation
import os
import uuid

import sys

class A2C:
    def __init__(self, sim, n_time_bins, train_windows, test_window,
            state_dim, action_dim, hidden_units=32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.sim = sim
        self.n_time_bins = n_time_bins
        self.actor_alpha = 0.0005
        self.critic_alpha = 0.0005
        self.gamma = 0.90
        self.epsilon = 0.5
        self.n = 5
        self.train_windows = train_windows
        self.test_window = test_window

        num_mins_per_bin = 2
        num_bins_in_hour = 60 / num_mins_per_bin
        num_hours_per_day = 24
        self.data_for_day = num_bins_in_hour * num_hours_per_day

        self.setup_actor_critic()
    
    def _add_lat_lng_cars(self, lat, lon, num_c, nodes):
        i = 0;
        for node in nodes.keys():
            loc = self.sim.geo_utils.get_centroid_v2(node, self.sim.n_lng_grids)
            lat[i] = loc[0];
            lon[i] = loc[1];
            num_c[i] = nodes[node];
            i += 1;
        return i;

    def _add_lat_lng_reqs(self, lat, lon, num_r, reqs, num_reqs):
        i = 0;
        for req in reqs:
            loc = self.sim.geo_utils.get_centroid_v2(req, self.sim.n_lng_grids)
            lat[i] = loc[0];
            lon[i] = loc[1];
            num_r[i] = num_reqs[req]
            i += 1;
        return i;

    def setup_actor_critic(self):
        with tf.Graph().as_default() as actor:
            actor_weights = {
                'h1': tf.Variable(tf.random_uniform([self.state_dim, 
                    self.hidden_units], 
                    dtype=tf.float32, minval=-.1, maxval=.1)),
                'h2': tf.Variable(tf.random_uniform([self.hidden_units, 
                    self.hidden_units], 
                    dtype=tf.float32, minval=-.1, maxval=.1)),
                'h3': tf.Variable(tf.random_uniform([self.hidden_units, 
                    self.hidden_units], 
                    dtype=tf.float32, minval=-.1, maxval=.1)),
                'out': tf.Variable(tf.random_uniform([self.hidden_units, 
                    self.action_dim], dtype=tf.float32, minval=-.1, maxval=.1))
                }

            actor_biases = {
                'b1': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'b2': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'b3': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'out': tf.Variable(tf.constant(0.0, shape=[self.action_dim]))
                }

            self.actor_states = tf.placeholder("float", [None, self.state_dim])
            self.actor_values = tf.placeholder("float", [None, self.action_dim])

            self.actor_l1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.actor_states, 
                actor_weights['h1']), actor_biases['b1']))
            self.actor_l2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.actor_l1, 
                actor_weights['h2']), actor_biases['b2']))
            self.actor_l3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.actor_l2, 
                actor_weights['h3']), actor_biases['b3']))
            self.actor_out_layer = tf.nn.softmax(tf.nn.bias_add(
                    tf.matmul(self.actor_l3, actor_weights['out']), 
                        actor_biases['out']))

            self.actor_loss_op = tf.reduce_mean(tf.multiply(-tf.log(tf.clip_by_value(\
                    self.actor_out_layer,1E-15,0.99)),self.actor_values))
            self.actor_optimizer = tf.train.AdamOptimizer(self.actor_alpha)
            self.actor_train_op = self.actor_optimizer.minimize(\
                    self.actor_loss_op)
            init = tf.global_variables_initializer()
        
        self.actor_sess = tf.Session(graph=actor)
        self.actor_sess.run(init)

        with tf.Graph().as_default() as critic:
            critic_weights = {
                'h1': tf.Variable(tf.random_uniform([self.state_dim, 
                    self.hidden_units], minval=0, maxval=.01)),
                'h2': tf.Variable(tf.random_uniform([self.hidden_units, 
                    self.hidden_units], minval=0, maxval=.01)),
                'h3': tf.Variable(tf.random_uniform([self.hidden_units,
                    self.hidden_units], minval=0, maxval=.01)),
                'out': tf.Variable(tf.random_uniform([self.hidden_units, 1],
                    minval=0, maxval=.01))
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
        
    def _aggregate(self, trajs, rewards, actions, times, states_t, \
            r_t, a_t, t_t, ids_t,ids_idx):
        for i in range(len(states_t)):
            car_id = ids_t[i];
            idx = ids_idx[car_id];
            trajs[car_id][idx] = states_t[i]
            rewards[car_id][idx] = r_t[i]
            actions[car_id][idx] = a_t[i]
            times[car_id][idx] = t_t;
            ids_idx[car_id] += 1;

    def update_animation(self,imaging_data, img_idx):
        loc_c = imaging_data[0];
        num_cars = imaging_data[1];
        loc_r = imaging_data[2];
        num_reqs = imaging_data[3];
        car_points = plt.scatter(loc_c[0],loc_c[1],num_cars,color='r',zorder=5);
        req_points = plt.scatter(loc_r[0],loc_r[1],num_reqs,color='b',zorder=4);
        plt.draw();
        plt.savefig("movies/img" + str(img_idx) +".png", bbox_inches='tight', dpi = 220);
        img_idx += 1;

        car_points.remove();
        req_points.remove();

    def create_animation(self,imaging_data, epoch, tstart, tend):
        img_idx = 0;
        plt.savefig("movies/img" + str(img_idx) +".png", bbox_inches='tight', dpi = 220);
        img_idx += 1;


        plot_data = {};
        for t in sorted(imaging_data.keys()):
            lat_c = imaging_data[t][0]
            lon_c = imaging_data[t][1]
            loc_c = self.m(lon_c,lat_c)
            num_c = imaging_data[t][2];

            lat_r = imaging_data[t][3]
            lon_r = imaging_data[t][4]
            loc_r = self.m(lon_r,lat_r)
            num_r = imaging_data[t][5]

            plot_data[t] = (loc_c,num_c,loc_r,num_r);
            self.update_animation(plot_data[t],img_idx)
            img_idx += 1;
        plt.draw();

        os.system("ffmpeg -r 1 -i movies/img%d.png -vcodec mpeg4 -y movies/sf_"+str(tstart)+"_to_"+str(tend)+"_run_"+str(epoch)+".mp4")
        
    def init_animation(self):
        self.fig = plt.figure(1)
        plt.ion();
        self.m = Basemap(projection = 'stere', 
                    llcrnrlat=37.765,urcrnrlat=37.81,
                    llcrnrlon=-122.445,urcrnrlon=-122.385,
                    resolution='f',
                    lat_0 = 37.78,lon_0 = -122.41)
        self.m.drawmapboundary(fill_color='aqua',zorder=1)
        self.m.fillcontinents(color='lightgreen',zorder=2)
        self.m.readshapefile("sf_road_shapefile/geo_export_bca4a474-0dad-4589-b7c2-f325f80f9119","sf_roads",zorder=3);
        plt.title("Downtown San Francisco");
        plt.xlabel("East-West");
        plt.ylabel("North-South");
        plt.draw();


    def train(self):
        max_epochs = 60
        rewards_train = [0] * max_epochs;
        rewards_test = [0] * max_epochs;
        costs = [0] * max_epochs
        fname = str(uuid.uuid4())
        train_out = open(fname + '.train', 'w')
        test_out = open(fname + '.test', 'w')
        for epoch in range(max_epochs):
            start_t = np.random.choice(self.train_windows, 1)[0] 
            self.sim.reset(start_t)

            num_ts = self.sim.end_t - self.sim.start_t;
            # classes should be total number of cars in simulation
            # this should be safe though
            num_cars = 0;
            ids_idx = [0] * self.sim.classes;
            trajs = [[]] * self.sim.classes;
            rewards = [[]] * self.sim.classes;
            actions = [[]] * self.sim.classes;
            times = [[]] * self.sim.classes;
            for i in range(self.sim.classes):
                trajs[i] = [0] * num_ts;
                rewards[i] = [0] * num_ts;
                actions[i] = [0] * num_ts;
                times[i] = [0] * num_ts;

            imaging_data = {};

            self.init_animation();
            
            #beginning of an episode run 
            for t in range(self.sim.start_t, self.sim.end_t): #self.sim.end_t

                pmr_t = t - self.sim.start_t
                p_t = self.actor_sess.run(self.actor_out_layer,
                        feed_dict={self.actor_states: self.sim.get_states()})
                a_t = [-1] * len(p_t);
                for j in range(len(p_t)):
                    a_t[j] = np.random.choice(self.action_dim, 1, p=p_t[j])[0]
                # obtain actions for previously (p) matched (m) rides (r) 
                    
                pmr_a_t = [];
                if self.sim.pmr_index[pmr_t] > 0:
                    pmr_p_t = self.actor_sess.run(self.actor_out_layer, 
                            feed_dict={\
                                    self.actor_states: self.sim.get_pmr_states(t)})

                    pmr_a_t = [-1] * len(pmr_p_t);
                    for j in range(len(pmr_p_t)):
                        pmr_a_t[j] = np.random.choice(self.action_dim, 
                                1, p=pmr_p_t[j])[0]

                states_t = self.sim.get_states();
                ids_t = self.sim.curr_ids[:self.sim.curr_index]
                num_cars = max(max(ids_t),num_cars);

                num_ids = len(ids_t);
                if self.sim.pmr_index[pmr_t] > 0:
                    num_ids += len(self.sim.pmr_ids[pmr_t][:self.sim.pmr_index[pmr_t]]);
                train_out_str = "i, %d, %d\n" % (t, num_ids)
                print('train ', train_out_str[:-1])
                train_out.write(train_out_str)
                
                lat_c = [-1] * (self.sim.curr_index+self.sim.pmr_index[pmr_t]);
                lon_c = [-1] * (self.sim.curr_index+self.sim.pmr_index[pmr_t]);
                num_c = [-1] * (self.sim.curr_index+self.sim.pmr_index[pmr_t]);
                curr_cars = Counter(self.sim.curr_nodes[:self.sim.curr_index] + \
                                    self.sim.pmr_dropoffs[pmr_t][:self.sim.pmr_index[pmr_t]])
                new_size = self._add_lat_lng_cars(lat_c, lon_c, num_c, curr_cars)
                lat_c = lat_c[:new_size];
                lon_c = lon_c[:new_size];
                num_c = num_c[:new_size];


                lat_r = [-1] * self.sim.classes;
                lon_r = [-1] * self.sim.classes;
                num_r = [-1] * self.sim.classes;
                num_reqs = np.array(self.sim.curr_req_size) - np.array(self.sim.curr_req_index);
                reqs = np.argwhere(num_reqs>0).flatten().tolist()
                new_size = self._add_lat_lng_reqs(lat_r, lon_r, num_r, reqs, num_reqs)
                lat_r = lat_r[:new_size];
                lon_r = lon_r[:new_size];
                num_r = num_r[:new_size];
            
                imaging_data[pmr_t] = (lat_c,lon_c,num_c,lat_r,lon_r,num_r);

                # step in the enviornment
                r_t = self.sim.step(a_t, pmr_a_t)

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
                            self.sim.pmr_ids[pmr_t][:self.sim.pmr_index[pmr_t]],ids_idx)
                assert (len(states_t) + len(pmr_a_t)) == len(r_t)  
                assert (len(ids_t) + len(pmr_a_t)) == len(r_t)
                
                

            #end of an episode run and results aggregated

            for car_id in range(num_cars):
                idx = ids_idx[car_id];
                trajs[car_id] = trajs[car_id][:idx];
                rewards[car_id] = rewards[car_id][:idx];
                actions[car_id] = actions[car_id][:idx];
                times[car_id] = times[car_id][:idx];
            
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
            rewards_test[epoch] = self.test();
            print('test rewards', rewards_test[epoch])
            
            train_out_str = "e, %d, %.2f, %.2f, %.2f\n" % (epoch, 
                    np.sum(temp_r), costs[-1], np.sum(np.abs(temp_c)))
            train_out.write(train_out_str)
            
            test_out_str = "%d, %.2f\n" % (epoch, rewards_test[epoch])
            test_out.write(test_out_str)
            
            test_out.flush()
            train_out.flush()
        train_out.close()
        test_out.close()
        """
        fig = plt.figure(1)
        ax1 = fig.add_subplot(1,1,1)
        plt.title('Training Results');
        ax1.plot(costs, color='b--', linewidth=1)
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        ax2 = ax1.twinx()
        ax2.plot(rewards_train, 'r-', linewidth=1)
        plt.ylabel('Reward')
        plt.show();
        '''
        fig = plt.figure(2)
        plt.title('Testing Results')
        plt.xlabel('Epochs');
        plt.ylabel('Reward');
        plt.plot(rewards_test,'r-',linewidth=1);
        plt.show();
        '''
        """

    def test(self):
        start_t = self.test_window
        self.sim.reset(start_t)
        rewards = [0] * (self.sim.end_t - self.sim.start_t);

        for t in range(self.sim.start_t, self.sim.end_t):

            pmr_t = t - self.sim.start_t;
            p_t = self.actor_sess.run(self.actor_out_layer,
                    feed_dict={self.actor_states: self.sim.get_states()})
            
            a_t = [-1] * len(p_t);
            for j in range(len(p_t)):
                a_t[j] = np.random.choice(self.action_dim, 1, p=p_t[j])[0]
            
            pmr_a_t = []
            if self.sim.pmr_index[pmr_t] > 0:
                pmr_p_t = self.actor_sess.run(self.actor_out_layer, 
                        feed_dict={\
                                self.actor_states: self.sim.get_pmr_states(t)})
                pmr_a_t = [-1] * len(pmr_p_t);
                for j in range(len(pmr_p_t)):
                    pmr_a_t[j] = np.random.choice(self.action_dim,
                            1, p=pmr_p_t[j])[0]


            states_t = self.sim.get_states()
            ids_t = self.sim.curr_ids[:self.sim.curr_index]
            num_ids = len(ids_t);
            
            if self.sim.pmr_index[pmr_t] > 0:
                num_ids += len(self.sim.pmr_ids[pmr_t][:self.sim.pmr_index[pmr_t]]);
            print("ts %d, ids %d" % (t, num_ids))
            
            r_t = self.sim.step(a_t, pmr_a_t)
            rewards[pmr_t] = np.sum(r_t)
        return np.sum(rewards)
