from __future__ import division
import tensorflow as tf, numpy as np, sys
import logging
import matplotlib.pyplot as plt
import pandas as pd

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

        num_mins_per_bin = 2
        num_bins_in_hour = 60 / num_mins_per_bin
        num_hours_per_day = 24
        self.data_for_day = num_bins_in_hour * num_hours_per_day

        self.setup_actor_critic()
        #init = tf.global_variables_initializer()
        #self.sess = tf.Session()
        #self.sess.run(init)

    def setup_actor_critic(self):
        with tf.Graph().as_default() as actor:
            actor_weights = {
                'h1': tf.Variable(tf.random_normal([self.state_dim, 
                    self.hidden_units])),
                'h2': tf.Variable(tf.random_normal([self.hidden_units, 
                    self.hidden_units])),
                'h3': tf.Variable(tf.random_normal([self.hidden_units, 
                    self.hidden_units])),
                'out': tf.Variable(tf.random_normal([self.hidden_units, 
                    self.action_dim]))
                }

            actor_biases = {
                'b1': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'b2': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'b3': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'out': tf.Variable(tf.constant(0.0, shape=[self.action_dim]))
                }

            actor_states = tf.placeholder("float", [None, self.state_dim])
            values = tf.placeholder("float", [None, self.action_dim])

            actor_l1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(actor_states, 
                actor_weights['h1']), actor_biases['b1']))
            actor_l2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(actor_l1, 
                actor_weights['h2']), actor_biases['b2']))
            actor_l3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(actor_l2, 
                actor_weights['h3']), actor_biases['b3']))
            actor_out_layer = tf.nn.softmax(tf.nn.bias_add(\
                    tf.matmul(actor_l3, actor_weights['out']), 
                    actor_biases['out']))

            actor_loss_op = tf.reduce_mean(actor_out_layer)
            actor_optimizer = tf.train.AdamOptimizer(self.actor_alpha)
            actor_train_op = actor_optimizer.minimize(\
                    actor_loss_op)
            init = tf.global_variables_initializer()
        
        self.actor_sess = tf.Session(graph=actor)
        self.actor_sess.run(init)


        with tf.Graph().as_default() as critic:
            critic_weights = {
                'h1': tf.Variable(tf.random_normal([self.state_dim, 
                    self.hidden_units])),
                'h2': tf.Variable(tf.random_normal([self.hidden_units, 
                    self.hidden_units])),
                'h3': tf.Variable(tf.random_normal([self.hidden_units,
                    self.hidden_units])),
                'out': tf.Variable(tf.random_normal([self.hidden_units, 1]))
                }
            
            critic_biases = {
                'b1': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'b2': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'b3': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'out': tf.Variable(tf.constant(0.0, shape=[1]))
                }
            
            critic_states = tf.placeholder("float", [None, self.state_dim])
            critic_value = tf.placeholder("float", [None])
           
            critic_l1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(critic_states, 
                critic_weights['h1']), critic_biases['b1']))
            critic_l2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(critic_l1, 
                critic_weights['h2']), critic_biases['b2']))
            critic_l3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(critic_l2, 
                critic_weights['h3']), critic_biases['b3']))
            critic_out_layer = tf.matmul(critic_l3, 
                    critic_weights['out']) + critic_biases['out']
            
            critic_loss_op = tf.reduce_mean(tf.square(\
                    tf.subtract(critic_value, critic_out_layer)));
            critic_optimizer = tf.train.AdamOptimizer(self.critic_alpha)
            critic_train_op = critic_optimizer.\
                    minimize(critic_loss_op)
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

    def train(self):
        costs = [];
        rewards = [];
        episodes_per_actor_update = 10;
        max_epochs = 100;
        for epoch in range(max_epochs):
            N = 5;
            cost = 0;
            iters = 0;
            num_time_steps = 50;

# correctly request values for currpd
#            curr_state = self.sim.get_state_rep(epoch) 
            curr_pd = pd.DataFrame({"car_ids":[1,2,3,4,5], "rewards":[0,0,1,0.8,0],
                                       "states":[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]});
            all_pd = curr_pd.copy()
            for i in range(num_time_steps):
                curr_state = curr_pd.states.as_matrix().tolist();
                pi_t = self.sess.run(self.actor_out_layer,
                        feed_dict={self.actor_states: curr_state});
                a_t = [];
                for j in range(len(pi_t)):
                    prob = abs(pi_t[j]) / sum(abs(pi_t[j]));
                    a_t.append(np.random.choice(self.action_dim,1,False,prob).tolist());
                    
                curr_pd['steps'] = i; 
#                curr_pd['policy'] = pi_t.tolist();  # i don't think you need this...
                curr_pd['actions'] = a_t;
                curr_pd['terminal'] = np.random.randint(0,2,5).tolist(); # fix with correct implementation
#                next_state = self.sim.get_state_rep(epoch+1) # get the values from env correctly
#                all_pd.append(curr_pd); # will want this
                all_pd = curr_pd.copy(); # comment this out / delete 

#                curr_state = next_state; #update state
            
                car_ids = np.unique(curr_pd.car_ids.as_matrix()).tolist()
                for car in car_ids:
                    if (curr_pd[curr_pd.car_ids == car].terminal.values[0] == True):
                        car_pd = all_pd[all_pd.car_ids == car];
                        rewards = car_pd.rewards.as_matrix();
                        map = ~np.isnan(rewards);
                        rewards = rewards[map].tolist();
                        discount = [];
                        for j in range(len(rewards)):
                            discount.append(rewards[j]);
                            for k in range(len(rewards[j+1:])):
                                discount[j] += discount[k] * self.gamma**k;
                                
                        states = car_pd.states.as_matrix()
                        states = states[map].tolist();
                        _, c = self.sess.run([self.critic_train_op, self.critic_loss_op], 
                                             feed_dict={self.critic_states: states, 
                                                        self.critic_reward: discount})
                    
            print(epoch)
                
            car_ids = np.unique(all_pd.car_ids.values).tolist();
            for car in car_ids:
                if (all_pd[all_pd.car_ids == car].terminal == 1).any():
                    car_pd = all_pd[all_pd.car_ids == car];
                    rewards = car_pd.rewards.as_matrix();
                    map = ~np.isnan(rewards);
                    rewards = rewards[map].tolist();
                    discount = [];
                    steps = range(len(rewards));
                    for j in steps:
                        discount.append(rewards[j]);
                        for k in steps[j+1:]:
                            discount[j] += discount[k] * self.gamma**k;
                    states = car_pd.states.as_matrix()[map].tolist();
                    critic_value=self.sess.run(self.critic_out_layer,feed_dict={self.critic_states: states}).flatten()
                    values = (discount - critic_value);
                    actions = car_pd.actions.as_matrix()[map].tolist();
                    one_hot_values = np.zeros([len(actions),self.action_dim]);
                    for j in steps:
                        one_hot_values[j,actions[j]] = values[j];
                    _, c = self.sess.run([self.actor_train_op, self.actor_loss_op], 
                                         feed_dict={self.actor_states: states, 
                                                    self.actor_values: one_hot_values});

#            logging.debug("train: epoch %d, time hour %d, cost %.4f" % 
#                          (epoch, self.sim.time_utils.get_hour_of_day(epoch), cost))
            costs.append(cost)
#            rewards.append(self.test())

        fig = plt.figure(1)
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(costs, color='r', linewidth=1)
        plt.xlabel('epochs')
        plt.ylabel('cost')
        ax2 = ax1.twinx()
        ax2.plot(rewards, 'b--', linewidth=1)
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
