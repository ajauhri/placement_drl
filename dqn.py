from __future__ import division
import tensorflow as tf, numpy as np, sys

class Replay_Memory():
    def __init__(self, memory_size=50000, burn_in=10000):
    
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.
        self.M = []
        self.memory_size = memory_size
        self.burn_in = burn_in

    def sample_batch(self, batch_size=100):
        idxs = np.random.choice(range(len(self.M)), batch_size, replace=False)
        return np.array(self.M)[idxs]

    def append(self, transition):
        # Appends transition to the memory. 	
        if len(self.M) == self.memory_size:
            self.M = self.M[1:]
        self.M.append(transition)


class DQN:
    def __init__(self, state_dim, action_dim, hidden_units, sim, n_time_bins,
            geo_utils):
        self.mem = Replay_Memory()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.sim = sim
        self.n_time_bins = n_time_bins
        self.geo_utils = geo_utils
        self.alpha = 0.0001
        
        self.weights = {
                'h1': tf.Variable(tf.random_normal([self.state_dim,
                    self.hidden_units])),
                'h2': tf.Variable(tf.random_normal([self.hidden_units, 
                    self.hidden_units])),
                'h3': tf.Variable(tf.random_normal([self.hidden_units,
                    self.hidden_units])),
                'out': tf.Variable(tf.random_normal([self.hidden_units,
                    self.action_dim]))
                }

        self.biases = {
                'b1': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'b2': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'b3': tf.Variable(tf.constant(0.0, shape=[self.hidden_units])),
                'out': tf.Variable(tf.constant(0.0, shape=[self.action_dim]))
                }

        self.X = tf.placeholder("float", [None, self.state_dim])
        self.y = tf.placeholder("float", [None, self.action_dim])
        self.out_layer = self.perceptron(self.X)
        self.loss_op = tf.reduce_mean(tf.squared_difference(self.y, 
            self.out_layer))
        self.optimizer = tf.train.AdamOptimizer(self.alpha)
        self.train_op = self.optimizer.minimize(self.loss_op)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def perceptron(self, X):
        l1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(X, self.weights['h1']), 
            self.biases['b1']))
        l2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(l1, self.weights['h2']), 
            self.biases['b2']))
        l3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(l2, self.weights['h3']), 
            self.biases['b3']))
        logits = tf.matmul(l3, self.weights['out']) + self.biases['out']
        return logits
    
    def burn_in_memory(self):
        """
        To burn-in for replay memory for some time snapshots
        """
        while len(self.mem.M) < self.mem.burn_in:
            ts = np.random.randint(self.n_time_bins)
            curr_node = self.sim.get_random_node(ts)
            a = self.sim.sample_action_space()
            next_node, next_th, r, curr_th = self.sim.step(curr_node, ts, a)
            curr_state = self.geo_utils.get_centroid(curr_node) + [curr_th]
            next_state = self.geo_utils.get_centroid(next_node) + [next_th]
            self.mem.append([curr_state, r, a, next_state])
 
    def train(self):
        self.burn_in_memory()
        return 
        for epoch in range(self.params['iters']):
            old_replay = []
            old_q = []
            for i in range(self.max_iters):
                x_j, x_j_plus_1, r_j, d_j, a_j = \
                        self.prepare_batch_data_for_percep()
                old_replay.append([ x_j, x_j_plus_1, r_j, d_j, a_j])
                q_j_plus_1 = self.sess.run(self.out_layer,
                        feed_dict={self.X: x_j_plus_1})
                old_q.append(q_j_plus_1)
             
            while iters <= self.max_iters:
                a = self.epsilon_greedy_policy(self.sess.run(self.out_layer,
                        feed_dict={self.X: [curr_s]}))
                
                next_s, r, done, _ = self.env.step(a)
                tot_r += r
                
                self.mem.append([curr_s, r, a, next_s, done]) 
                curr_s = next_s
                
                x_j, x_j_plus_1, r_j, d_j, a_j = old_replay[iters]
                q_j_plus_1 = old_q[iters]
                q_j = self.sess.run(self.out_layer, feed_dict={self.X: x_j})

                # SGD updates to model using the replay memory
                for i in range(self.batch_size):
                    target_q = [0]*self.action_dim
                    for not_taken in range(self.action_dim):
                        if not_taken != a_j[i]:
                            target_q[not_taken] = q_j[i][not_taken]
                    
                    if d_j[i]:
                        target_q[a_j[i]] = r_j[i]
                    else:
                        target_q[a_j[i]] = r_j[i] + \
                                self.params[self.env_name]['gamma'] * \
                                                        np.max(q_j_plus_1[i])
                    _, c = self.sess.run([self.train_op, self.loss_op], 
                                      feed_dict={self.X: [x_j[i]], 
                                                 self.y: [target_q]})
                    cost.append(c)
                iters += 1
                if done:
                    break
