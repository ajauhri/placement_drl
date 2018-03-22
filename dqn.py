from __future__ import division
import tensorflow as tf, numpy as np, sys
import logging

class Replay_Memory():
    def __init__(self, memory_size=5000, burn_in=1000):
    
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
    def __init__(self, state_dim, action_dim, hidden_units, sim, n_time_bins):
        self.mem = Replay_Memory()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.sim = sim
        self.n_time_bins = n_time_bins
        self.alpha = 0.0001
        self.gamma = 1
        self.epsilon = 0.5
        
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
            while True:
                ts = np.random.randint(720)
                curr_node = self.sim.get_random_node(ts)
                if curr_node != -1:
                    break
            a = self.sim.sample_action_space()
            next_node, r = self.sim.step(curr_node, ts, a)
            curr_state = self.sim.get_state_rep(curr_node, ts)
            next_state = self.sim.get_state_rep(next_node, ts+1)
            self.mem.append([curr_state, r, a, next_state])

    def epsilon_greedy(self, q_values):
        if np.random.rand() <= self.epsilon:
            return self.sim.sample_action_space()
        else:
            return np.argmax(q_values)
 
    def train(self):
        self.burn_in_memory()
        rewards = []
        costs = []
        for epoch in range(self.n_time_bins)[:720]:
            old_replay = []
            old_q = []
            cost = 0
            iters = 0
            
            nodes = self.sim.get_random_nodes(epoch, 50) 
            if not len(nodes):
                continue

            for i in range(len(nodes)):
                old_replay.append(self.mem.sample_batch())
                x_j_plus_1 = np.array(old_replay[-1][:, 3].flatten().tolist())
                q_j_plus_1 = self.sess.run(self.out_layer,
                        feed_dict={self.X: x_j_plus_1})
                old_q.append(q_j_plus_1)
            
            while iters < len(nodes):
                curr_state = self.sim.get_state_rep(nodes[iters], epoch)
                a = self.epsilon_greedy(self.sess.run(self.out_layer,
                        feed_dict={self.X: [curr_state]}))
                next_node, r = self.sim.step(nodes[iters], epoch, a)
                next_state = self.sim.get_state_rep(next_node, epoch+1) 
                #tot_r += r
                
                self.mem.append([curr_state, r, a, next_state])
                
                x_j = np.array(old_replay[iters][:, 0].flatten().tolist())
                r_j = old_replay[iters][:, 1]
                a_j = old_replay[iters][:, 2]
                x_j_plus_1 = np.array(old_replay[iters][:, 3].flatten().tolist())
                q_j_plus_1 = old_q[iters]
                q_j = self.sess.run(self.out_layer, feed_dict={self.X: x_j})

                # SGD updates to model using the replay memory
                for k in range(len(x_j)):
                    target_q = [0]*self.action_dim
                    for not_taken in range(self.action_dim):
                        if not_taken != a_j[k]:
                            target_q[not_taken] = q_j[k][not_taken]
                    
                    target_q[a_j[k]] = r_j[k] + \
                            self.gamma * np.max(q_j_plus_1[k])
                    _, c = self.sess.run([self.train_op, self.loss_op], 
                                      feed_dict={self.X: [x_j[k]], 
                                                 self.y: [target_q]})
                    cost += c
                iters += 1
            logging.debug("train: epoch %d, time hour %d,  cost %.4f" % (epoch, 
                self.sim.time_utils.get_hour_of_day(epoch),
                cost))
            costs.append(cost)
            rewards.append(self.test())

        import matplotlib.pyplot as plt
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
