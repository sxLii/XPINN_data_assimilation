import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import time
import tensorflow as tf

np.random.seed(124)
tf.set_random_seed(134)

class XPINNs:
    def __init__(self, pipe_number,x,t,x0,t0,h0,q0,lb,ub,
                 node_number,node_index,inner_index,xi,ti,qiflow,
                 a,D,fric,
                 layers_number,layers,
                 w1,w2,w3):

        self.useInt=True
        if node_number==0:
            self.useInt=False

        # constant
        self.a = a
        self.D = D
        self.fric = fric
        self.pipe_number=pipe_number
        self.node_number=node_number
        self.node_index=node_index
        # pipe
        self.x, self.t = [], []
        self.x0, self.t0 = [], []
        self.h0, self.q0 = [], []
        self.lb, self.ub = lb,ub
        for i in range(self.pipe_number):
            self.x.append(x[i].flatten()[:, None])
            self.t.append(t[i].flatten()[:, None])
            self.x0.append(x0[i].flatten()[:, None])
            self.t0.append(t0[i].flatten()[:, None])
            self.h0.append(h0[i].flatten()[:, None])
            self.q0.append(q0[i].flatten()[:, None])

        # interface
        if self.useInt==True:
            self.xi = [[] for _ in node_index]
            self.ti = [[] for _ in node_index]
            self.qiflow = []

            for i in range(self.node_number):  # 0,1,2
                for j, j_inx in enumerate(node_index[i]):
                    self.xi[i].append(xi[i][j].flatten()[:, None])
                    self.ti[i].append(ti[i][j].flatten()[:, None])
                self.qiflow.append(qiflow[i].flatten()[:, None])

        # layers
        self.layers = layers

        # Initialize NN
        self.weight = []
        self.bias = []
        for i in range(layers_number):
            wt, bt = self.initialize_NN(layers[i])
            self.weight.append(wt)
            self.bias.append(bt)


        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        # save weight and bias
        self.saver = tf.train.Saver()

        self.x1_tf,self.t1_tf,self.x01_tf,self.t01_tf=[],[],[],[]
        self.x2_tf, self.t2_tf,self.x02_tf, self.t02_tf = [], [],[],[]
        for i in range(self.pipe_number):
            # for h
            self.x1_tf.append(tf.placeholder(tf.float32, shape=[None, self.x[i].shape[1]]))
            self.t1_tf.append(tf.placeholder(tf.float32, shape=[None, self.t[i].shape[1]]))
            self.x01_tf.append(tf.placeholder(tf.float32, shape=[None, self.x0[i].shape[1]]))
            self.t01_tf.append(tf.placeholder(tf.float32, shape=[None, self.t0[i].shape[1]]))

            # for q
            self.x2_tf.append(tf.placeholder(tf.float32, shape=[None, self.x[i].shape[1]]))
            self.t2_tf.append(tf.placeholder(tf.float32, shape=[None, self.t[i].shape[1]]))
            self.x02_tf.append(tf.placeholder(tf.float32, shape=[None, self.x0[i].shape[1]]))
            self.t02_tf.append(tf.placeholder(tf.float32, shape=[None, self.t0[i].shape[1]]))

        if self.useInt==True:
            self.xi_h_tf = [[] for _ in node_index]
            self.ti_h_tf = [[] for _ in node_index]
            self.xi_q_tf = [[] for _ in node_index]
            self.ti_q_tf = [[] for _ in node_index]

            for i in range(self.node_number): # 0,1,2
                for j,j_inx in enumerate(self.node_index[i]):
                    self.xi_h_tf[i].append( tf.placeholder(tf.float32, shape=[None, self.xi[i][j].shape[1]]))
                    self.ti_h_tf[i].append( tf.placeholder(tf.float32, shape=[None, self.ti[i][j].shape[1]]))
                    self.xi_q_tf[i].append( tf.placeholder(tf.float32, shape=[None, self.xi[i][j].shape[1]]))
                    self.ti_q_tf[i].append( tf.placeholder(tf.float32, shape=[None, self.ti[i][j].shape[1]]))



        # prediction
        self.h0_pred,self.q0_pred=[],[]
        self.f0_pred, self.g0_pred = [], []
        self.f_pred, self.g_pred = [], []
        for i in range(self.pipe_number):
            h0pred,q0pred,f0pred,g0pred=self.net_NS(i,self.x01_tf[i], self.t01_tf[i],self.x02_tf[i], self.t02_tf[i])
            _, _, fpred, gpred = self.net_NS(i,self.x1_tf[i], self.t1_tf[i], self.x2_tf[i], self.t2_tf[i])
            self.h0_pred.append(h0pred)
            self.q0_pred.append(q0pred)
            self.f0_pred.append(f0pred)
            self.g0_pred.append(g0pred)
            self.f_pred.append(fpred)
            self.g_pred.append(gpred)

        # interface
        if self.useInt==True:
            self.hi,self.qi=[[] for _ in node_index],[[] for _ in node_index]
            self.fi,self.gi=[[] for _ in node_index],[[] for _ in node_index]
            for i in range(self.node_number): # 0,1,2
                for j,j_inx in enumerate(self.node_index[i]): # 1,2,3
                    hit,qit,fit,git=self.net_NS(j_inx-1,self.xi_h_tf[i][j], self.ti_h_tf[i][j],self.xi_q_tf[i][j], self.ti_q_tf[i][j])
                    self.hi[i].append(hit)
                    self.qi[i].append(qit)
                    self.fi[i].append(fit)
                    self.gi[i].append(git)

        # weight
        self.weight1=w1
        self.weight2=w2
        self.weight3=w3

        for i in range(self.pipe_number):
            w3t=tf.reduce_sum(tf.square(self.h0[i] - self.h0_pred[i]))/tf.reduce_sum(tf.square(self.q0[i] - self.q0_pred[i]))
            w3t=tf.clip_by_value(w3t, clip_value_min=1e-4, clip_value_max=1e4)
            self.weight3[i]=0.9*w3[i]+0.1*w3t
            w3[i]=w3t
        
        # calculation loss
        losspde1,losspde2,losspde3=0,0,0
        lossobs=0
        lossint=0
        for i in range(self.pipe_number):
            losspde1 += tf.square(self.f0_pred[i])+tf.square(self.g0_pred[i])
            losspde2 += tf.square(self.f_pred[i])+tf.square(self.g_pred[i])


            lossobs += tf.square(self.h0[i] - self.h0_pred[i]) + self.weight3[i] * tf.square(self.q0[i] - self.q0_pred[i])

        if self.useInt==True:
            for i in range(self.node_number):  # 0,1,2
                inter_h=0
                inter_q=0
                for j, j_inx in enumerate(inner_index[i]):
                    losspde3 += tf.square(self.fi[i][j])+tf.square(self.gi[i][j])
                    if j!=0:
                        inter_h += tf.square(self.hi[i][0]-self.hi[i][j])

                    if j_inx==0:
                        inter_q -= self.qi[i][j]
                    else:
                        inter_q += self.qi[i][j]

                inter_q -=self.qiflow[i]

                lossint += tf.reduce_sum(inter_h+tf.square(inter_q))

        self.loss_pde =  tf.reduce_sum(losspde1)+tf.reduce_sum(losspde2)#+tf.reduce_sum(losspde3)
        if self.useInt==True:
            self.loss_pde += tf.reduce_sum(losspde3)

        self.loss_observation = tf.reduce_sum(lossobs)

        self.loss = self.weight1 * self.loss_pde  + self.loss_observation

        if self.useInt==True:
            self.loss_interface = lossint
            self.loss += self.weight2 * self.loss_interface
        # self.loss = self.weight1*self.loss_pde + self.weight2*self.loss_interface+self.loss_observation
        #
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 500000,
                                                                         'maxfun': 500000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss) # get_zeros

        init = tf.global_variables_initializer()
        self.sess.run(init)


    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32,name='Bias')
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32,name='Weight')

    def neural_net(self, X, weights, biases):
        # forward propagation, from x to Y
        num_layers = len(weights) + 1
        H=X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))  # update H
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_NS(self,i, x1, t1, x2, t2):
        # pipe
        g = 9.81
        D = self.D[i]
        A = 3.14 * D * D / 4
        fric = self.fric[i]
        a = self.a[i]
        # normalization
        Xh = 2.0 * (tf.concat([x1, t1],1) - self.lb[i]) / (self.ub[i] - self.lb[i]) - 1.0
        Xq = 2.0 * (tf.concat([x2, t2], 1) - self.lb[i]) / (self.ub[i] - self.lb[i]) - 1.0

        # forward propagation
        h_ = self.neural_net(Xh, self.weight[2*i], self.bias[2*i])
        q_ = self.neural_net(Xq, self.weight[2*i+1], self.bias[2*i+1])

        # partial derivatives from the DNN
        h_t = tf.gradients(h_, t1,name='h_t')[0]
        h_x = tf.gradients(h_, x1,name='h_x')[0]
        q_t = tf.gradients(q_, t2,name='q_t')[0]
        q_x = tf.gradients(q_, x2,name='q_x')[0]

        ff = A * q_t + q_ * q_x + g * A * A * h_x+fric*q_*tf.abs(q_)/2/D
        gf = A * h_t + q_ * h_x + a * a * q_x / g
        return h_, q_, ff, gf


    def callback(self, loss,loss_pde, loss_observation):
        print('Loss: %.3e, Loss_pds: %.3e,Loss_obs: %.3e' % (loss,loss_pde, loss_observation))


    def train(self, nIter):
        self.loss_pdes=[]
        self.loss_observaitons=[]
        self.loss_interfaces=[]
        self.losses=[]
        tf_dict = {
            **{placeholder: data for placeholder, data in zip(self.x1_tf, self.x)},
            **{placeholder: data for placeholder, data in zip(self.t1_tf, self.t)},
            **{placeholder: data for placeholder, data in zip(self.x01_tf, self.x0)},
            **{placeholder: data for placeholder, data in zip(self.t01_tf, self.t0)},
            **{placeholder: data for placeholder, data in zip(self.x2_tf, self.x)},
            **{placeholder: data for placeholder, data in zip(self.t2_tf, self.t)},
            **{placeholder: data for placeholder, data in zip(self.x02_tf, self.x0)},
            **{placeholder: data for placeholder, data in zip(self.t02_tf, self.t0)},
        }

        # Update tf_dict with xi and ti data
        if self.useInt==True:
            for i in range(self.node_number):
                tf_dict.update({
                    placeholder: data for placeholder, data in zip(self.xi_h_tf[i], self.xi[i])
                })
                tf_dict.update({
                    placeholder: data for placeholder, data in zip(self.ti_h_tf[i], self.ti[i])
                })
                tf_dict.update({
                    placeholder: data for placeholder, data in zip(self.xi_q_tf[i], self.xi[i])
                })
                tf_dict.update({
                    placeholder: data for placeholder, data in zip(self.ti_q_tf[i], self.ti[i])
                })

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 100 == 0:
                elapsed = time.time() - start_time

                loss_pde_value = self.sess.run(self.loss_pde, tf_dict)
                loss_obs_value = self.sess.run(self.loss_observation, tf_dict)
                if self.useInt==True:
                    loss_int_value = self.sess.run(self.loss_interface, tf_dict)

                loss_value = self.sess.run(self.loss, tf_dict)
                self.loss_pdes+=[ loss_pde_value ]
                self.loss_observaitons+=[loss_obs_value ]
                self.losses+=[ loss_value ]
                print('loss_pde',loss_pde_value)
                print('loss_obs',loss_obs_value)


                if self.useInt==True:
                    self.loss_interfaces += [loss_int_value]
                    print('loss_int',loss_int_value)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.loss_pde, self.loss_observation],
                                loss_callback=self.callback)

        self.saver.save(self.sess, save_path='TwoNN_saved_model/model')

    def predict(self,i, x_star, t_star):
        x_star=x_star
        t_star=t_star
        tf_dict = {self.x01_tf[i]: x_star, self.t01_tf[i]: t_star,
                   self.x02_tf[i]: x_star, self.t02_tf[i]: t_star}
        h_star = self.sess.run(self.h0_pred[i], tf_dict)
        q_star = self.sess.run(self.q0_pred[i], tf_dict)
        return h_star, q_star
