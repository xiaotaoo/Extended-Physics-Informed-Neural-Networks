import numpy as np
import time
from pyDOE import lhs
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import scipy.io
import random
import pandas as pd
import shutil
import os
from sklearn.model_selection import train_test_split

#flow past cylinder with uniform U, upper and lower wall BC
#full u,v specified BC
#top, bottom u=1, v=0
#domain size xy = -5,-5 to 9.333/5
#added p_outlet
#seperate WALL into actual cylinder wall, TOP_BOTTOM BC -< not done yet
#also use internal pts for training

# Setup GPU for training (use tensorflow v1.9 for CuDNNLSTM)
import tensorflow as tf

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;

# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")
    
#     sys.exit()

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

data_ref = 'fluent'
use_old_model = 0
debug = 0
#activation function = 0 (tanh), 1 (ReLU), 2 (LeakyReLU), 3 (SELU)
activation_fn = 0
test_size = 0.75
#compare diff = absolute or percentage or mean sq error
diff = 'absolute'
#whether to use OF internal pts for training
use_internal = 0
beta = 6
CFD_grid = 0

#used for force calculations
surface_pts = 200
rho = 1.0
mu = 0.025
Re=40
consolidate_collo=2
use_neural_form =1
optimizer = 0 #0 for adam, 1 for yellowfin

delta_dist = 1e-4

consolidate_collo=0

def fwd_gradients(Y, x):
        
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x

def Gradient_Velocity_2D(u, v, x, y):
        
    Y = tf.concat([u, v], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    
    return [u_x, v_x, u_y, v_y]

class XPINN_laminar_flow:
    # Initialize the class
    def __init__(self, collo1, collo2, interface, INLET, internal, OUTLET, CYLD, WALL, uv_layers, lb, ub, ExistModel=0, uvDir=''):

        # Count for callback function
        self.count=0
        self.count2=0

        # Bounds
        self.lb = lb
        self.ub = ub

        # Mat. properties
        self.rho = 1.0
        #self.mu = 0.02
        self.mu = 0.025
        self.re=5
        
        # Collocation point for subdomain 1
        self.x_c1 = collo1[:, 0:1]
        self.y_c1 = collo1[:, 1:2]

        #Collocation point for subdomain 2
        self.x_c2 = collo2[:, 0:1]
        self.y_c2 = collo2[:, 1:2]

        #Collocation point for interface
        self.x_i1 = interface[:, 0:1]
        self.y_i1 = interface[:, 1:2]

        self.x_INLET = INLET[:, 0:1]
        self.y_INLET = INLET[:, 1:2]
        self.u_INLET = INLET[:, 2:3]
        self.v_INLET = INLET[:, 3:4]
        
        #Cylinder pts
        self.x_CYLD = CYLD[:, 0:1]
        self.y_CYLD = CYLD[:, 1:2]
        self.u_CYLD = CYLD[:, 2:3]
        self.v_CYLD = CYLD[:, 3:4]

        #internal pts
        self.x_internal = internal[:, 0:1]
        self.y_internal = internal[:, 1:2]
        self.u_internal = internal[:, 2:3]
        self.v_internal = internal[:, 3:4]
        self.p_internal = internal[:, 4:5]

        self.x_OUTLET = OUTLET[:, 0:1]
        self.y_OUTLET = OUTLET[:, 1:2]
        self.p_OUTLET = OUTLET[:, 2:3]

        self.x_WALL = WALL[:, 0:1]
        self.y_WALL = WALL[:, 1:2]
        self.u_WALL = WALL[:, 2:3]
        self.v_WALL = WALL[:, 3:4]
        
        # Define layers
        self.layer1 = uv_layers[0]
        self.layer2= uv_layers[1]

        self.loss_rec = []

        # Initialize NNs
        if ExistModel== 0 :
            self.weights1, self.biases1 = self.initialize_NN(self.layer1)
            self.weights2, self.biases2 = self.initialize_NN(self.layer2)
        else:
            print("Loading uv NN ...")
            self.weight1, self.biases1, self.weights2, self.biases2 = self.load_NN(uvDir, self.uv_layers)

        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        self.x1_tf = tf.placeholder(tf.float32, shape=[None, self.x_c1.shape[1]])
        self.y1_tf = tf.placeholder(tf.float32, shape=[None, self.y_c1.shape[1]])
        self.x2_tf = tf.placeholder(tf.float32, shape=[None, self.x_c2.shape[1]])
        self.y2_tf = tf.placeholder(tf.float32, shape=[None, self.y_c2.shape[1]])

        self.x_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.x_WALL.shape[1]])
        self.y_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.y_WALL.shape[1]])
        self.u_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.u_WALL.shape[1]])
        self.v_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.v_WALL.shape[1]])
        
        self.x_CYLD_tf = tf.placeholder(tf.float32, shape=[None, self.x_CYLD.shape[1]])
        self.y_CYLD_tf = tf.placeholder(tf.float32, shape=[None, self.y_CYLD.shape[1]])
        self.u_CYLD_tf = tf.placeholder(tf.float32, shape=[None, self.u_CYLD.shape[1]])
        self.v_CYLD_tf = tf.placeholder(tf.float32, shape=[None, self.v_CYLD.shape[1]])

        self.x_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_OUTLET.shape[1]])
        self.y_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_OUTLET.shape[1]])
        self.p_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.p_OUTLET.shape[1]])

        self.x_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_INLET.shape[1]])
        self.y_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_INLET.shape[1]])
        self.u_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.u_INLET.shape[1]])
        self.v_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.v_INLET.shape[1]])
        
        self.x_internal_tf = tf.placeholder(tf.float32, shape=[None, self.x_internal.shape[1]])
        self.y_internal_tf = tf.placeholder(tf.float32, shape=[None, self.y_internal.shape[1]])
        self.u_internal_tf = tf.placeholder(tf.float32, shape=[None, self.u_internal.shape[1]])
        self.v_internal_tf = tf.placeholder(tf.float32, shape=[None, self.v_internal.shape[1]])
        self.p_internal_tf = tf.placeholder(tf.float32, shape=[None, self.p_internal.shape[1]])

        self.x_c1_tf = tf.placeholder(tf.float32, shape=[None, self.x_c1.shape[1]])
        self.y_c1_tf = tf.placeholder(tf.float32, shape=[None, self.y_c1.shape[1]])

        self.x_c2_tf = tf.placeholder(tf.float32, shape=[None, self.x_c2.shape[1]])
        self.y_c2_tf = tf.placeholder(tf.float32, shape=[None, self.y_c2.shape[1]])

        self.xi1_tf = tf.placeholder(tf.float32, shape=[None, self.x_i1.shape[1]])
        self.yi1_tf = tf.placeholder(tf.float32, shape=[None, self.y_i1.shape[1]])
        # tf graphs
        self.u1_pred, self.v1_pred, self.p1_pred, _, _, _ = self.net_u1(self.x1_tf, self.y1_tf)
        self.u2_pred, self.v2_pred, self.p2_pred, _, _, _ = self.net_u2(self.x2_tf, self.y2_tf)

        self.f_pred_mass1, self.f_pred_u1, self.f_pred_v1, self.f_pred_sxx1, self.f_pred_syy1, self.f_pred_sxy1, self.f_pred_p1, \
            self.f_pred_mass2, self.f_pred_u2, self.f_pred_v2, self.f_pred_sxx2, self.f_pred_syy2, self.f_pred_sxy2, self.f_pred_p2= \
                self.net_f(self.x_c1_tf, self.y_c1_tf, self.x_c2_tf, self.y_c2_tf)
        
        self.u1i , self.v1i, self.p1i, self.sxxi1, self.syyi1, self.sxyi1 = self.net_u1(self.xi1_tf, self.yi1_tf)
        self.u2i , self.v2i, self.p2i, self.sxxi2, self.syyi2, self.sxyi2 = self.net_u2(self.xi1_tf, self.yi1_tf)

        self.u_internal_pred, self.v_internal_pred, self.p_internal_pred, _, _, _ = self.net_u1(self.x_internal_tf, self.y_internal_tf) 

        self.uavgi, self.vavgi, self.pavgi, self.sxxavgi, self.syyavgi, self.sxyavgi, self.f_massi, self.f_ui, self.f_vi, self.f_sxxi, self.f_syyi, self.f_sxyi , self.f_pi= \
            self.net_interface(self.xi1_tf, self.yi1_tf)

        self.u_WALL_pred, self.v_WALL_pred, _, _, _, _ = self.net_u1(self.x_WALL_tf, self.y_WALL_tf) #For top and bottom
        self.u_CYLD_pred, self.v_CYLD_pred, _, _, _, _ = self.net_u2(self.x_CYLD_tf, self.y_CYLD_tf) #Cylinder in domain 2
        self.u_INLET_pred, self.v_INLET_pred, _, _, _, _ = self.net_u1(self.x_INLET_tf, self.y_INLET_tf) #For inlet in subdomain 1
     
        self.loss_internal = tf.reduce_mean(tf.square(self.u_internal_pred-self.u_internal_tf)) \
                         + tf.reduce_mean(tf.square(self.v_internal_pred-self.v_internal_tf)) \
                             + tf.reduce_mean(tf.square(self.p_internal_pred-self.p_internal_tf))

        _, _, self.p_OUTLET_pred, self.s11_OUTLET_pred, _, self.s12_OUTLET1_pred = self.net_u1(self.x_OUTLET_tf, self.y_OUTLET_tf) #Outlet domain 1

        self.p_bc1, self.p_bc2=self.stressfree_bc(self.x_OUTLET_tf, self.y_OUTLET_tf)
        self.loss_f1 = tf.reduce_mean(tf.square(self.f_pred_mass1)) \
                      + tf.reduce_mean(tf.square(self.f_pred_u1)) \
                      + tf.reduce_mean(tf.square(self.f_pred_v1)) \
                      + tf.reduce_mean(tf.square(self.f_pred_sxx1))\
                              + tf.reduce_mean(tf.square(self.f_pred_syy1)) \
                                  + tf.reduce_mean(tf.square(self.f_pred_sxy1)) \
                                      +tf.reduce_mean(tf.square(self.f_pred_p1))
        
        [self.u_x_CYLD_pred,self.v_x_CYLD_pred,self.u_y_CYLD_pred,self.v_y_CYLD_pred] = \
        Gradient_Velocity_2D(self.u_CYLD_pred, self.v_CYLD_pred,self.x_CYLD_tf, self.y_CYLD_tf)
                      
        self.loss_WALL = tf.reduce_mean(tf.square(self.u_WALL_pred-self.u_WALL_tf)) \
                       + tf.reduce_mean(tf.square(self.v_WALL_pred-self.v_WALL_tf))
        self.loss_CYLD = tf.reduce_mean(tf.square(self.u_CYLD_pred-self.u_CYLD_tf)) \
                       + tf.reduce_mean(tf.square(self.v_CYLD_pred-self.v_CYLD_tf))               
        self.loss_INLET = tf.reduce_mean(tf.square(self.u_INLET_pred-self.u_INLET_tf)) \
                         + tf.reduce_mean(tf.square(self.v_INLET_pred-self.v_INLET_tf))
        

                         
        self.loss_OUTLET = tf.reduce_mean(tf.square(self.p_bc1)) + tf.reduce_mean(tf.square(self.p_bc2)) #The stress free boundary condition
        #self.loss_OUTLET = tf.reduce_mean(tf.square(self.p_OUTLET_pred))

        #Losses using subnet 2 
        #Losses using subnet 2 
        self.loss_f2 = tf.reduce_mean(tf.square(self.f_pred_mass2)) \
                      + tf.reduce_mean(tf.square(self.f_pred_u2)) \
                      + tf.reduce_mean(tf.square(self.f_pred_v2)) \
                      + tf.reduce_mean(tf.square(self.f_pred_sxx2))\
                              + tf.reduce_mean(tf.square(self.f_pred_syy2)) \
                                  + tf.reduce_mean(tf.square(self.f_pred_sxy2)) \
                                  +tf.reduce_mean(tf.square(self.f_pred_p2))
        
        #Interface loss
        self.loss_continuity = tf.reduce_mean(tf.square(self.f_massi)) \
                             + tf.reduce_mean(tf.square(self.f_ui)) \
                             + tf.reduce_mean(tf.square(self.f_vi)) \
                             + tf.reduce_mean(tf.square(self.f_sxxi)) \
                             + tf.reduce_mean(tf.square(self.f_syyi)) \
                             + tf.reduce_mean(tf.square(self.f_sxyi)) \
                            + tf.reduce_mean(tf.square(self.f_pi)) 


        self.loss_discontinuous1 = tf.reduce_mean(tf.square(self.u1i - self.uavgi)) \
                                    + tf.reduce_mean(tf.square(self.v1i - self.vavgi)) \
                                    + tf.reduce_mean(tf.square(self.p1i - self.pavgi)) \
                                    +tf.reduce_mean(tf.square(self.sxxi1-self.sxxavgi)) \
                                    +tf.reduce_mean(tf.square(self.syyi1-self.syyavgi)) \
                                    +tf.reduce_mean(tf.square(self.sxyi1-self.sxyavgi)) 

        self.loss_discontinuous2 = tf.reduce_mean(tf.square(self.u2i - self.uavgi)) \
                                    + tf.reduce_mean(tf.square(self.v2i - self.vavgi)) \
                                    + tf.reduce_mean(tf.square(self.p2i - self.pavgi)) \
                                    +tf.reduce_mean(tf.square(self.sxxi2-self.sxxavgi)) \
                                    +tf.reduce_mean(tf.square(self.syyi2-self.syyavgi)) \
                                    +tf.reduce_mean(tf.square(self.sxyi2-self.sxyavgi)) 

        if use_internal == 0:
            
            # self.loss = self.loss_f + 2*(self.loss_CYLD + self.loss_WALL + self.loss_INLET + self.loss_OUTLET)
            self.loss1 = self.loss_f1 + 2*(self.loss_INLET + self.loss_OUTLET + self.loss_WALL) + self.loss_discontinuous1 + self.loss_continuity
            self.loss2 = self.loss_f2 + 2*self.loss_CYLD + self.loss_discontinuous2 + self.loss_continuity
            
        else:
            
            self.loss1 = self.loss_f + 2*(self.loss_WALL + self.loss_INLET + self.loss_internal + self.loss_OUTLET)
        
        if debug == 0:
        
            # Optimizer for solution
            self.optimizer1 = tf.contrib.opt.ScipyOptimizerInterface(self.loss1 + self.loss2,
                                                                    var_list=self.weights1 + self.biases1 +self.weights2 +self.biases2,
                                                                    method='L-BFGS-B',
                                                                    options={'maxiter': 100000,
                                                                              'maxfun': 100000,
                                                                              'maxcor': 50,
                                                                              'maxls': 50,
                                                                              'ftol': 1*np.finfo(float).eps})

            self.optimizer2 = tf.contrib.opt.ScipyOptimizerInterface(self.loss2,
                                                                    var_list=self.weights2 + self.biases2,
                                                                    method='L-BFGS-B',
                                                                    options={'maxiter': 100000,
                                                                              'maxfun': 100000,
                                                                              'maxcor': 50,
                                                                              'maxls': 50,
                                                                              'ftol': 1*np.finfo(float).eps}) #The weights and biases are the variables to be updated based on the loss
        
        elif debug == 1:
            
            self.optimizer1 = tf.contrib.opt.ScipyOptimizerInterface(self.loss1,
                                                                    var_list=self.weights1 + self.biases1,
                                                                    method='L-BFGS-B',
                                                                    options={'maxiter': 100000,
                                                                              'maxfun': 90,
                                                                              'maxcor': 50,
                                                                              'maxls': 50,
                                                                              'ftol': 1*np.finfo(float).eps})
            self.optimizer2 = tf.contrib.opt.ScipyOptimizerInterface(self.loss2,
                                                                    var_list=self.weights2 + self.biases2,
                                                                    method='L-BFGS-B',
                                                                    options={'maxiter': 100000,
                                                                              'maxfun': 90,
                                                                              'maxcor': 50,
                                                                              'maxls': 50,
                                                                              'ftol': 1*np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.optimizer_Adam2 = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam1 = self.optimizer_Adam.minimize(self.loss1,
                                                          var_list=self.weights1 + self.biases1)
        self.train_op_Adam2 = self.optimizer_Adam2.minimize(self.loss2,
                                                          var_list=self.weights2 + self.biases2)
        self.final_train = tf.group(self.train_op_Adam1, self.train_op_Adam2)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)


    def save_NN(self, fileDir):

        weights1 = self.sess.run(self.weights1)
        biases1 = self.sess.run(self.biases1)
        weights2 = self.sess.run(self.weights2)
        biases2 = self.sess.run(self.biases2)
        with open(fileDir, 'wb') as f:
            pickle.dump([weights1, biases1, weights2, biases2], f)
            print("Save uv NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases = pickle.load(f)

            # Stored model must has the same # of layers
            assert num_layers == (len(uv_weights)+1)

            for num in range(0, num_layers - 1):
                W = tf.Variable(uv_weights[num], dtype=tf.float32)
                b = tf.Variable(uv_biases[num], dtype=tf.float32)
                weights.append(W)
                biases.append(b)
                print(" - Load NN parameters successfully...")
        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        # H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        
        if activation_fn == 0:
            
            for l in range(0, num_layers - 2):
                W = weights[l]
                b = biases[l]
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
                
        elif activation_fn == 1:
            
            for l in range(0, num_layers - 2):
                W = weights[l]
                b = biases[l]
                
                H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
                
        elif activation_fn == 2:
            
            for l in range(0, num_layers - 2):
                W = weights[l]
                b = biases[l]
                
                H = tf.nn.leaky_relu(tf.add(tf.matmul(H, W), b))
                
        elif activation_fn == 3:
            
            for l in range(0, num_layers - 2):
                W = weights[l]
                b = biases[l]
                
                H = tf.nn.selu(tf.add(tf.matmul(H, W), b))
            
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u1(self, x, y):
        psips = self.neural_net(tf.concat([x, y], 1), self.weights1, self.biases1)
        
        # psi = psips[:,0:1]
        # p = psips[:,1:2]
        # s11 = psips[:, 2:3]
        # s22 = psips[:, 3:4]
        # s12 = psips[:, 4:5]
        # u = tf.gradients(psi, y)[0]
        # v = -tf.gradients(psi, x)[0]
        
        u = psips[:,0:1]
        v = psips[:,1:2]
        p = psips[:,2:3]
        s11 = psips[:,3:4]
        s22 = psips[:,4:5]
        s12 = psips[:,5:6]
        
        
        
        return u, v, p, s11, s22, s12
    
    def net_u2(self, x, y):
        psips = self.neural_net(tf.concat([x, y], 1), self.weights2, self.biases2)
        
        # psi = psips[:,0:1]
        # p = psips[:,1:2]
        # s11 = psips[:, 2:3]
        # s22 = psips[:, 3:4]
        # s12 = psips[:, 4:5]
        # u = tf.gradients(psi, y)[0]
        # v = -tf.gradients(psi, x)[0]
        
        u = psips[:,0:1]
        v = psips[:,1:2]
        p = psips[:,2:3]
        s11 = psips[:,3:4]
        s22 = psips[:,4:5]
        s12 = psips[:,5:6]
        
        
        
        return u, v, p, s11, s22, s12

    def stressfree_bc(self, x1,y1):
        u1, v1, p1, s11, s22, s12 = self.net_u1(x1, y1)
        re=self.re
        u_x1 = tf.gradients(u1, x1)[0]
        u_y1 = tf.gradients(u1, y1)[0]

        v_x1 = tf.gradients(v1, x1)[0]

        #Sxx
        bc_1 = (2/re)*(u_x1) - p1
        bc_2 = (1/re)*(u_y1 + v_x1)

        return bc_1, bc_2

    def net_f(self, x1, y1, x2, y2): #this is the physics governing equation function

        rho=self.rho
        mu=self.mu
        re=self.re

        #Subnet 1 
        u1, v1, p1, sxx1, syy1, sxy1 = self.net_u1(x1, y1) #The last number represents the subnet 
        sxx1_1 = tf.gradients(sxx1, x1)[0] #ds11_dx, it is a list of all the gradients of ds11/dx and we are taking the first result.
        sxy1_2 = tf.gradients(sxy1, y1)[0]
        syy1_2 = tf.gradients(syy1, y1)[0]
        sxy1_1 = tf.gradients(sxy1, x1)[0]

        # Plane stress problem
        u1_x = tf.gradients(u1, x1)[0] #this part is the automated differentiation
        u1_y = tf.gradients(u1, y1)[0]

        v1_x = tf.gradients(v1, x1)[0]
        v1_y = tf.gradients(v1, y1)[0]
        
        #eqn 1: continuity eqn to ensure the flow is divergent (incompressible)
        f_mass1 = (u1_x + v1_y) #the sum of this should be 0, so the loss is automatically anything that is not 0
        

        # f_u:=Sxx_x+Sxy_y
        #eqn 3
        f_u1 = (u1*u1_x + v1*u1_y) - sxx1_1 - sxy1_2 #The steady code doesnt have the partial derivative of t
        f_v1 = (u1*v1_x + v1*v1_y) - sxy1_1 - syy1_2
       
        #eqn 4
        f_sxx1 = -p1 + (2/re)*u1_x - sxx1
        f_syy1 = -p1 + (2/re)*v1_y - syy1
        f_sxy1 = (1/re)*(u1_y+v1_x) - sxy1

        f_p1 = p1 + (sxx1+syy1)/2

        #Subnet 2
        u2, v2, p2, sxx2, syy2, sxy2 = self.net_u2(x2, y2) #The last number represents the subnet 
        sxx2_1 = tf.gradients(sxx2, x2)[0] #ds11_dx, it is a list of all the gradients of ds11/dx and we are taking the first result.
        sxy2_2 = tf.gradients(sxy2, y2)[0]
        syy2_2 = tf.gradients(syy2, y2)[0]
        sxy2_1 = tf.gradients(sxy2, x2)[0]

        # Plane stress problem
        u2_x = tf.gradients(u2, x2)[0] #this part is the automated differentiation
        u2_y = tf.gradients(u2, y2)[0]

        v2_x = tf.gradients(v2, x2)[0]
        v2_y = tf.gradients(v2, y2)[0]
        
        #eqn 1: continuity eqn to ensure the flow is divergent (incompressible)
        f_mass2 = (u2_x + v2_y) #the sum of this should be 0, so the loss is automatically anything that is not 0
        

        # f_u:=Sxx_x+Sxy_y
        #eqn 3
        f_u2 = (u2*u2_x + v2*u2_y) - sxx2_1 - sxy2_2 #The steady code doesnt have the partial derivative of t
        f_v2 = (u2*v2_x + v2*v2_y) - sxy2_1 - syy2_2
       
        #eqn 4
        f_sxx2 = -p2 + (2/re)*u2_x - sxx2
        f_syy2 = -p2 + (2/re)*v2_y - syy2
        f_sxy2 = (1/re)*(u2_y+v2_x) - sxy2
        f_p2 = p2 + (sxx2+syy2)/2

        return f_mass1, f_u1, f_v1, f_sxx1, f_syy1, f_sxy1, f_p1, f_mass2, f_u2, f_v2, f_sxx2, f_syy2, f_sxy2, f_p2 #returns the physics losses 6 values
    
    def net_interface(self, x1, y1):
        re = self.re
        u1i, v1i, p1i, sxxi1, syyi1, sxyi1 = self.net_u1(x1, y1)
        u2i, v2i, p2i, sxxi2, syyi2, sxyi2 = self.net_u2(x1, y1)       
        uavgi = (u1i + u2i)/2   
        vavgi = (v1i + v2i)/2
        pavgi = (p1i + p2i)/2
        sxxavgi = (sxxi1+sxxi2)/2
        syyavgi = (syyi1+syyi2)/2
        sxyavgi = (sxyi1+sxyi2)/2

        sxxi1_1 = tf.gradients(sxxi1, x1)[0]
        sxyi1_2 = tf.gradients(sxyi1, y1)[0]
        syyi1_2 = tf.gradients(syyi1, y1)[0]
        sxyi1_1 = tf.gradients(sxyi1, x1)[0]

        sxxi2_1 = tf.gradients(sxxi2, x1)[0]
        sxyi2_2 = tf.gradients(sxyi2, y1)[0]
        syyi2_2 = tf.gradients(syyi2, y1)[0]
        sxyi2_1 = tf.gradients(sxyi2, x1)[0]

        ui1_x = tf.gradients(u1i, x1)[0] #this part is the automated differentiation
        ui1_y = tf.gradients(u1i, y1)[0]
        vi1_x = tf.gradients(v1i, x1)[0]
        vi1_y = tf.gradients(v1i, y1)[0]

        ui2_x = tf.gradients(u2i, x1)[0] #this part is the automated differentiation
        ui2_y = tf.gradients(u2i, y1)[0]
        vi2_x = tf.gradients(v2i, x1)[0]
        vi2_y = tf.gradients(v2i, y1)[0]

        f_massi= (ui1_x + vi1_y) - (ui2_x + vi2_y)

        f_ui = (u1i*ui1_x + v1i*ui1_y) - sxxi1_1 - sxyi1_2 - ((u2i*ui2_x + v2i*ui2_y) - sxxi2_1 - sxyi2_2)
        f_vi = (u1i*vi1_x + v1i*vi1_y) - sxyi1_1 - syyi1_2 - ((u2i*vi2_x + v2i*vi2_y) - sxyi2_1 - syyi2_2)

        f_sxxi = -p1i + (2/re)*ui1_x - sxxi1 - (-p2i + (2/re)*ui2_x - sxxi2)
        f_syyi = -p1i + (2/re)*vi1_y - syyi1 - (-p2i + (2/re)*vi2_y - syyi2)
        f_sxyi = (1/re)*(ui1_y+vi1_x) - sxyi1 - ((1/re)*(ui2_y+vi2_x) - sxyi2)
        p_i = p1i + ((sxxi1+syyi1)/2) - p2i - ((sxxi2+syyi2)/2)
        return uavgi, vavgi, pavgi, sxxavgi, syyavgi,sxyavgi, f_massi, f_ui, f_vi, f_sxxi, f_syyi, f_sxyi, p_i


    def outlet_bc(self, x,y):
        u, v, p, s11, s22, s12 = self.net_u1(x, y)
        re=self.re
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]

        v_x = tf.gradients(v, x)[0]

        #Sxx
        bc_1 = (2/re)*(u_x) - p 
        bc_2 = (1/re)*(u_y + v_x)
        return bc_1, bc_2

    def callback(self, loss, loss2):
        self.count = self.count+1
        self.loss_rec.append(loss)
        if self.count % 40 == 0:
            print('{} th iterations, Loss1: {}, Loss2: {}'.format(self.count, loss, loss2))

    def callback2(self, loss):
        self.count2 = self.count2+1
        self.loss_rec.append(loss)
        if self.count2 % 40 == 0:
            print('{} th iterations, Loss2: {}'.format(self.count2, loss))
    
    def viscous_forces_autodiff(self,points):
        #calculate viscous forces of circle using TF autodiff
        
        theta = np.linspace(0.0,2*np.pi,points)[:,None] # N x 1
        d_theta = theta[1,0] - theta[0,0]
        x_cyl = 0.5*np.cos(theta) # N x 1
        y_cyl = 0.5*np.sin(theta) # N x 1
            
        
        tf_dict = {self.x_CYLD_tf: x_cyl, self.y_CYLD_tf: y_cyl}
        
        [u_x_star,
         u_y_star,
         v_x_star,
         v_y_star] = self.sess.run([self.u_x_CYLD_pred,
                                    self.u_y_CYLD_pred,
                                    self.v_x_CYLD_pred,
                                    self.v_y_CYLD_pred], tf_dict)
        
        
    
        INT0 = (2*mu*u_x_star[0:-1])*x_cyl[0:-1] + mu*(u_y_star[0:-1] + v_x_star[0:-1])*y_cyl[0:-1]
        INT1 = (2*mu*u_x_star[1:])*x_cyl[1:] + mu*(u_y_star[1:] + v_x_star[1:])*y_cyl[1:]
            
        F_D = 0.5*np.sum(INT0.T+INT1.T, axis = 1)*d_theta # T x 1
    
        
        INT0 = (2*mu*v_y_star[0:-1])*y_cyl[0:-1] + mu*(u_y_star[0:-1] + v_x_star[0:-1])*x_cyl[0:-1]
        INT1 = (2*mu*v_y_star[1:])*y_cyl[1:] + mu*(u_y_star[1:] + v_x_star[1:])*x_cyl[1:]
            
        F_L = 0.5*np.sum(INT0.T+INT1.T, axis = 1)*d_theta # T x 1
            
        fx = 2*F_D; fy = 2*F_L
            
        return fx, fy


    def train(self, iter, learning_rate):

        tf_dict = {self.x_c1_tf: self.x_c1, self.y_c1_tf: self.y_c1, self.x_c2_tf: self.x_c2, self.y_c2_tf: self.y_c2, self.xi1_tf: self.x_i1, self.yi1_tf: self.y_i1, 
                   self.x_CYLD_tf: self.x_CYLD, self.y_CYLD_tf: self.y_CYLD, self.u_CYLD_tf: self.u_CYLD, self.v_CYLD_tf: self.v_CYLD,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL, self.u_WALL_tf: self.u_WALL, self.v_WALL_tf: self.v_WALL,                   
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_internal_tf: self.x_internal, self.y_internal_tf: self.y_internal, self.u_internal_tf: self.u_internal, self.v_internal_tf: self.v_internal, self.p_internal_tf: self.p_internal,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET, self.p_OUTLET_tf: self.p_OUTLET,
                   self.learning_rate: learning_rate}

        loss_CYLD = []
        loss_WALL = []
        loss_f1 = []
        loss_INLET = []
        loss_internal = []
        loss_OUTLET = []
        iterate_count = []
        loss_list=[]
        loss_list2=[]
        loss_interface_list=[]

        for it in range(iter):
            
            if it == 1000:
                tf_dict[self.learning_rate] = learning_rate/5
                #self.sess.run(self.train_op_Adam1, tf_dict) #Adam takes the learning rate from tf_dict
                #self.sess.run(self.train_op_Adam2, tf_dict)
                self.sess.run(self.final_train, tf_dict)

            else:    
                #self.sess.run(self.train_op_Adam1, tf_dict)
                #self.sess.run(self.train_op_Adam2, tf_dict)
                self.sess.run(self.final_train, tf_dict)

            # Print
            if it % 40 == 0:
                loss_value = self.sess.run(self.loss1, tf_dict) + self.sess.run(self.loss2, tf_dict)
                loss1_value = self.sess.run(self.loss1, tf_dict)
                loss2_value = self.sess.run(self.loss2, tf_dict)
                loss_interface = self.sess.run(self.loss_discontinuous1,tf_dict)+self.sess.run(self.loss_discontinuous2,tf_dict)+self.sess.run(self.loss_continuity,tf_dict)
                print('It: %d, Loss: %.3e, Loss1: %.3e, Loss2 %.3e, Loss Interface %.3e' %
                      (it, loss_value, loss1_value, loss2_value,loss_interface ))

            loss_CYLD.append(self.sess.run(self.loss_CYLD, tf_dict))
            loss_WALL.append(self.sess.run(self.loss_WALL, tf_dict))            
            loss_f1.append(self.sess.run(self.loss_f1, tf_dict))
            loss_INLET.append(self.sess.run(self.loss_INLET, tf_dict))
            loss_internal.append(self.sess.run(self.loss_internal, tf_dict))
            loss_OUTLET.append(self.sess.run(self.loss_OUTLET, tf_dict))
            iterate_count.append(it+1)
            loss_list.append(self.sess.run(self.loss1, tf_dict))
            loss_list2.append(self.sess.run(self.loss2, tf_dict))
            loss_interface_list.append(self.sess.run(self.loss_discontinuous1,tf_dict) + self.sess.run(self.loss_discontinuous2,tf_dict) +self.sess.run(self.loss_continuity,tf_dict))



        # return loss_CYLD, loss_WALL, loss_INLET, loss_OUTLET, loss_f, self.loss
        return loss_WALL, loss_INLET, loss_CYLD, loss_internal, loss_OUTLET, loss_f1, iterate_count, loss_list, loss_list2,  loss_interface_list

    def train_bfgs(self):

        tf_dict = {self.x_c1_tf: self.x_c1, self.y_c1_tf: self.y_c1, self.x_c2_tf: self.x_c2, self.y_c2_tf: self.y_c2, self.xi1_tf: self.x_i1, self.yi1_tf: self.y_i1, 
                   self.x_CYLD_tf: self.x_CYLD, self.y_CYLD_tf: self.y_CYLD, self.u_CYLD_tf: self.u_CYLD, self.v_CYLD_tf: self.v_CYLD,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL, self.u_WALL_tf: self.u_WALL, self.v_WALL_tf: self.v_WALL,                   
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_internal_tf: self.x_internal, self.y_internal_tf: self.y_internal, self.u_internal_tf: self.u_internal, self.v_internal_tf: self.v_internal, self.p_internal_tf: self.p_internal,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET, self.p_OUTLET_tf: self.p_OUTLET}

        self.optimizer1.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss1, self.loss2],
                                loss_callback=self.callback)




    def predict(self, x_star1, x_star2):
        u1_star = self.sess.run(self.u1_pred, {self.x1_tf: x_star1[:, 0:1], self.y1_tf: x_star1[:,1:2]})
        v1_star = self.sess.run(self.v1_pred, {self.x1_tf: x_star1[:, 0:1], self.y1_tf: x_star1[:,1:2]})
        p1_star = self.sess.run(self.p1_pred, {self.x1_tf: x_star1[:, 0:1], self.y1_tf: x_star1[:,1:2]})
        u2_star = self.sess.run(self.u2_pred, {self.x2_tf: x_star2[:, 0:1], self.y2_tf: x_star2[:,1:2]})
        v2_star = self.sess.run(self.v2_pred, {self.x2_tf: x_star2[:, 0:1], self.y2_tf: x_star2[:,1:2]})
        p2_star = self.sess.run(self.p2_pred, {self.x2_tf: x_star2[:, 0:1], self.y2_tf: x_star2[:,1:2]})
        return u1_star, v1_star, p1_star, u2_star, v2_star, p2_star

    def pressure_predict(self, x_star, y_star):
        u_star = self.sess.run(self.u2_pred, {self.x2_tf: x_star, self.y2_tf: y_star})
        v_star = self.sess.run(self.v2_pred, {self.x2_tf: x_star, self.y2_tf: y_star})
        p_star = self.sess.run(self.p2_pred, {self.x2_tf: x_star, self.y2_tf: y_star})
        return u_star, v_star, p_star


    def getloss(self):

        tf_dict = {self.x_c1_tf: self.x_c1, self.y_c1_tf: self.y_c1, self.x_c2_tf: self.x_c2, self.y_c2_tf: self.y_c2, self.xi1_tf: self.x_i1, self.yi1_tf: self.y_i1, 
                   self.x_CYLD_tf: self.x_CYLD, self.y_CYLD_tf: self.y_CYLD, self.u_CYLD_tf: self.u_CYLD, self.v_CYLD_tf: self.v_CYLD,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL, self.u_WALL_tf: self.u_WALL, self.v_WALL_tf: self.v_WALL,                   
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_internal_tf: self.x_internal, self.y_internal_tf: self.y_internal, self.u_internal_tf: self.u_internal, self.v_internal_tf: self.v_internal, self.p_internal_tf: self.p_internal,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET, self.p_OUTLET_tf: self.p_OUTLET}

        loss_f1 = self.sess.run(self.loss_f1, tf_dict)
        loss_CYLD = self.sess.run(self.loss_CYLD, tf_dict)
        loss_WALL = self.sess.run(self.loss_WALL, tf_dict)
        loss_INLET = self.sess.run(self.loss_INLET, tf_dict)
        loss_internal = self.sess.run(self.loss_internal, tf_dict)
        loss1 = self.sess.run(self.loss1, tf_dict)
        loss2 = self.sess.run(self.loss2, tf_dict)
        loss_OUTLET = self.sess.run(self.loss_OUTLET, tf_dict)

        # return loss_CYLD, loss_WALL, loss_INLET, loss_OUTLET, loss_f, loss
        return loss_WALL, loss_INLET, loss_internal, loss_OUTLET, loss_f1, loss1, loss2

def DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    '''
    delete points within cylinder
    '''
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst>r,:]


def mse(ite,los, los2, cyld, interface, outlet):
    plt.plot(ite,los)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Loss (MSE)")
    plt.title("Mean Square Error Loss")
    plt.savefig('./Non_Dim_Output/mse_plot', dpi=360)
    plt.close('all')
    error_df=pd.DataFrame({"Iteration":ite, "Loss": los, "Loss2": los2, "Cylinder": cyld, "Interface": interface, "Outlet": outlet})
    error_df.transpose().to_csv('./Non_Dim_Output/Error_Plot.csv', index=False)

def postProcess(filename, xmin, xmax, ymin, ymax, field_ref, field_MIXED, s=2, alpha=0.5, marker='o'):
    
    #compares between PINN and OF side by side
    
    #adjust font size
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # cmap = mpl.cm.rainbow
    # norm = mpl.colors.Normalize(vmin=0.0, vmax=1.5)
    
    # cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
    #                                 norm=norm,
    #                                 orientation='vertical')
    # cb1.set_label('m/s')
    
    x_ref, y_ref, u_ref, v_ref, p_ref = field_ref[:,0:1], field_ref[:, 1:2], field_ref[:, 2:3], field_ref[:, 3:4], field_ref[:, 4:5]
    x_MIXED, y_MIXED, u_MIXED, v_MIXED, p_MIXED = field_MIXED[:,0:1], field_MIXED[:,1:2], field_MIXED[:,2:3], field_MIXED[:,3:4], field_MIXED[:,4:5]
    velo=[]
    for i in range (len(u_ref)):
        velo.append(((u_ref[i]**2)+(v_ref[i]**2))**0.5)
    velo_pred=[]
    for i in range (len(u_MIXED)):
        velo_pred.append(((u_MIXED[i]**2)+(v_MIXED[i]**2))**0.5)
    #fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7, 4))
    #fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig, ax = plt.subplots(nrows=4, ncols=2)
    

    # Plot MIXED result
    cf = ax[0, 0].scatter(x_MIXED, y_MIXED, c=u_MIXED, vmin=0, vmax=1.5, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=int(s))
    #cf = ax[0,0].tricontourf(x_MIXED[:,0], y_MIXED[:,0], u_MIXED[:,0], levels=np.linspace(0,1.5,11), alpha=alpha, cmap='viridis')
    #cf = ax[0, 0].contourf(x_MIXED, y_MIXED, u_MIXED, vmin=0, vmax=1.5, cmap='viridis')
    # cf = ax[0,0].imshow(u_MIXED, vmin=0, vmax=1.5, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ1 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[0,0].add_patch(circ1)
    ax[0, 0].axis('square')
    for key, spine in ax[0, 0].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 0].set_title(r'$u$ (m/s)')
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)

    cf = ax[1, 0].scatter(x_MIXED, y_MIXED, c=v_MIXED, vmin=-0.5, vmax=0.5, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=int(s)) #vmin and vmax are for colors
    # cf = ax[1,0].tricontourf(x_MIXED[:,0], y_MIXED[:,0], v_MIXED[:,0], 
    #                          levels=np.linspace(-0.5,0.5,11), alpha=alpha, cmap='viridis')
    #cf = ax[1,0].tricontourf(x_MIXED[:,0], y_MIXED[:,0], v_MIXED[:,0], 101, alpha=alpha, cmap='viridis')
    # cf = ax[1,0].contourf(x_MIXED, y_MIXED, v_MIXED, 
    #                           levels=np.linspace(-0.5,0.5,11), alpha=alpha, cmap='viridis')
    # cf = ax[1,0].imshow(v_MIXED, vmin=-0.5, vmax=0.5, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ2 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[1,0].add_patch(circ2)
    ax[1, 0].axis('square')
    for key, spine in ax[1, 0].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 0].set_title(r'$v$ (m/s)')
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)

    cf = ax[2, 0].scatter(x_MIXED, y_MIXED, c=velo_pred, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(s))
    ax[2, 0].axis('square')
    for key, spine in ax[2, 0].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[2, 0].set_xticks([])
    ax[2, 0].set_yticks([])
    ax[2, 0].set_xlim([xmin, xmax])
    ax[2, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[2, 0].set_title('Velocity Magnitude')
    fig.colorbar(cf, ax=ax[2, 0], fraction=0.046, pad=0.04)

    cf = ax[3, 0].scatter(x_MIXED, y_MIXED, c=p_MIXED, vmin=0., vmax=2., alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=int(s))
    # cf = ax[2,0].tricontourf(x_MIXED[:,0], y_MIXED[:,0], p_MIXED[:,0], 
    #                          levels=np.linspace(0,2.,11), alpha=alpha, cmap='viridis')
    # cf = ax[2,0].contourf(x_MIXED, y_MIXED, p_MIXED, 
    #                          levels=np.linspace(0,2.,11), alpha=alpha, cmap='viridis')
    # cf = ax[2,0].imshow(p_MIXED, vmin=0, vmax=2, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ3 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[2,0].add_patch(circ3)
    ax[3, 0].axis('square')
    for key, spine in ax[3, 0].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[3, 0].set_xticks([])
    ax[3, 0].set_yticks([])
    ax[3, 0].set_xlim([xmin, xmax])
    ax[3, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[3, 0].set_title('Pressure (Pa)')
    fig.colorbar(cf, ax=ax[3, 0], fraction=0.046, pad=0.04)

    # Plot ref result
    cf = ax[0, 1].scatter(x_ref, y_ref, c=u_ref, vmin=0, vmax=1.5, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=s)
    #not working
    #cf = ax[0, 1].tricontourf(x_ref[:,0], y_ref[:,0], u_ref[:,0], levels=np.linspace(0,1.5,11), alpha=alpha, cmap='viridis')
    # cf = ax[0, 1].contourf(x_ref, y_ref, c=u_ref, vmin=0, vmax=1.5, 
    #                        alpha=alpha, edgecolors='none', cmap='viridis', 
    #                        marker=marker, s=s)
    #cf = ax[0, 1].tripcolor(x_ref[:,0], y_ref[:,0], u_ref[:,0], alpha=alpha, cmap='viridis')
    # cf = ax[0,1].imshow(u_ref, vmin=0, vmax=1.5, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ4 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[0,1].add_patch(circ4)
    ax[0, 1].axis('square')
    for key, spine in ax[0, 1].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 1].set_title(r'$u$ (m/s)')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)

    cf = ax[1, 1].scatter(x_ref, y_ref, c=v_ref, vmin=-0.5, vmax=0.5, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=s)
    #cf = ax[1, 1].tricontourf(x_ref[:,0], y_ref[:,0], v_ref[:,0], levels=np.linspace(-0.5,0.5,11), alpha=alpha, cmap='viridis')
    # cf = ax[1, 1].contourf(x_ref, y_ref, v_ref, 
    #                      levels=np.linspace(-0.5,0.5,11), alpha=alpha, cmap='viridis')
    # cf = ax[1,1].imshow(v_ref, vmin=-0.5, vmax=0.5, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ5 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[1,1].add_patch(circ5)
    ax[1, 1].axis('square')
    for key, spine in ax[1, 1].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 1].set_title(r'$v$ (m/s)')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)

    cf = ax[2, 1].scatter(x_ref, y_ref, c=velo, alpha=alpha, edgecolors='none', cmap='rainbow', marker=marker, s=s)
    ax[2, 1].axis('square')
    for key, spine in ax[2, 1].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[2, 1].set_xticks([])
    ax[2, 1].set_yticks([])
    ax[2, 1].set_xlim([xmin, xmax])
    ax[2, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[2, 1].set_title('Velocity Magnitude')
    fig.colorbar(cf, ax=ax[2, 1], fraction=0.046, pad=0.04)

    cf = ax[3, 1].scatter(x_ref, y_ref, c=p_ref, vmin=0., vmax=2., alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=s)
    #cf = ax[2, 1].tricontourf(x_ref[:,0], y_ref[:,0], p_ref[:,0], levels=np.linspace(0,2.0,11), alpha=alpha, cmap='viridis')
    # cf = ax[2, 1].contourf(x_ref, y_ref, c=p_ref, vmin=0., vmax=2., alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=s)
    # cf = ax[2,1].imshow(p_ref, vmin=0, vmax=2, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ6 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[2,1].add_patch(circ6)
    ax[3, 1].axis('square')
    for key, spine in ax[3, 1].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[3, 1].set_xticks([])
    ax[3, 1].set_yticks([])
    ax[3, 1].set_xlim([xmin, xmax])
    ax[3, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[3, 1].set_title('Pressure (Pa)')
    fig.colorbar(cf, ax=ax[3, 1], fraction=0.046, pad=0.04)
    plt.tight_layout()

    #plt.savefig('./uvp_mixed_uvp.png', dpi=360)    
    plt.savefig('./Non_Dim_Output/'+filename + '.png',dpi=360)
    plt.close('all')

def postProcess3(filename, xmin, xmax, ymin, ymax, field_ref, field_MIXED, field_diff, s=2, alpha=0.5, marker='o'):
    
    #compares between PINN, OF and the difference
    
    #adjust font size
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    

    x_ref, y_ref, u_ref, v_ref, p_ref = field_ref[:,0:1], field_ref[:,1:2], field_ref[:,2:3], field_ref[:,3:4], field_ref[:,4:5] #from open foam
    x_MIXED, y_MIXED, u_MIXED, v_MIXED, p_MIXED = field_MIXED[:,0:1], field_MIXED[:,1:2], field_MIXED[:,2:3], field_MIXED[:,3:4], field_MIXED[:,4:5]  #From trained data
    x_diff, y_diff, u_diff, v_diff, p_diff, velo_diff = field_MIXED[:,0:1], field_MIXED[:,1:2], field_diff[:,2:3], field_diff[:,3:4], field_diff[:,4:5], field_diff[:,5:6]
    velo=[]
    for i in range (len(u_ref)):
        velo.append(((u_ref[i]**2)+(v_ref[i]**2))**0.5)
    velo_pred=[]
    for i in range (len(u_MIXED)):
        velo_pred.append(((u_MIXED[i]**2)+(v_MIXED[i]**2))**0.5)
    #fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(7, 4))
    fig, ax = plt.subplots(nrows=4, ncols=3)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Plot MIXED result
    cf = ax[0, 0].scatter(x_MIXED, y_MIXED, c=u_MIXED, vmin=0, vmax=1.5, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=int(s)) #s is the size of the marker
    # cf = ax[0,0].imshow(u_MIXED, vmin=0, vmax=1.5, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ1 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[0,0].add_patch(circ1)
    ax[0, 0].axis('square')
    for key, spine in ax[0, 0].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 0].set_title(r'$u$ (m/s)')
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)

    cf = ax[1, 0].scatter(x_MIXED, y_MIXED, c=v_MIXED, vmin=-0.5, vmax=0.5, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=int(s))
    # cf = ax[1,0].imshow(v_MIXED, vmin=-0.5, vmax=0.5, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ2 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[1,0].add_patch(circ2)
    ax[1, 0].axis('square')
    for key, spine in ax[1, 0].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 0].set_title(r'$v$ (m/s)')
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)

    cf = ax[2, 0].scatter(x_MIXED, y_MIXED, c=velo_pred, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=int(s))
    ax[2, 0].axis('square')
    for key, spine in ax[2, 0].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[2, 0].set_xticks([])
    ax[2, 0].set_yticks([])
    ax[2, 0].set_xlim([xmin, xmax])
    ax[2, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[2, 0].set_title('Velocity Magnitude')
    fig.colorbar(cf, ax=ax[2, 0], fraction=0.046, pad=0.04)

    cf = ax[3, 0].scatter(x_MIXED, y_MIXED, c=p_MIXED, vmin=0., vmax=2., alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=int(s))
    # cf = ax[2,0].imshow(p_MIXED, vmin=0, vmax=2, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ3 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[2,0].add_patch(circ3)
    ax[3, 0].axis('square')
    for key, spine in ax[2, 0].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[3, 0].set_xticks([])
    ax[3, 0].set_yticks([])
    ax[3, 0].set_xlim([xmin, xmax])
    ax[3, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[3, 0].set_title('Pressure (Pa)')
    fig.colorbar(cf, ax=ax[3, 0], fraction=0.046, pad=0.04)

    # Plot ref result
    cf = ax[0, 1].scatter(x_ref, y_ref, c=u_ref, vmin=0, vmax=1.5, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=s)
    # cf = ax[0,1].imshow(u_ref, vmin=0, vmax=1.5, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ4 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[0,1].add_patch(circ4)
    ax[0, 1].axis('square')
    for key, spine in ax[0, 1].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 1].set_title(r'$u$ (m/s)')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)

    cf = ax[1, 1].scatter(x_ref, y_ref, c=v_ref, vmin=-0.5, vmax=0.5, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=s)
    # cf = ax[1,1].imshow(v_ref, vmin=-0.5, vmax=0.5, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ5 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[1,1].add_patch(circ5)
    ax[1, 1].axis('square')
    for key, spine in ax[1, 1].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 1].set_title(r'$v$ (m/s)')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)

    cf = ax[2, 1].scatter(x_ref, y_ref, c=velo, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=int(s))
    ax[2, 1].axis('square')
    for key, spine in ax[2, 1].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[2, 1].set_xticks([])
    ax[2, 1].set_yticks([])
    ax[2, 1].set_xlim([xmin, xmax])
    ax[2, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[2, 1].set_title('Velocity Magnitude')
    fig.colorbar(cf, ax=ax[2, 1], fraction=0.046, pad=0.04)
    
    cf = ax[3, 1].scatter(x_ref, y_ref, c=p_ref, vmin=0., vmax=2., alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=s)
    # cf = ax[2,1].imshow(p_ref, vmin=0, vmax=2, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ6 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[2,1].add_patch(circ6)
    ax[3, 1].axis('square')
    for key, spine in ax[3, 1].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[3, 1].set_xticks([])
    ax[3, 1].set_yticks([])
    ax[3, 1].set_xlim([xmin, xmax])
    ax[3, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[3, 1].set_title('Pressure (Pa)')
    fig.colorbar(cf, ax=ax[3, 1], fraction=0.046, pad=0.04)
    
    # Plot diff result
    if diff == 'absolute':
                
        ax[0, 2].set_title('u abs diff')
        cf = ax[0, 2].scatter(x_diff, y_diff, c=u_diff, vmin=-1e-2, vmax=1e-2, alpha=alpha, 
                              edgecolors='none', cmap='viridis', marker=marker, s=int(s))
        # cf = ax[0,2].imshow(u_diff, vmin=-1e-2, vmax=1e-2, extent=[xmin,xmax,ymin,ymax], 
        #                 origin='lower', cmap='viridis')
        #circ7 = plt.Circle((0,0), radius=0.5, color='w')
        #ax[0,2].add_patch(circ7)
        
    elif diff == 'percentage':
        
        ax[0, 2].set_title('u diff (%)')
        cf = ax[0, 2].scatter(x_diff, y_diff, c=u_diff, vmin=-10, vmax=10, alpha=alpha, 
                              edgecolors='none', cmap='viridis', marker=marker, s=int(s))
    ax[0, 2].axis('square')
    for key, spine in ax[0, 2].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_xlim([xmin, xmax])
    ax[0, 2].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black') 
    
        
    #vlimit = np.linspace(-.10, 10, 11, endpoint=True)
    #fig.colorbar(cf, ax=ax[0, 2], fraction=0.046, pad=0.04, ticks=vlimit)
    fig.colorbar(cf, ax=ax[0, 2], fraction=0.046, pad=0.04)
    
    if diff == 'absolute':
                
        ax[1, 2].set_title('v abs diff')
        cf = ax[1, 2].scatter(x_diff, y_diff, c=v_diff, vmin=-1e-2, vmax=1e-2, alpha=alpha, 
                              edgecolors='none', cmap='viridis', marker=marker, s=int(s))
        # cf = ax[1,2].imshow(v_diff, vmin=-1e-2, vmax=1e-2, extent=[xmin,xmax,ymin,ymax], 
        #                 origin='lower', cmap='viridis')
        #circ8 = plt.Circle((0,0), radius=0.5, color='w')
        #ax[1,2].add_patch(circ8)
        
    elif diff == 'percentage':
        
        ax[1, 2].set_title('v diff (%)')
        cf = ax[1, 2].scatter(x_diff, y_diff, c=v_diff, vmin=-10, vmax=10, alpha=alpha, 
                              edgecolors='none', cmap='viridis', marker=marker, s=int(s))
        
    ax[1, 2].axis('square')
    for key, spine in ax[1, 2].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    ax[1, 2].set_xlim([xmin, xmax])
    ax[1, 2].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    
    fig.colorbar(cf, ax=ax[1, 2], fraction=0.046, pad=0.04)

    if diff == 'absolute':
                
        ax[2, 2].set_title('v abs diff')
        cf = ax[2, 2].scatter(x_diff, y_diff, c=velo_diff, vmin=-1e-2, vmax=1e-2, alpha=alpha, 
                              edgecolors='none', cmap='viridis', marker=marker, s=int(s))
        # cf = ax[1,2].imshow(v_diff, vmin=-1e-2, vmax=1e-2, extent=[xmin,xmax,ymin,ymax], 
        #                 origin='lower', cmap='viridis')
        #circ8 = plt.Circle((0,0), radius=0.5, color='w')
        #ax[1,2].add_patch(circ8)
    ax[2, 2].axis('square')
    for key, spine in ax[2, 2].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[2, 2].set_xticks([])
    ax[2, 2].set_yticks([])
    ax[2, 2].set_xlim([xmin, xmax])
    ax[2, 2].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    
    fig.colorbar(cf, ax=ax[2, 2], fraction=0.046, pad=0.04)

    if diff == 'absolute':
                
        ax[3, 2].set_title('p abs diff')
        cf = ax[3, 2].scatter(x_diff, y_diff, c=p_diff, vmin=-1e-2, vmax=1e-2, alpha=alpha, 
                              edgecolors='none', cmap='viridis', marker=marker, s=int(s))
        # cf = ax[2,2].imshow(p_diff, vmin=-1e-2, vmax=1e-2, extent=[xmin,xmax,ymin,ymax], 
        #                 origin='lower', cmap='viridis')
        #circ9 = plt.Circle((0,0), radius=0.5, color='w')
        #ax[2,2].add_patch(circ9)
        
    elif diff == 'percentage':
        
        ax[3, 2].set_title('p diff (%)')
        cf = ax[2, 2].scatter(x_diff, y_diff, c=p_diff, vmin=-10, vmax=10, alpha=alpha, 
                              edgecolors='none', cmap='viridis', marker=marker, s=int(s))
        
    ax[3, 2].axis('square')
    for key, spine in ax[3, 2].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[3, 2].set_xticks([])
    ax[3, 2].set_yticks([])
    ax[3, 2].set_xlim([xmin, xmax])
    ax[3, 2].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    
    fig.colorbar(cf, ax=ax[3, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    #plt.savefig('./uvp_mixed_uvp.png', dpi=360)    
    plt.savefig('./Non_Dim_Output/'+filename+'.png',dpi=360)
    plt.close('all')

def preprocess(dir='FenicsSol.mat'):
    '''
    Load reference solution from Fenics or Fluent
    '''
    data = scipy.io.loadmat(dir)

    X = data['x']
    Y = data['y']
    P = data['p']
    vx = data['vx']
    vy = data['vy']

    x_star = X.flatten()[:, None]
    y_star = Y.flatten()[:, None]
    p_star = P.flatten()[:, None]
    vx_star = vx.flatten()[:, None]
    vy_star = vy.flatten()[:, None]

    return x_star, y_star, vx_star, vy_star, p_star

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))

def pressure_forces(points):
    #calculate pressure forces of circle, also plot pressure vs theta
    
    pi = np.pi
    delta = pi/points
    
    theta = np.zeros((points)); p_pred = np.zeros((points))
    x_loc = np.zeros((points)); y_loc = np.zeros((points))
    
    fx = 0.; fy = 0.
    
    for i in range(points):
        
        theta[i] = i*delta
        x_loc[i] = 0.50*np.cos(theta[i]); y_loc[i] = 0.50*np.sin(theta[i])
    
    x_loc = x_loc.reshape(-1, 1); y_loc = y_loc.reshape(-1, 1)
    _, _, p_pred = model.pressure_predict(x_loc, y_loc)
    
    for i in range(points):
        
        dx = delta*np.cos(theta[i]); dy = delta*np.sin(theta[i])
        fx = fx - p_pred[i]*dx; fy = fy - p_pred[i]*dy
        
    fig = plt.figure()
    error_df=pd.DataFrame({"Theta": theta, "Pressure": p_pred.flatten()})
    error_df.transpose().to_csv('./Non_Dim_Output/Pressure_plot.csv', index=False)
     
    circle_ref = np.loadtxt('pvt.csv',delimiter=',',skiprows=6)
    x_circle = circle_ref[:,0:1]
    y_circle = circle_ref[:,1:2]

    x_circle = (x_circle/180)*np.pi


    plt.plot(x_circle, y_circle, label="CFD Reference")
    plt.plot(theta, p_pred, label="PINN Prediction")
    plt.title("Pressure vs Theta")
    plt.xlabel("Theta")
    plt.ylabel("Pressure")
    plt.legend()
    plt.show()
    plt.savefig('./Non_Dim_Output/pressure_vs_theta.png',dpi=360)
    
    fx = 2*fx; fy = 2*fy
        
    return fx, fy


def viscous_forces(points,delta_dist):
    #calculate pressure forces of circle, also plot pressure vs theta
    
    #pi = 3.141592653
    delta = np.pi/points
    test_points_x = np.zeros((5,1)); test_points_y = np.zeros((5,1))
    
    theta = np.zeros((points)); p_pred = np.zeros((points))
    x_loc = np.zeros((points)); y_loc = np.zeros((points))
    
    fx = 0.; fy = 0.
    
    for i in range(points):
        
        theta[i] = i*delta
        x_loc[i] = 0.50*np.cos(theta[i]) + 1e-10 #prevent dist_x/y = 0
        y_loc[i] = 0.50*np.sin(theta[i]) + 1e-10
        dist_x = np.sign(x_loc[i])*delta_dist; dist_y = np.sign(y_loc[i])*delta_dist
        x1 = x_loc[i] + dist_x; x2 = x_loc[i] + 2.*dist_x
        y1 = y_loc[i] + dist_y; y2 = y_loc[i] + 2.*dist_y
        
        test_points_x[0,0] = x_loc[i]; test_points_y[0,0] = y_loc[i]
        test_points_x[1,0] = x1; test_points_y[1,0] = y_loc[i]
        test_points_x[2,0] = x2; test_points_y[2,0] = y_loc[i]
        test_points_x[3,0] = x_loc[i]; test_points_y[3,0] = y1
        test_points_x[4,0] = x_loc[i]; test_points_y[4,0] = y2
        
        u_pred, v_pred, _ = model.pressure_predict(test_points_x, test_points_y)
        u0 = u_pred[0,0]; v0 = v_pred[0,0]
        ui1 = u_pred[1,0]; vi1 = v_pred[1,0]
        ui2 = u_pred[2,0]; vi2 = v_pred[2,0]
        uj1 = u_pred[3,0]; vj1 = v_pred[3,0]
        uj2 = u_pred[4,0]; vj2 = v_pred[4,0]
        
        du_dx = -np.sign(x_loc[i])*(3*u0 - 4*ui1 + ui2)/(2*delta_dist)
        dv_dx = -np.sign(x_loc[i])*(3*v0 - 4*vi1 + vi2)/(2*delta_dist)
        du_dy = -np.sign(y_loc[i])*(3*u0 - 4*uj1 + uj2)/(2*delta_dist)
        dv_dy = -np.sign(y_loc[i])*(3*v0 - 4*vj1 + vj2)/(2*delta_dist)
        
        dx = delta*np.cos(theta[i]); dy = delta*np.sin(theta[i])
        
        fx = fx + (1./Re)*(2*du_dx*dx + du_dy*dy + dv_dx*dy )
        fy = fy + (1./Re)*(2*dv_dy*dy + du_dy*dx + dv_dx*dx )   
   
    fx = 2*fx; fy = 2*fy
        
    return fx, fy

if __name__ == "__main__":
    
    shutil.rmtree('./Non_Dim_Output', ignore_errors=True)
    os.makedirs('./Non_Dim_Output')
    # Domain bounds
    xlb = -5
    ylb = -5
    xub = 9.33
    yub = 5
    ixlb = -2
    iylb = -2.
    ixub = 5
    iyub = 2
    lb = np.array([xlb, ylb])   #lower left x,y coord of boundary
    ub = np.array([xub, yub])  #upper right x,y coord of boundary
    
    u_ref = 1.0

    # Network configuration
    uv_layers = [[2] + 8*[30] + [6], [2] + 8*[40] + [6]]  # 2 i/p, 8 layers x 40 neurons each, 6 o/p

    #Number of points of each domain
    N_f1= 12000
    N_f2= 12000
    #Subdomain 1
    #Boundary Points in Subdomain 1
    # WALL = [x, y], u=v=0
    WALL_TOP = [xlb, yub] + [xub-xlb, 0.0] * lhs(2, 801) #lhs = Latin-Hypercube from pyDOE
    #lhs(2,441) = generate 2 variables, 441 pts each
    x_WALL_TOP = WALL_TOP[:,0:1]
    y_WALL_TOP = WALL_TOP[:,1:2]
    u_WALL_TOP = np.zeros_like(x_WALL_TOP)
    v_WALL_TOP = np.zeros_like(x_WALL_TOP)
    u_WALL_TOP[:] = u_ref
    v_WALL_TOP[:] = 0.
    #WALL_TOP = np.concatenate((WALL_TOP, u_WALL_TOP, v_WALL_TOP), 1)
    
    #[0.0, 4.1] + [11.0, 0.0]* = starting + delta * to get values within range
    WALL_BOTTOM = [xlb, xlb] + [xub-xlb, 0.0] * lhs(2, 801)
    x_WALL_BOTTOM = WALL_BOTTOM[:,0:1]
    y_WALL_BOTTOM = WALL_BOTTOM[:,1:2]
    u_WALL_BOTTOM = np.zeros_like(x_WALL_BOTTOM)
    v_WALL_BOTTOM = np.zeros_like(x_WALL_BOTTOM)
    u_WALL_BOTTOM[:] = 1 #Scaled should always be 1
    v_WALL_BOTTOM[:] = 0.
    #WALL_BOTTOM = np.concatenate((WALL_BOTTOM, u_WALL_BOTTOM, v_WALL_BOTTOM), 1)

    # INLET = [x, y, u, v]
    
    INLET = [xlb, ylb] + [0.0, yub-ylb] * lhs(2, 801)
    y_INLET = INLET[:,1:2]
    # u_INLET = 4*U_max*y_INLET*(4.1-y_INLET)/(4.1**2) #parabolic u inlet, max at mid y position
    u_INLET = np.zeros_like(y_INLET)
    u_INLET[:] = u_ref
    v_INLET = 0*y_INLET #v = 0 at inlet
    INLET = np.concatenate((INLET, u_INLET, v_INLET), 1)

    # plt.scatter(INLET[:, 1:2], INLET[:, 2:3], marker='o', alpha=0.2, color='red')
    # plt.show()
    
    #use OF results as training data for internal pts
    dataset = np.loadtxt('uvp_openfoam1.csv',delimiter=',',usecols=range(5),skiprows=1)
    internal, internal_test= train_test_split(dataset, test_size=test_size, random_state=1234) #5% of data used for training

    # INLET = [x, y], p=0 #or OUTLET?
    OUTLET = [xub, ylb] + [0.0, yub-ylb] * lhs(2, 801)
    x_OUTLET = OUTLET[:,0:1]
    y_OUTLET = OUTLET[:,1:2]
    p_OUTLET = np.zeros_like(y_OUTLET)
    p_OUTLET[:] = 0.
    OUTLET = np.concatenate((OUTLET, p_OUTLET), 1)

    WALL_x = np.concatenate((x_WALL_TOP, x_WALL_BOTTOM), 0)
    WALL_y = np.concatenate((y_WALL_TOP, y_WALL_BOTTOM), 0)
    WALL_u = np.concatenate((u_WALL_TOP, u_WALL_BOTTOM), 0)
    WALL_v = np.concatenate((v_WALL_TOP, v_WALL_BOTTOM), 0)
    WALL = np.concatenate((WALL_x, WALL_y, WALL_u, WALL_v), 1) #These are the top and bottom wall coordinates and velocity
    
    collo1 = lb + (ub - lb) * lhs(2, 40000)
    collo1 = list(filter(lambda x: ((x[0]<-2 or x[0]>5) or (x[1]<-2 or x[1]>2)), collo1)) #Delete points within the subdomain 
    collo1 = np.array(collo1)
    idx1 = np.random.choice(collo1.shape[0], N_f1, replace=False)    
    collo1 = collo1[idx1,:] #Randomly selected point from sub domain 1, N_f1 is specified 

    #Interface Coordinates
    ITOP = [ixlb, iyub] + [ixub-ixlb, 0] * lhs(2, 401)
    IBOT = [ixlb, iylb] + [ixub-ixlb,0] * lhs(2, 401)
    II = [ixlb, iylb] + [0, iyub-iylb] * lhs(2, 201)
    IO = [ixub, iylb] + [0,iyub-iylb] * lhs(2,201)
    interface= np.concatenate((ITOP,IBOT,II,IO),0) #Interface coordinates 

    #Subdomain 2
    # Cylinder surface
    r = 0.5
    theta = [0.0] + [2*np.pi] * lhs(1, 1001)
    x_CYLD = np.multiply(r, np.cos(theta))+0.0
    y_CYLD = np.multiply(r, np.sin(theta))+0.0
    u_CYLD = np.zeros_like(x_CYLD)
    v_CYLD = np.zeros_like(x_CYLD)
    u_CYLD[:] = 0.
    v_CYLD[:] = 0.
    CYLD = np.concatenate((x_CYLD, y_CYLD, u_CYLD, v_CYLD),1)

    collo2 = [ixlb, iylb] + [ixub-ixlb,iyub-iylb] * lhs(2, N_f2)
    collo2 = DelCylPT(collo2, xc= 0.0, yc = 0.0, r=0.5)


    # Visualize the collocation points
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(collo1[:,0:1], collo1[:,1:2], marker='o', alpha=0.5 ,color='blue', label= "Domain 1")
    plt.scatter(collo2[:,0:1], collo2[:,1:2], marker="+", alpha =0.5, color="yellow", label = "Domain 2")
    plt.scatter(interface[:,0:1],interface[:,1:2], marker="o", alpha=0.5, color="black", label = "Interface")
    plt.scatter(CYLD[:,0:1],CYLD[:,1:2],marker="o", alpha=0.5, color="green", label = "Boundary Points")
    # plt.scatter(CYLD[:,0:1], CYLD[:,1:2], marker='o', alpha=0.2 , color='yellow')
    plt.scatter(WALL[:,0:1], WALL[:,1:2], marker='o', alpha=0.2 , color='green')
    plt.scatter(OUTLET[:, 0:1], OUTLET[:, 1:2], marker='o', alpha=0.2, color='green')
    plt.scatter(INLET[:, 0:1], INLET[:, 1:2], marker='o', alpha=0.2, color='green')
    #plt.scatter(internal[:, 0:1], internal[:, 1:2], marker='o', alpha=0.2, color='yellow')
    plt.legend(title="Point Classification", loc="upper center", fancybox=True, bbox_to_anchor=(0.5,-0.05),shadow=True, ncol=3)
    plt.show()

    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # Train from scratch
        # model = PINN_laminar_flow(XY_c, INLET, OUTLET, WALL, uv_layers, lb, ub)

        # Load trained neural network
        model = XPINN_laminar_flow(collo1, collo2, interface, INLET, internal, OUTLET, CYLD, WALL, uv_layers, lb, ub, ExistModel = use_old_model, uvDir = 'uvNN.pickle')

        start_time = time.time()
        
        if debug == 0:
            
            loss_WALL, loss_INLET, loss_CYLD, loss_internal, loss_OUTLET, loss_f1, iterate_count, loss_list, loss_list2, loss_interface = model.train(iter=10000, learning_rate=5e-4)
        
        elif debug == 1:
            
            loss_WALL, loss_INLET, loss_CYLD, loss_internal, loss_OUTLET, loss_f1, iterate_count, loss_list, loss_list2, loss_interface = model.train(iter=60, learning_rate=5e-4)
        
        model.train_bfgs()
        print("--- %s seconds ---" % (time.time() - start_time))

        # Save neural network
        model.save_NN('uvNN.pickle')

        # Save loss history
        with open('loss_history.pickle', 'wb') as f:
            pickle.dump(model.loss_rec, f)
            
        if data_ref == 'fluent':

            field_ref = np.loadtxt('Fluent Normal Domain.csv',delimiter=',',skiprows=6)
            x_ref = field_ref[:,0:1]
            y_ref = field_ref[:,1:2]
            u_ref = field_ref[:,4:5]
            v_ref = field_ref[:,5:6]
            p_ref = field_ref[:,3:4]
            
            field_ref = [x_ref, y_ref, u_ref, v_ref, p_ref]
            
        elif data_ref == 'openfoam':
            
            field_ref = np.loadtxt('Openfoam Dr Tay.csv',delimiter=',',skiprows=6)
            x_ref = field_ref[:,0:1]
            y_ref = field_ref[:,1:2]
            u_ref = field_ref[:,3:4]
            v_ref = field_ref[:,4:5]
            p_ref = field_ref[:,6:7]
            
            field_ref = [x_ref, y_ref, u_ref, v_ref, p_ref]
        
        # Get mixed-form PINN prediction
        x_PINN1 = np.linspace(xlb, xub, 251)
        y_PINN1 = np.linspace(ylb, yub, 101)
        x_PINN1, y_PINN1 = np.meshgrid(x_PINN1, y_PINN1)
        x_PINN1 = (x_PINN1.flatten()[:, None])
        y_PINN1 = (y_PINN1.flatten()[:, None])
        x_star1 = np.concatenate((x_PINN1, y_PINN1),1)
        x_star1 = list(filter(lambda x: ((x[0]<=-2 or x[0]>=5) or (x[1]<=-2 or x[1]>=2)), x_star1))
        x_star1= np.array(x_star1)
        x_PINN1 = x_star1[:,0:1]
        y_PINN1 = x_star1[:,1:2]
        x_PINN1 = (x_PINN1.flatten()[:, None])
        y_PINN1 = (y_PINN1.flatten()[:, None])        


        x_PINN2 = np.linspace(ixlb, ixub, 251)
        y_PINN2 = np.linspace(iylb, iyub, 101)
        x_PINN2, y_PINN2 = np.meshgrid(x_PINN2, y_PINN2)
        x_PINN2 = x_PINN2.flatten()[:, None]
        y_PINN2 = y_PINN2.flatten()[:, None]
        dst = ((x_PINN2-0.0)**2+(y_PINN2-0.0)**2)**0.5
        x_PINN2 = x_PINN2[dst >= 0.5]
        y_PINN2 = y_PINN2[dst >= 0.5]
        x_PINN2 = x_PINN2.flatten()[:, None]
        y_PINN2 = y_PINN2.flatten()[:, None]
        x_star2 = np.concatenate((x_PINN2, y_PINN2),1)
        u_PINN1, v_PINN1, p_PINN1, u_PINN2, v_PINN2, p_PINN2 = model.predict(x_star1, x_star2)
        field_MIXED1 = np.concatenate((x_PINN1, y_PINN1, u_PINN1, v_PINN1, p_PINN1),1)
        field_MIXED2 = np.concatenate((x_PINN2, y_PINN2, u_PINN2, v_PINN2, p_PINN2),1)
        field_MIXED = np.concatenate((field_MIXED1, field_MIXED2),0)
        df = pd.DataFrame(field_MIXED)
        df.to_csv('uvp_mixed.csv',index=False)
        Cov = pd.read_csv("uvp_mixed.csv", names=["x", "y", "u", "v","p"])
        Cov.to_csv('uvp_mixed.csv',index=False)
        lines = open('uvp_mixed.csv', 'r').readlines()
        del lines[1]
        open('uvp_mixed.csv', 'w').writelines(lines)
        
        #interpolate OF data to match current XPINN x,y
        u_ref_int1 = scipy.interpolate.griddata((x_ref.flatten(),y_ref.flatten()),u_ref.flatten() , (x_PINN1,y_PINN1),method='cubic')
        v_ref_int1 = scipy.interpolate.griddata((x_ref.flatten(),y_ref.flatten()),v_ref.flatten() , (x_PINN1,y_PINN1),method='cubic')
        p_ref_int1 = scipy.interpolate.griddata((x_ref.flatten(),y_ref.flatten()),p_ref.flatten() , (x_PINN1,y_PINN1),method='cubic')
        u_ref_int2 = scipy.interpolate.griddata((x_ref.flatten(),y_ref.flatten()),u_ref.flatten() , (x_PINN2,y_PINN2),method='cubic')
        v_ref_int2 = scipy.interpolate.griddata((x_ref.flatten(),y_ref.flatten()),v_ref.flatten() , (x_PINN2,y_PINN2),method='cubic')
        p_ref_int2 = scipy.interpolate.griddata((x_ref.flatten(),y_ref.flatten()),p_ref.flatten() , (x_PINN2,y_PINN2),method='cubic')
        field_ref_int1 = np.concatenate((x_PINN1, y_PINN1, u_ref_int1, v_ref_int1, p_ref_int1),1)
        field_ref_int2 = np.concatenate((x_PINN2, y_PINN2, u_ref_int2, v_ref_int2, p_ref_int2),1)      
        field_ref_int = np.concatenate((field_ref_int1, field_ref_int2),0)
        
        # Plot the comparison of u, v, p
        postProcess('uvp_PINN_vs_OF_SBC', xmin=-5.0, xmax=9.33, ymin=-5.0, ymax=5.0, field_ref=field_ref_int, field_MIXED=field_MIXED, s=5, alpha=1)
        mse(ite=iterate_count, los=loss_list, los2=loss_list2, cyld = loss_CYLD, interface = loss_interface, outlet=loss_OUTLET)        
        if diff == 'absolute':
            
            u_diff = field_MIXED[: , 2:3] - field_ref_int[: , 2:3]
            v_diff = field_MIXED[: , 3:4] - field_ref_int[: , 3:4]
            p_diff = field_MIXED[: , 4:5] - field_ref_int[: , 4:5]
            velo_diff = ((field_MIXED[: , 2:3]**2+field_MIXED[: , 3:4]**2)**0.5) - (field_ref_int[: , 2:3]**2+field_ref_int[: , 3:4]**2)**0.5
            
        elif diff == 'percentage':
            
            field_ref_int[: , 2:3] = field_ref_int[: , 2:3] + 1e-10 #prevent 0 divide
            field_ref_int[: , 3:4] = field_ref_int[: , 3:4]+ 1e-10 #prevent 0 divide
            field_ref_int[: , 4:5] = field_ref_int[: , 4:5] + 1e-10 #prevent 0 divide
        
            u_diff = 100.*(field_MIXED[: , 2:3] - field_ref_int[: , 2:3])/field_ref_int[: , 2:3]
            v_diff = 100.*(field_MIXED[: , 3:4]- field_ref_int[: , 3:4])/field_ref_int[: , 3:4]
            p_diff = 100.*(field_MIXED[: , 4:5] - field_ref_int[: , 4:5])/field_ref_int[: , 4:5]
            
            # u_diff_xy = 100.*(u_pred_xy - u_ref_xy)/u_ref_xy
            # v_diff_xy = 100.*(v_pred_xy - v_ref_xy)/v_ref_xy
            # p_diff_xy = 100.*(p_pred_xy - p_ref_xy)/p_ref_xy
        
    field_diff = np.concatenate((field_MIXED[:,0:1], field_MIXED[:,1:2], u_diff, v_diff, p_diff, velo_diff),1)

    postProcess3('uvp_PINN_vs_OF_diff_SBC', xmin=-5.0, xmax=9.33,ymin=-5.0, ymax=5.0, field_ref=field_ref_int,field_MIXED=field_MIXED, field_diff=field_diff, s=5, alpha=1)
    fx_p,fy_p = pressure_forces(surface_pts)
    fx_v,fy_v = model.viscous_forces_autodiff(surface_pts)
    
    fx = fx_p + fx_v
    fy = fy_p + fy_v
    print(fx_p,fx_v,fx)
    print(fy_p,fy_v,fy)