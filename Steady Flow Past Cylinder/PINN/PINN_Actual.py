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
import lic
from yellowfin.yellowfin import YFOptimizer #importing the yellowfin optimizer


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
beta = 2
CFD_grid = 0

#used for force calculations
surface_pts = 200
rho = 1.0
mu = 0.025
Re=40
consolidate_collo=2
use_neural_form =0
optimizer = 0#0 for adam, 1 for yellowfin

delta_dist = 1e-4

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

class PINN_laminar_flow:
    # Initialize the class
    def __init__(self, Collo, INLET, internal, OUTLET_n, OUTLET_d, CYLD, TOP_BOTTOM, uv_layers, lb, ub, ExistModel, uvDir):

        # Count for callback function
        self.count=0

        # Bounds
        self.lb = lb
        self.ub = ub

        # Mat. properties
        self.rho = 1.0
        #self.mu = 0.02
        self.mu = 0.025
        self.re=40
        
        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]

        self.x_INLET = INLET[:, 0:1]
        self.y_INLET = INLET[:, 1:2]
        self.u_INLET = INLET[:, 2:3]
        self.v_INLET = INLET[:, 3:4]
        
        #internal pts
        self.x_internal = internal[:, 0:1]
        self.y_internal = internal[:, 1:2]
        self.u_internal = internal[:, 2:3]
        self.v_internal = internal[:, 3:4]
        self.p_internal = internal[:, 4:5]

        self.x_OUTLET_d = OUTLET_d[:, 0:1]
        self.y_OUTLET_d = OUTLET_d[:, 1:2]
        self.p_OUTLET_d = OUTLET_d[:, 2:3]

        self.x_OUTLET_n = OUTLET_n[:, 0:1]
        self.y_OUTLET_n = OUTLET_n[:, 1:2]
        self.p_OUTLET_n = OUTLET_n[:, 2:3]

        self.x_TOP_BOTTOM = TOP_BOTTOM[:, 0:1]
        self.y_TOP_BOTTOM = TOP_BOTTOM[:, 1:2]
        self.u_TOP_BOTTOM = TOP_BOTTOM[:, 2:3]
        self.v_TOP_BOTTOM = TOP_BOTTOM[:, 3:4]
        
        self.x_CYLD = CYLD[:, 0:1]
        self.y_CYLD = CYLD[:, 1:2]
        self.u_CYLD = CYLD[:, 2:3]
        self.v_CYLD = CYLD[:, 3:4]

        # Define layers
        self.uv_layers = uv_layers

        self.loss_rec = []

        # Initialize NNs
        if ExistModel== 0 :
            self.uv_weights, self.uv_biases = self.initialize_NN(self.uv_layers)
        else:
            print("Loading uv NN ...")
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, self.uv_layers)

        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.inlet_weight =tf.placeholder(tf.float32, shape=[])
        self.outlet_weight = tf.placeholder(tf.float32, shape=[])
        self.wall_weight= tf.placeholder(tf.float32, shape=[])
        self.cyld_weight = tf.placeholder(tf.float32, shape=[])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        self.x_TOP_BOTTOM_tf = tf.placeholder(tf.float32, shape=[None, self.x_TOP_BOTTOM.shape[1]])
        self.y_TOP_BOTTOM_tf = tf.placeholder(tf.float32, shape=[None, self.y_TOP_BOTTOM.shape[1]])
        self.u_TOP_BOTTOM_tf = tf.placeholder(tf.float32, shape=[None, self.u_TOP_BOTTOM.shape[1]])
        self.v_TOP_BOTTOM_tf = tf.placeholder(tf.float32, shape=[None, self.v_TOP_BOTTOM.shape[1]])
        
        self.x_CYLD_tf = tf.placeholder(tf.float32, shape=[None, self.x_CYLD.shape[1]])
        self.y_CYLD_tf = tf.placeholder(tf.float32, shape=[None, self.y_CYLD.shape[1]])
        self.u_CYLD_tf = tf.placeholder(tf.float32, shape=[None, self.u_CYLD.shape[1]])
        self.v_CYLD_tf = tf.placeholder(tf.float32, shape=[None, self.v_CYLD.shape[1]])

        self.x_OUTLET_d_tf = tf.placeholder(tf.float32, shape=[None, self.x_OUTLET_d.shape[1]])
        self.y_OUTLET_d_tf = tf.placeholder(tf.float32, shape=[None, self.y_OUTLET_d.shape[1]])
        self.p_OUTLET_d_tf = tf.placeholder(tf.float32, shape=[None, self.p_OUTLET_d.shape[1]])

        self.x_OUTLET_n_tf = tf.placeholder(tf.float32, shape=[None, self.x_OUTLET_n.shape[1]])
        self.y_OUTLET_n_tf = tf.placeholder(tf.float32, shape=[None, self.y_OUTLET_n.shape[1]])
        self.p_OUTLET_n_tf = tf.placeholder(tf.float32, shape=[None, self.p_OUTLET_n.shape[1]])

        self.x_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_INLET.shape[1]])
        self.y_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_INLET.shape[1]])
        self.u_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.u_INLET.shape[1]])
        self.v_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.v_INLET.shape[1]])
        
        self.x_internal_tf = tf.placeholder(tf.float32, shape=[None, self.x_internal.shape[1]])
        self.y_internal_tf = tf.placeholder(tf.float32, shape=[None, self.y_internal.shape[1]])
        self.u_internal_tf = tf.placeholder(tf.float32, shape=[None, self.u_internal.shape[1]])
        self.v_internal_tf = tf.placeholder(tf.float32, shape=[None, self.v_internal.shape[1]])
        self.p_internal_tf = tf.placeholder(tf.float32, shape=[None, self.p_internal.shape[1]])

        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        # tf graphs
        self.u_pred, self.v_pred, self.p_pred, _, _, _ = self.net_uv(self.x_tf, self.y_tf)
        self.f_pred_mass, self.f_pred_u, self.f_pred_v, self.f_pred_s11, \
            self.f_pred_s22, self.f_pred_s12, self.f_pred_p = \
                self.net_f(self.x_c_tf, self.y_c_tf)
        
        self.u_TOP_BOTTOM_pred, self.v_TOP_BOTTOM_pred, _, _, _, _ = self.net_uv(self.x_TOP_BOTTOM_tf, self.y_TOP_BOTTOM_tf)
        self.u_CYLD_pred, self.v_CYLD_pred, _, _, _, _ = self.net_uv(self.x_CYLD_tf, self.y_CYLD_tf)
        self.f_pred_CYLD_dp_dn = self.net_neumann_normal(self.x_CYLD_tf, self.y_CYLD_tf,cyld_x_center=0,cyld_y_center=0,radius=0.5)
        self.u_INLET_pred, self.v_INLET_pred, _, _, _, _ = self.net_uv(self.x_INLET_tf, self.y_INLET_tf)
        self.u_internal_pred, self.v_internal_pred, self.p_internal_pred, _, _, _ = self.net_uv(self.x_internal_tf, self.y_internal_tf)
        _, _, self.p_OUTLET_pred, _, _, _ = self.net_uv(self.x_OUTLET_d_tf, self.y_OUTLET_d_tf)
        self.f_pred_OUTLET_dp_dx = self.net_neumann_outlet(self.x_OUTLET_n_tf, self.y_OUTLET_n_tf)
        self.bc_1, self.bc_2 = self.stressfree_bc(self.x_OUTLET_n_tf, self.y_OUTLET_n_tf)

        self.loss_f = tf.reduce_mean(tf.square(self.f_pred_mass)) \
                      + tf.reduce_mean(tf.square(self.f_pred_u)) \
                      + tf.reduce_mean(tf.square(self.f_pred_v)) \
                      + tf.reduce_mean(tf.square(self.f_pred_s11))\
                            + tf.reduce_mean(tf.square(self.f_pred_p)) \
                              + tf.reduce_mean(tf.square(self.f_pred_s22)) \
                                  + tf.reduce_mean(tf.square(self.f_pred_s12)) 
                        
        [self.u_x_CYLD_pred,self.v_x_CYLD_pred,self.u_y_CYLD_pred,self.v_y_CYLD_pred] = \
        Gradient_Velocity_2D(self.u_CYLD_pred, self.v_CYLD_pred,self.x_CYLD_tf, self.y_CYLD_tf)
        #self.du_dx, self.dv_dy = self.viscous_forces_auto_diff(self.x_CYLD_tf, self.y_CYLD_tf)              
        
        self.loss_TOP_BOTTOM = tf.reduce_mean(tf.square(self.u_TOP_BOTTOM_pred-self.u_TOP_BOTTOM_tf)) \
                        + tf.reduce_mean(tf.square(self.v_TOP_BOTTOM_pred-self.v_TOP_BOTTOM_tf))
        self.loss_CYLD = tf.reduce_mean(tf.square(self.u_CYLD_pred-self.u_CYLD_tf)) \
                        + tf.reduce_mean(tf.square(self.v_CYLD_pred-self.v_CYLD_tf))
        self.loss_CYLD_neumann = tf.reduce_mean(tf.square(self.f_pred_CYLD_dp_dn))
        self.loss_INLET = tf.reduce_mean(tf.square(self.u_INLET_pred-self.u_INLET_tf)) \
                          + tf.reduce_mean(tf.square(self.v_INLET_pred-self.v_INLET_tf))
                         
        self.loss_internal = tf.reduce_mean(tf.square(self.u_internal_pred-self.u_internal_tf)) \
                         + tf.reduce_mean(tf.square(self.v_internal_pred-self.v_internal_tf)) \
                             + tf.reduce_mean(tf.square(self.p_internal_pred-self.p_internal_tf))
                         
        #self.loss_OUTLET = tf.reduce_mean(tf.square(self.p_OUTLET_pred-self.p_OUTLET_tf))
        self.loss_OUTLET_dirichlet = tf.reduce_mean(tf.square(self.p_OUTLET_pred-self.p_OUTLET_d_tf))
        
        self.loss_OUTLET_neumann = tf.reduce_mean(tf.square(self.f_pred_OUTLET_dp_dx))

        self.loss_stress_free = tf.reduce_mean(tf.square(self.bc_1)) + tf.reduce_mean(tf.square(self.bc_2))

        
        if use_internal == 0:
            if use_neural_form==0:
                
                self.loss_bc = (self.cyld_weight*self.loss_CYLD + self.wall_weight*self.loss_TOP_BOTTOM + self.inlet_weight*self.loss_INLET + self.outlet_weight*self.loss_stress_free)
                self.loss = self.loss_f + (self.cyld_weight*self.loss_CYLD + self.wall_weight*self.loss_TOP_BOTTOM + self.inlet_weight*self.loss_INLET + self.outlet_weight*self.loss_stress_free)
                
                #self.loss=self.loss_f + beta*(self.loss_CYLD + self.loss_TOP_BOTTOM +self.loss_INLET +self.loss_stress_free)
            #self.loss = self.loss_f + 0* self.loss_internal + \
            #    2*(self.loss_TOP_BOTTOM + self.loss_INLET + self.loss_OUTLET)
            else:

                self.loss = self.loss_f + beta*(self.loss_CYLD + self.loss_CYLD_neumann + self.loss_OUTLET_neumann + self.loss_OUTLET_dirichlet)
                self.loss_bc = beta*(self.loss_CYLD + self.loss_CYLD_neumann + self.loss_OUTLET_neumann + self.loss_OUTLET_dirichlet)

            #self.loss= self.loss_f + beta*(self.loss_CYLD +  self.loss_stress_free )
                                         
        else:
            
            # self.loss = self.loss_f + 2*(self.loss_TOP_BOTTOM + self.loss_INLET + \
            #             self.loss_internal + self.loss_OUTLET)
                
            self.loss = self.loss_f + beta*(self.loss_CYLD + self.loss_CYLD_neumann + \
                        self.loss_internal + self.loss_OUTLET_neumann + \
                        self.loss_OUTLET_dirichlet)
        
        if debug == 0:
        
            # Optimizer for solution
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                    var_list=self.uv_weights + self.uv_biases,
                                                                    method='L-BFGS-B',
                                                                    options={'maxiter': 100000,
                                                                              'maxfun': 100000,
                                                                              'maxcor': 50,
                                                                              'maxls': 50,
                                                                              'ftol': 1*np.finfo(float).eps}) #The weights and biases are the variables to be updated based on the loss
        
        elif debug == 1:
            
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                    var_list=self.uv_weights + self.uv_biases,
                                                                    method='L-BFGS-B',
                                                                    options={'maxiter': 100,
                                                                              'maxfun': 90,
                                                                              'maxcor': 50,
                                                                              'maxls': 50,
                                                                              'ftol': 1*np.finfo(float).eps})
        #Implementing the Adam Optimizer
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                          var_list=self.uv_weights + self.uv_biases)
                                                  
        #Implementing the yellowfin optimizer                                                 
        self.optimizer_Yellowfin = YFOptimizer()
        self.train_op_Yellowfin = self.optimizer_Yellowfin.minimize(self.loss,
                                                          var_list=self.uv_weights + self.uv_biases)
        
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

        uv_weights = self.sess.run(self.uv_weights)
        uv_biases = self.sess.run(self.uv_biases)

        with open(fileDir, 'wb') as f:
            pickle.dump([uv_weights, uv_biases], f)
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

    def net_uv(self, x, y):
        psips = self.neural_net(tf.concat([x, y], 1), self.uv_weights, self.uv_biases)
        
        # psi = psips[:,0:1]
        # p = psips[:,1:2]
        # s11 = psips[:, 2:3]
        # s22 = psips[:, 3:4]
        # s12 = psips[:, 4:5]
        # u = tf.gradients(psi, y)[0]
        # v = -tf.gradients(psi, x)[0]
        
        #u = psips[:,0:1]
        #v = psips[:,1:2]
        
        if use_neural_form==0:
            u = psips[:,0:1]
            v = psips[:,1:2]

        else:
            dx = xub - xlb
            dy = yub - ylb
            
            axy = (xub - x)/(dx)*u_ref + (x - xlb)/(dx)*u_ref + \
            ((yub - y)/dy)*(u_ref - (((xub - x)/dx)*u_ref + ((x - xlb)/dx)*u_ref)) + \
            ((y - ylb)/dy)*(u_ref -(((xub - x)/dx)*u_ref + ((x - xlb)/dx)*u_ref))
            
            u = axy + psips[:,0:1]*(x - xlb)*(yub - y)*(y - ylb)/(dx*(dy**2))
            
            axy = (xub - x)/(dx)*v_ref + (x - xlb)/(dx)*v_ref + \
            ((yub - y)/dy)*(v_ref - (((xub - x)/dx)*v_ref + ((x - xlb)/dx)*v_ref)) + \
            ((y - ylb)/dy)*(v_ref -(((xub - x)/dx)*v_ref + ((x - xlb)/dx)*v_ref))
            
            v = axy + psips[:,1:2]*(x - xlb)*(yub - y)*(y - ylb)/(dx*(dy**2))
        p = psips[:,2:3]
        s11 = psips[:,3:4]
        s22 = psips[:,4:5]
        s12 = psips[:,5:6]
        
        return u, v, p, s11, s22, s12

    def net_f(self, x, y): #this is the physics governing equation function

        rho=self.rho
        mu=self.mu
        re=self.re
        u, v, p, s11, s22, s12 = self.net_uv(x, y)

        s11_1 = tf.gradients(s11, x)[0] #ds11_dx, it is a list of all the gradients of ds11/dx and we are taking the first result.
        s12_2 = tf.gradients(s12, y)[0]
        s22_2 = tf.gradients(s22, y)[0]
        s12_1 = tf.gradients(s12, x)[0]

        # Plane stress problem
        u_x = tf.gradients(u, x)[0] #this part is the automated differentiation
        u_y = tf.gradients(u, y)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        
        #eqn 1: continuity eqn to ensure the flow is divergent (incompressible)
        f_mass = (u_x + v_y) #the sum of this should be 0, so the loss is automatically anything that is not 0
        

        # f_u:=Sxx_x+Sxy_y
        #eqn 3
        f_u = (u*u_x + v*u_y) - s11_1 - s12_2 #The steady code doesnt have the partial derivative of t
        f_v = (u*v_x + v*v_y) - s12_1 - s22_2
       
        #eqn 4
        f_s11 = -p + (2/re)*u_x - s11
        f_s22 = -p + (2/re)*v_y - s22
        f_s12 = (1/re)*(u_y+v_x) - s12
        
        # tr sigma = (s11+s22)
        f_p = p + (s11+s22)/2
        return f_mass, f_u, f_v, f_s11, f_s22, f_s12, f_p #returns the physics losses 6 values

    def stressfree_bc(self, x,y):
        u, v, p, s11, s22, s12 = self.net_uv(x, y)
        re=self.re
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]

        v_x = tf.gradients(v, x)[0]

        #Sxx
        bc_1 = (2/re)*(u_x) - p 
        bc_2 = (1/re)*(u_y + v_x)
        return bc_1, bc_2

    def net_neumann_outlet(self, x, y):
        #dp/dx outlet
        
        _, _, p, _, _, _ = self.net_uv(x, y)
        
        #u_x = tf.gradients(u, x)[0]
        #neumann BC
        #f_du_dx = u_x
        f_dp_dx = tf.gradients(p, x)[0]

        #return f_du_dx, f_dp_dx
        return f_dp_dx

    def net_neumann_normal(self, x, y,cyld_x_center,cyld_y_center,radius):
        #dp/dn for cyld
        
        _, _, p, _, _, _ = self.net_uv(x, y)
        
        #neumann BC
        #f_du_dx = u_x
        f_dp_dn = tf.gradients(p, x)[0]*(x - cyld_x_center)/radius  +  tf.gradients(p, y)[0]*(y - cyld_y_center)/radius
        
        return f_dp_dn

    def callback(self, loss):
        self.count = self.count+1
        self.loss_rec.append(loss)
        if self.count % 40 == 0:
            print('{} th iterations, Loss: {}'.format(self.count, loss))

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
        
    def train(self, iter, learning_rate, initial):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_CYLD_tf: self.x_CYLD, self.y_CYLD_tf: self.y_CYLD, self.u_CYLD_tf: self.u_CYLD, self.v_CYLD_tf: self.v_CYLD,
                   self.x_TOP_BOTTOM_tf: self.x_TOP_BOTTOM, self.y_TOP_BOTTOM_tf: self.y_TOP_BOTTOM, self.u_TOP_BOTTOM_tf: self.u_TOP_BOTTOM, self.v_TOP_BOTTOM_tf: self.v_TOP_BOTTOM,                   
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_internal_tf: self.x_internal, self.y_internal_tf: self.y_internal, self.u_internal_tf: self.u_internal, self.v_internal_tf: self.v_internal, self.p_internal_tf: self.p_internal,
                   self.x_OUTLET_d_tf: self.x_OUTLET_d, self.y_OUTLET_d_tf: self.y_OUTLET_d, self.p_OUTLET_d_tf: self.p_OUTLET_d,
                   self.x_OUTLET_n_tf: self.x_OUTLET_n, self.y_OUTLET_n_tf: self.y_OUTLET_n, self.p_OUTLET_n_tf: self.p_OUTLET_n,
                   self.learning_rate: learning_rate, self.wall_weight:initial, self.cyld_weight:initial, self.outlet_weight: initial, self.inlet_weight:initial}

        loss_CYLD = []
        loss_CYLD_neumann = []
        loss_TOP_BOTTOM = []
        loss_f = []
        loss_INLET = []
        loss_internal = []
        loss_OUTLET_dirichlet = []
        loss_OUTLET_neumann = []
        iterate_count = []
        loss_list = []
        loss_OUTLET = [] 

        for it in range(iter):

            if optimizer == 0:
                self.sess.run(self.train_op_Adam, tf_dict)

            if optimizer == 1: 
                self.sess.run(self.train_op_Yellowfin, tf_dict)  

            # Print
            if it % 100 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                
                if (self.sess.run(self.loss_TOP_BOTTOM,tf_dict) / self.sess.run(self.loss_f,tf_dict)) >=10:
                    tf_dict[self.wall_weight]= 0.1
                if (self.sess.run(self.loss_TOP_BOTTOM,tf_dict) / self.sess.run(self.loss_f,tf_dict)) <=0.1:
                    tf_dict[self.wall_weight]= 10
                if (self.sess.run(self.loss_TOP_BOTTOM,tf_dict) / self.sess.run(self.loss_f,tf_dict)) >=0.1 and (self.sess.run(self.loss_TOP_BOTTOM,tf_dict) / self.sess.run(self.loss_f,tf_dict)) <=10 :
                    tf_dict[self.wall_weight]= 2        
                if (self.sess.run(self.loss_CYLD,tf_dict) / self.sess.run(self.loss_f,tf_dict)) >=10:
                    tf_dict[self.cyld_weight]= 0.1
                if (self.sess.run(self.loss_CYLD,tf_dict) / self.sess.run(self.loss_f,tf_dict)) <=0.1:
                    tf_dict[self.cyld_weight]= 10
                if (self.sess.run(self.loss_CYLD,tf_dict) / self.sess.run(self.loss_f,tf_dict)) >=0.1 and (self.sess.run(self.loss_CYLD,tf_dict) / self.sess.run(self.loss_f,tf_dict)) <=10 :
                    tf_dict[self.cyld_weight]= 2
                if (self.sess.run(self.loss_INLET,tf_dict) / self.sess.run(self.loss_f,tf_dict)) >=10:
                    tf_dict[self.inlet_weight]= 0.1
                if (self.sess.run(self.loss_INLET,tf_dict) / self.sess.run(self.loss_f,tf_dict)) <=0.1:
                    tf_dict[self.inlet_weight]= 10
                if (self.sess.run(self.loss_INLET,tf_dict) / self.sess.run(self.loss_f,tf_dict)) >=0.1 and (self.sess.run(self.loss_INLET,tf_dict) / self.sess.run(self.loss_f,tf_dict)) <=10 :
                    tf_dict[self.inlet_weight]= 2 
                if (self.sess.run(self.loss_stress_free,tf_dict) / self.sess.run(self.loss_f,tf_dict)) >=10:
                    tf_dict[self.outlet_weight]= 0.1
                if (self.sess.run(self.loss_stress_free,tf_dict) / self.sess.run(self.loss_f,tf_dict)) <=0.1:
                    tf_dict[self.outlet_weight]= 10
                if (self.sess.run(self.loss_stress_free,tf_dict) / self.sess.run(self.loss_f,tf_dict)) >=0.1 and (self.sess.run(self.loss_stress_free,tf_dict) / self.sess.run(self.loss_f,tf_dict)) <=10 :
                    tf_dict[self.outlet_weight]= 2
                
                print('It: %d, Loss: %.3e, Wall: %.3e, Cyld: %.3e, Inlet: %.3e, Outlet: %.3e' %
                      (it, loss_value, self.sess.run(self.loss_TOP_BOTTOM, tf_dict),self.sess.run(self.loss_CYLD, tf_dict), self.sess.run(self.loss_INLET, tf_dict), self.sess.run(self.loss_stress_free, tf_dict) ))

            loss_CYLD.append(self.sess.run(self.loss_CYLD, tf_dict))
            loss_CYLD_neumann.append(self.sess.run(self.loss_CYLD_neumann, tf_dict))
            loss_TOP_BOTTOM.append(self.sess.run(self.loss_TOP_BOTTOM, tf_dict))            
            loss_f.append(self.sess.run(self.loss_f, tf_dict))
            self.loss_rec.append(self.sess.run(self.loss, tf_dict))
            loss_INLET.append(self.sess.run(self.loss_INLET, tf_dict))
            loss_internal.append(self.sess.run(self.loss_internal, tf_dict))
            loss_OUTLET_dirichlet.append(self.sess.run(self.loss_OUTLET_dirichlet, tf_dict))
            loss_OUTLET_neumann.append(self.sess.run(self.loss_OUTLET_neumann, tf_dict))
            iterate_count.append(it+1)
            loss_list.append(self.sess.run(self.loss, tf_dict))
            loss_OUTLET.append(self.sess.run(self.loss_stress_free,tf_dict))

        # return loss_CYLD, loss_TOP_BOTTOM, loss_INLET, loss_OUTLET, loss_f, self.loss
        return loss_CYLD, loss_CYLD_neumann, loss_internal, \
        loss_OUTLET_dirichlet,loss_OUTLET_neumann, loss_f, self.loss, iterate_count, loss_list, loss_TOP_BOTTOM, loss_INLET, loss_OUTLET

    def train_bfgs(self,initial):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_CYLD_tf: self.x_CYLD, self.y_CYLD_tf: self.y_CYLD, self.u_CYLD_tf: self.u_CYLD, self.v_CYLD_tf: self.v_CYLD,
                   self.x_TOP_BOTTOM_tf: self.x_TOP_BOTTOM, self.y_TOP_BOTTOM_tf: self.y_TOP_BOTTOM, self.u_TOP_BOTTOM_tf: self.u_TOP_BOTTOM, self.v_TOP_BOTTOM_tf: self.v_TOP_BOTTOM,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_internal_tf: self.x_internal, self.y_internal_tf: self.y_internal, self.u_internal_tf: self.u_internal, self.v_internal_tf: self.v_internal, self.p_internal_tf: self.p_internal,
                   self.x_OUTLET_d_tf: self.x_OUTLET_d, self.y_OUTLET_d_tf: self.y_OUTLET_d, self.p_OUTLET_d_tf: self.p_OUTLET_d,
                   self.x_OUTLET_n_tf: self.x_OUTLET_n, self.y_OUTLET_n_tf: self.y_OUTLET_n, self.p_OUTLET_n_tf: self.p_OUTLET_n,
                   self.wall_weight:initial, self.cyld_weight:initial, self.outlet_weight: initial, self.inlet_weight:initial}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, x_star, y_star):
        u_star = self.sess.run(self.u_pred, {self.x_tf: x_star, self.y_tf: y_star})
        v_star = self.sess.run(self.v_pred, {self.x_tf: x_star, self.y_tf: y_star})
        p_star = self.sess.run(self.p_pred, {self.x_tf: x_star, self.y_tf: y_star})
        return u_star, v_star, p_star

    def getloss(self):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_CYLD_tf: self.x_CYLD, self.y_CYLD_tf: self.y_CYLD, self.u_CYLD_tf: self.u_CYLD, self.v_CYLD_tf: self.v_CYLD,
                   self.x_TOP_BOTTOM_tf: self.x_TOP_BOTTOM, self.y_TOP_BOTTOM_tf: self.y_TOP_BOTTOM, self.u_TOP_BOTTOM_tf: self.u_TOP_BOTTOM, self.v_TOP_BOTTOM_tf: self.v_TOP_BOTTOM,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_internal_tf: self.x_internal, self.y_internal_tf: self.y_internal, self.u_internal_tf: self.u_internal, self.v_internal_tf: self.v_internal, self.p_internal_tf: self.p_internal,
                   self.x_OUTLET_d_tf: self.x_OUTLET_d, self.y_OUTLET_d_tf: self.y_OUTLET_d, self.p_OUTLET_d_tf: self.p_OUTLET_d,
                   self.x_OUTLET_n_tf: self.x_OUTLET_n, self.y_OUTLET_n_tf: self.y_OUTLET_n, self.p_OUTLET_n_tf: self.p_OUTLET_n}

        loss_f = self.sess.run(self.loss_f, tf_dict)
        loss_CYLD = self.sess.run(self.loss_CYLD, tf_dict)
        loss_CYLD_neumann = self.sess.run(self.loss_CYLD_neumann, tf_dict)
        #loss_TOP_BOTTOM = self.sess.run(self.loss_TOP_BOTTOM, tf_dict)
        #loss_INLET = self.sess.run(self.loss_INLET, tf_dict)
        loss_internal = self.sess.run(self.loss_internal, tf_dict)
        loss = self.sess.run(self.loss, tf_dict)
        loss_OUTLET_dirichlet = self.sess.run(self.loss_OUTLET_dirichlet, tf_dict)
        loss_OUTLET_neumann = self.sess.run(self.loss_OUTLET_neumann, tf_dict)

        # return loss_CYLD, loss_TOP_BOTTOM, loss_INLET, loss_OUTLET, loss_f, loss
        return loss_CYLD, loss_CYLD_neumann, loss_OUTLET_dirichlet, \
        loss_OUTLET_neumann, loss_OUTLET, loss_f, loss

def DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    '''
    delete points within cylinder
    '''
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst>r,:]

def mse(ite,los,cylinder, inlet, outlet, residual, wall):
    plt.figure(3)
    plt.plot(ite,los, label="Total Loss")
    plt.plot(ite, cylinder, label="Cylinder Loss")
    plt.plot(ite,inlet, label="Inlet")
    plt.plot(ite, outlet, label="Outlet")
    plt.plot(ite, wall, label="Top Bottom")
    plt.plot(ite, residual, label="Residual")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Loss (MSE)")
    plt.legend(loc="lower left")
    plt.title("Mean Square Error Loss")
    plt.savefig('./Non_Dim_Output/mse_plot', dpi=360)
    plt.close('all')
    error_df=pd.DataFrame({"Iteration":ite, "Loss": los, "Cylinder": cylinder, "Inlet": inlet, "Outlet": outlet, "Residual": residual, "Wall": wall})
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
    
    [x_ref, y_ref, u_ref, v_ref, p_ref] = field_ref
    [x_MIXED, y_MIXED, u_MIXED, v_MIXED, p_MIXED] = field_MIXED

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

    cf = ax[3, 0].scatter(x_MIXED, y_MIXED, c=p_MIXED, vmin=-1., vmax=1., alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=int(s))
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

    cf = ax[3, 1].scatter(x_ref, y_ref, c=p_ref, vmin=-1., vmax=1., alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=s)
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

def postProcess2(filename, field_ref, field_MIXED, s=2, alpha=0.5, marker='o'):
    [x_REF, y_REF, u_REF, v_REF, p_REF] = field_ref #from open foam
    [x_MIXED, y_MIXED, u_MIXED, v_MIXED, p_MIXED] = field_MIXED #From trained data
    x,y= np.meshgrid(np.linspace(-5.0, 9.33, 251),np.linspace(-5,5,101))
    plt.figure(1)
    fig, ax = plt.subplots(nrows=2, ncols=1)
    u_mixed=[]
    v_mixed=[]
    check=[]
    count=0
    for i in range (0,len(y_MIXED)):
        if y_MIXED[i][0] not in check:
            check.append(y_MIXED[i][0])
            u_mixed.append([])
            v_mixed.append([])
            count+=1
        dst=((x_MIXED[i][0]-0.0)**2+(y_MIXED[i][0]-0.0)**2)**0.5
        if dst<=0.5:
            u_mixed[count-1].append(0.0001)
            v_mixed[count-1].append(0.0001)
        else:
            u_mixed[count-1].append(u_MIXED[i][0])
            v_mixed[count-1].append(v_MIXED[i][0])
    u_mixed=np.array(u_mixed)
    v_mixed=np.array(v_mixed)

    
    lic_mixed = lic.lic(u_mixed, v_mixed, length=30, contrast=True)
    cf = ax[0].imshow(lic_mixed, origin='lower', cmap='viridis', extent=[-5,9.33,-5,5])
    fig.colorbar(cf, ax=ax[0], pad=0.04)
    

    u_ref=[]
    v_ref=[]
    check1=[]
    count1=0
    for i in range (0,len(y_REF)):
        if y_REF[i][0] not in check1:
            check1.append(y_REF[i][0])
            u_ref.append([])
            v_ref.append([])
            count1+=1
        dst=((x_REF[i][0]-0.0)**2+(y_REF[i][0]-0.0)**2)**0.5
        if dst<=0.5:
            u_ref[count1-1].append(0.0001)
            v_ref[count1-1].append(0.0001)
        else:
            u_ref[count1-1].append(u_REF[i][0])
            v_ref[count1-1].append(v_REF[i][0])
    u_ref=np.array(u_ref)
    v_ref=np.array(v_ref)
    
    lic_ref = lic.lic(u_ref, v_ref, length=50, contrast=True)
    
    cf = ax[1].imshow(lic_ref, origin='lower', cmap='viridis', extent=[-5,9.33,-5,5])
    fig.colorbar(cf, ax=ax[1], pad=0.04)
    plt.tight_layout()

    #plt.savefig('./uvp_mixed_uvp.png', dpi=360)    
    plt.savefig('./Non_Dim_Output/'+filename + '.png',dpi=400)
    
    plt.figure(2)
    skip=(slice(None,None,8),slice(None,None,8))
    fig1, ax1 = plt.subplots(nrows=2, ncols=2)
    cf=ax1[0,0].streamplot(x, y, u_mixed, v_mixed, color=u_mixed, linewidth=2, cmap='autumn')
    cf=ax1[0,1].streamplot(x, y, u_ref, v_ref, color=u_mixed, linewidth=2, cmap='autumn')
    cf=ax1[1,0].quiver(x[skip],y[skip],u_mixed[skip],v_mixed[skip], pivot="mid", cmap="autumn")
    cf=ax1[1,1].quiver(x[skip],y[skip],u_ref[skip],v_ref[skip], pivot="mid", cmap="autumn")
    plt.tight_layout()
    plt.savefig('./Non_Dim_Output/Quiver Streamline.png',dpi=400)
    plt.close("all")
    

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
    

    [x_ref, y_ref, u_ref, v_ref, p_ref] = field_ref #from open foam
    [x_MIXED, y_MIXED, u_MIXED, v_MIXED, p_MIXED] = field_MIXED #From trained data
    [x_diff, y_diff, u_diff, v_diff, p_diff, velo_diff] = field_diff
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

    cf = ax[3, 0].scatter(x_MIXED, y_MIXED, c=p_MIXED, vmin=-1., vmax=1., alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=int(s))
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
    
    cf = ax[3, 1].scatter(x_ref, y_ref, c=p_ref, vmin=-1., vmax=1., alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=s)
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
                
        ax[0, 2].set_title('X-velocity Diff')
        cf = ax[0, 2].scatter(x_diff, y_diff, c=u_diff, vmin=-0.15, vmax=0.15, alpha=alpha, 
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
                
        ax[1, 2].set_title('Y-velocity Diff')
        cf = ax[1, 2].scatter(x_diff, y_diff, c=v_diff, vmin=-0.15, vmax=0.15, alpha=alpha, 
                              edgecolors='none', cmap='viridis', marker=marker, s=int(s))
        # cf = ax[1,2].imshow(v_diff, vmin=-1e-2, vmax=1e-2, extent=[xmin,xmax,ymin,ymax], 
        #                 origin='lower', cmap='viridis')
        #circ8 = plt.Circle((0,0), radius=0.5, color='w')
        #ax[1,2].add_patch(circ8)
        
    elif diff == 'percentage':
        
        ax[1, 2].set_title('v diff (%)')
        cf = ax[1, 2].scatter(x_diff, y_diff, c=v_diff, vmin=-0.05, vmax=0.05, alpha=alpha, 
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
                
        ax[2, 2].set_title('Velocity Magnitude Diff')
        cf = ax[2, 2].scatter(x_diff, y_diff, c=velo_diff, alpha=alpha, 
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
                
        ax[3, 2].set_title('Pressure Diff')
        cf = ax[3, 2].scatter(x_diff, y_diff, c=p_diff, vmin=-0.1, vmax=0.1, alpha=alpha, 
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
    _, _, p_pred = model.predict(x_loc, y_loc)
    
    for i in range(points):
        
        dx = delta*np.cos(theta[i]); dy = delta*np.sin(theta[i])
        fx = fx - p_pred[i]*dx; fy = fy - p_pred[i]*dy
        
    error_df=pd.DataFrame({"Theta": theta, "Pressure": p_pred.flatten()})
    error_df.transpose().to_csv('./Non_Dim_Output/Pressure_plot.csv', index=False)

    fig = plt.figure()
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
    plt.savefig('./Non_Dim_Output/pressure_vs_theta.png',dpi=360)
    plt.show()
    
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
        
        u_pred, v_pred, _ = model.predict(test_points_x, test_points_y)
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

    xlb = -5
    ylb = -5
    xub = 9.33
    yub = 5

    # Refine Points
    top_bottom_pts = 401
    inlet_pts = 801
    outlet_pts = 801
    cylinder_pts = 801
    collo_pts = 10000
    refine_pts = 2500

    # Domain bounds
    lb = np.array([xlb, ylb])   #lower left x,y coord of boundary
    lbo= np.array([xlb, 0]) #in crafting symmetric bounds
    ub = np.array([xub, yub])  #upper right x,y coord of boundary
    
    u_ref = 1.0
    v_ref = 0.0
    # Network configuration
    uv_layers = [2] + 8*[40] + [6]  # 2 i/p, 8 layers x 40 neurons each, 6 o/p

    # TOP_BOTTOM
    TOP = [xlb, yub] + [xub - xlb, 0.0] * lhs(2, top_bottom_pts) #lhs = Latin-Hypercube from pyDOE
    #lhs(2,441) = generate 2 variables, 441 pts each
    x_TOP = TOP[:,0:1]
    y_TOP = TOP[:,1:2]
    u_TOP = np.zeros_like(x_TOP)
    v_TOP = np.zeros_like(x_TOP)
    u_TOP[:] = u_ref
    v_TOP[:] = 0.
    #TOP = np.concatenate((TOP, u_TOP, v_TOP), 1)
    
    #[0.0, 4.1] + [11.0, 0.0]* = starting + delta * to get values within range
    BOTTOM = [xlb, ylb] + [xub - xlb, 0.0] * lhs(2, top_bottom_pts)
    x_BOTTOM = BOTTOM[:,0:1]
    y_BOTTOM = BOTTOM[:,1:2]
    u_BOTTOM = np.zeros_like(x_BOTTOM)
    v_BOTTOM = np.zeros_like(x_BOTTOM)
    u_BOTTOM[:] = u_ref
    v_BOTTOM[:] = 0.
    #BOTTOM = np.concatenate((BOTTOM, u_BOTTOM, v_BOTTOM), 1)

    TOP_BOTTOM_x = np.concatenate((x_TOP, x_BOTTOM), 0)
    TOP_BOTTOM_y = np.concatenate((y_TOP, y_BOTTOM), 0)
    TOP_BOTTOM_u = np.concatenate((u_TOP, u_BOTTOM), 0)
    TOP_BOTTOM_v = np.concatenate((v_TOP, v_BOTTOM), 0)
    TOP_BOTTOM = np.concatenate((TOP_BOTTOM_x, TOP_BOTTOM_y, TOP_BOTTOM_u, TOP_BOTTOM_v), 1)
    # INLET = [x, y, u, v]
    
    INLET = [xlb, ylb] + [0.0, yub-ylb] * lhs(2, inlet_pts)
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

    #dirichlet
    OUTLET_d = [xub, ylb] + [0.0, 0.0] * lhs(2, 1)
    x_OUTLET_d = OUTLET_d[:,0:1]
    y_OUTLET_d = OUTLET_d[:,1:2]
    p_OUTLET_d = np.zeros_like(y_OUTLET_d)
    p_OUTLET_d[:] = 0.
    OUTLET_d = np.concatenate((OUTLET_d, p_OUTLET_d), 1)
    
    #neumann
    OUTLET_n = [xub, ylb] + [0.0, yub - ylb] * lhs(2, outlet_pts)
    x_OUTLET_n = OUTLET_n[:,0:1]
    y_OUTLET_n = OUTLET_n[:,1:2]
    p_OUTLET_n = np.zeros_like(y_OUTLET_n)
    p_OUTLET_n[:] = 0.
    OUTLET_n = np.concatenate((x_OUTLET_n, y_OUTLET_n, p_OUTLET_n), 1)

    # Cylinder surface
    r = 0.5
    theta = [0.0] + [2*np.pi] * lhs(1, cylinder_pts)
    x_CYLD = np.multiply(r, np.cos(theta))+0.0
    y_CYLD = np.multiply(r, np.sin(theta))+0.0
    u_CYLD = np.zeros_like(x_CYLD)
    v_CYLD = np.zeros_like(x_CYLD)
    u_CYLD[:] = 0.
    v_CYLD[:] = 0.
    CYLD = np.concatenate((x_CYLD, y_CYLD, u_CYLD, v_CYLD), 1)  #x,y coord of circle

    if CFD_grid ==0:
        # Collocation point for equation residual
        XY_c_upper = lbo + (ub - lbo) * lhs(2, int(collo_pts/2))   #x,y coord for entire domain
        XY_c_lower = np.concatenate((XY_c_upper[:,0:1],-1*XY_c_upper[:,1:2]),1) #create the lower symmetrical
        XY_c= np.concatenate((XY_c_upper, XY_c_lower), 0)
        XY_c_refine_upper = [-1.0, 0] + [7.0, 1.0] * lhs(2, int(refine_pts/2))  #upper symmetrical for refined domain
        XY_c_refine_lower= np.concatenate((XY_c_refine_upper[:,0:1],-1*XY_c_refine_upper[:,1:2]),1) #lower symmetrical for lower domain
        XY_c_refine=np.concatenate((XY_c_refine_upper, XY_c_refine_lower),0)
        XY_c_refine=DelCylPT(XY_c_refine, xc=0.0, yc= 0.0,r=0.5)
        XY_c_plot= DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.5)  
        XY_c = np.concatenate((XY_c, XY_c_refine), 0)
        XY_c = DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.5)   #remove pts inside circle
    
    elif CFD_grid==1:
        #Collocation points with reference to CFD Grids
        dataset = np.loadtxt('Openfoam Dr Tay.csv',delimiter=',',skiprows=6)
        XY_c = np.concatenate((dataset[:,0:1],dataset[:,1:2]),1)

    # Visualize the collocation points
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(XY_c[:,0:1], XY_c[:,1:2], marker='o', alpha=0.1 ,color='blue', label="Collocation Points")
    plt.scatter(XY_c_refine[:,0:1], XY_c_refine[:,1:2], marker="+", alpha =0.3, color="yellow", label="Refined Points")
    plt.scatter(CYLD[:,0:1], CYLD[:,1:2], marker='o', alpha=0.2 , color='green')
    plt.scatter(TOP_BOTTOM[:,0:1], TOP_BOTTOM[:,1:2], marker='o', alpha=0.2 , color='green', label="Boundary Points")
    plt.scatter(OUTLET_n[:, 0:1], OUTLET_n[:, 1:2], marker='o', alpha=0.2, color='green')
    plt.scatter(OUTLET_d[:, 0:1], OUTLET_d[:, 1:2], marker='o', alpha=0.2, color='black')  
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
        model = PINN_laminar_flow(XY_c, INLET, internal, OUTLET_d, OUTLET_n, CYLD, TOP_BOTTOM, uv_layers, lb, ub, ExistModel = use_old_model, uvDir = 'uvNN.pickle')

        start_time = time.time()
        
        if debug == 0:
            if use_old_model==1:
                loss_CYLD, loss_CYLD_neumann, loss_internal, loss_OUTLET_dirichlet, \
                loss_OUTLET_neumann, loss_f, loss, iterate_count, loss_list, loss_TOP_BOTTOM, loss_INLET, loss_OUTLET = model.train(iter=2000, learning_rate=0.0005, initial=2)
            else:
                loss_CYLD, loss_CYLD_neumann, loss_internal, loss_OUTLET_dirichlet, \
                loss_OUTLET_neumann, loss_f, loss, iterate_count, loss_list, loss_TOP_BOTTOM, loss_INLET, loss_OUTLET = model.train(iter=10000, learning_rate=0.0005, initial=2)
        
        elif debug == 1:
            
                loss_CYLD, loss_CYLD_neumann, loss_internal, loss_OUTLET_dirichlet, \
                loss_OUTLET_neumann, loss_f, loss, iterate_count, loss_list, loss_TOP_BOTTOM, loss_INLET, loss_OUTLET = model.train(iter=60, learning_rate=0.0005, initial=1)
        
        model.train_bfgs(initial=1)
        print("--- %s seconds ---" % (time.time() - start_time))

        # Save neural network
        model.save_NN('uvNN.pickle')

        # Save loss history
        with open('loss_history.pickle', 'wb') as f:
            pickle.dump(model.loss_rec, f)
            
        if data_ref == 'fluent':

            #field_ref = np.loadtxt('Fluent Normal Domain.csv',delimiter=',',skiprows=6)
            field_ref = np.loadtxt('Re=40.csv',delimiter=',',skiprows=6)
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
        if CFD_grid ==0:
            x_PINN = np.linspace(xlb, xub, 251)   #251 equally spaced points means 250 grid spaces
            x_PINN1 = np.linspace(xlb, xub, 251)
            y_PINN = np.linspace(ylb, yub, 101)    #100 grid spaces for y axis
            y_PINN1 = np.linspace(ylb, yub, 101)        
            x_PINN, y_PINN = np.meshgrid(x_PINN, y_PINN) #Each row of x_PINN is the same and y_PINN varies
            x_PINN = x_PINN.flatten()[:, None]
            y_PINN = y_PINN.flatten()[:, None]
            x_PINN1, y_PINN1 = np.meshgrid(x_PINN1, y_PINN1) #Each row of x_PINN is the same and y_PINN varies
            x_PINN1 = x_PINN1.flatten()[:, None]
            y_PINN1 = y_PINN1.flatten()[:, None]
            dst = ((x_PINN-0.0)**2+(y_PINN-0.0)**2)**0.5
            x_PINN = x_PINN[dst >= 0.5]
            y_PINN = y_PINN[dst >= 0.5]
            x_PINN = x_PINN.flatten()[:, None]
            y_PINN = y_PINN.flatten()[:, None]
            u_PINN, v_PINN, p_PINN = model.predict(x_PINN, y_PINN) 
            field_MIXED = [x_PINN, y_PINN, u_PINN, v_PINN, p_PINN]
            u_PINN1, v_PINN1, p_PINN1 = model.predict(x_PINN1, y_PINN1) 
            field_MIXED1 = [x_PINN1, y_PINN1, u_PINN1, v_PINN1, p_PINN1]
        elif CFD_grid ==1:
            u_PINN, v_PINN, p_PINN = model.predict(x_ref, y_ref) 
            field_MIXED = [x_ref, y_ref, u_PINN, v_PINN, p_PINN]
        
        field_MIXED_csv = np.asarray(field_MIXED)
        field_MIXED_csv = field_MIXED_csv[:,:,0]
        field_MIXED_csv = np.transpose(field_MIXED_csv)
        df = pd.DataFrame(field_MIXED_csv)
        df.to_csv('uvp_mixed.csv',index=False)
        Cov = pd.read_csv("uvp_mixed.csv", names=["x", "y", "u", "v","p"])
        Cov.to_csv('uvp_mixed.csv',index=False)
        lines = open('uvp_mixed.csv', 'r').readlines()
        del lines[1]
        open('uvp_mixed.csv', 'w').writelines(lines)
        if CFD_grid ==0:
            u_ref_int = scipy.interpolate.griddata((x_ref.flatten(),y_ref.flatten()),u_ref.flatten() , (x_PINN,y_PINN),method='cubic')
            v_ref_int = scipy.interpolate.griddata((x_ref.flatten(),y_ref.flatten()),v_ref.flatten() , (x_PINN,y_PINN),method='cubic')
            p_ref_int = scipy.interpolate.griddata((x_ref.flatten(),y_ref.flatten()),p_ref.flatten() , (x_PINN,y_PINN),method='cubic')
            field_ref_int = [x_PINN, y_PINN, u_ref_int, v_ref_int, p_ref_int]
        elif CFD_grid ==1:
            field_ref_int = [x_ref, y_ref, u_ref, v_ref, p_ref]

        # Plot the comparison of u, v, p
        postProcess('uvp_PINN_vs_OF_SBC', xmin=xlb, xmax=xub, ymin=ylb, ymax=yub, field_ref=field_ref_int, field_MIXED=field_MIXED, s=5, alpha=1)
        #postProcess2('LIC Comparison', field_ref=field_ref_int, field_MIXED=field_MIXED1, s=5, alpha=1)
        mse(ite=iterate_count, los=loss_list, cylinder=loss_CYLD, inlet=loss_INLET, outlet = loss_OUTLET, residual = loss_f, wall=loss_TOP_BOTTOM)      
        if diff == 'absolute':
            if CFD_grid ==1:
                u_diff = u_PINN - u_ref
                v_diff = v_PINN - v_ref
                p_diff = p_PINN - p_ref
                velo_diff = ((u_PINN**2+v_PINN**2)**0.5) - (u_ref**2+v_ref**2)**0.5
            elif CFD_grid ==0:
                u_diff = u_PINN - u_ref_int
                v_diff = v_PINN - v_ref_int
                p_diff = p_PINN - p_ref_int
                velo_diff = ((u_PINN**2+v_PINN**2)**0.5) - (u_ref_int**2+v_ref_int**2)**0.5          
            
        elif diff == 'percentage':
            
            u_ref_int = u_ref + 1e-10 #prevent 0 divide
            v_ref_int = v_ref + 1e-10 #prevent 0 divide
            p_ref_int = p_ref + 1e-10 #prevent 0 divide
        
            u_diff = 100.*(u_PINN - u_ref_int)/u_ref_int
            v_diff = 100.*(v_PINN - v_ref_int)/v_ref_int
            p_diff = 100.*(p_PINN - p_ref_int)/p_ref_int
            
            # u_diff_xy = 100.*(u_pred_xy - u_ref_xy)/u_ref_xy
            # v_diff_xy = 100.*(v_pred_xy - v_ref_xy)/v_ref_xy
            # p_diff_xy = 100.*(p_pred_xy - p_ref_xy)/p_ref_xy
    if CFD_grid ==0:  
        field_diff = [x_PINN, y_PINN, u_diff, v_diff, p_diff, velo_diff]
    elif CFD_grid==1:
        field_diff = [x_ref, y_ref, u_diff, v_diff, p_diff, velo_diff]
    
    postProcess3('uvp_PINN_vs_OF_diff_SBC', xmin=xlb, xmax=xub,ymin=ylb, ymax=yub, field_ref=field_ref_int,field_MIXED=field_MIXED, field_diff=field_diff, s=5, alpha=1)
    fx_p,fy_p = pressure_forces(surface_pts)
    fx_v,fy_v = model.viscous_forces_autodiff(surface_pts)
    
    fx = fx_p + fx_v
    fy = fy_p + fy_v
    print(fx_p,fx_v,fx)
    print(fy_p,fy_v,fy)