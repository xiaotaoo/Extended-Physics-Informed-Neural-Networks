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
import scipy.interpolate
from scipy.interpolate import griddata

#flow past cylinder with uniform U, upper and lower wall BC
#full u,v specified BC
#top, bottom u=1, v=0
#domain size xy = -5,-5 to 9.333/5
#added p_outlet
#seperate WALL into actual cylinder wall, TOP_BOTTOM BC -< not done yet
#also use internal pts for training
#remove xyc

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

data_ref = 'openfoam'
use_old_model = 0
debug = 0
#activation function = 0 (tanh), 1 (ReLU), 2 (LeakyReLU), 3 (SELU)
activation_fn = 0
#compare diff = absolute or percentage or mean sq error
diff = 'absolute'
#amt of training data of OF to use
training_size = 0.1
#whether to use OF internal pts for training
use_internal = 0
#no of surface points
surface_pts = 200
rho = 1.0
mu = 0.025
#small delta for du/dx etc approx
delta_dist = 1e-4

class PINN_laminar_flow:
    # Initialize the class
    def __init__(self, Collo, INLET, internal, OUTLET, WALL, uv_layers, lb, ub, ExistModel=0, uvDir=''):

        # Count for callback function
        self.count=0

        # Bounds
        self.lb = lb
        self.ub = ub

        # Mat. properties
        #self.rho = 1.0
        #self.mu = 0.02
        #self.mu = 0.025
        
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

        self.x_OUTLET = OUTLET[:, 0:1]
        self.y_OUTLET = OUTLET[:, 1:2]
        self.p_OUTLET = OUTLET[:, 2:3]

        self.x_WALL = WALL[:, 0:1]
        self.y_WALL = WALL[:, 1:2]
        self.u_WALL = WALL[:, 2:3]
        self.v_WALL = WALL[:, 3:4]
        
        # self.x_CYLD = CYLD[:, 0:1]
        # self.y_CYLD = CYLD[:, 1:2]
        # self.u_CYLD = CYLD[:, 2:3]
        # self.v_CYLD = CYLD[:, 3:4]

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
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        self.x_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.x_WALL.shape[1]])
        self.y_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.y_WALL.shape[1]])
        self.u_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.u_WALL.shape[1]])
        self.v_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.v_WALL.shape[1]])
        
        # self.x_CYLD_tf = tf.placeholder(tf.float32, shape=[None, self.x_CYLD.shape[1]])
        # self.y_CYLD_tf = tf.placeholder(tf.float32, shape=[None, self.y_CYLD.shape[1]])
        # self.u_CYLD_tf = tf.placeholder(tf.float32, shape=[None, self.u_CYLD.shape[1]])
        # self.v_CYLD_tf = tf.placeholder(tf.float32, shape=[None, self.v_CYLD.shape[1]])

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

        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        # tf graphs
        self.u_pred, self.v_pred, self.p_pred, _, _, _ = self.net_uv(self.x_tf, self.y_tf)
        self.f_pred_mass, self.f_pred_u, self.f_pred_v, self.f_pred_s11, \
            self.f_pred_s22, self.f_pred_s12, self.f_pred_p = \
                self.net_f(self.x_c_tf, self.y_c_tf)
        
        self.u_WALL_pred, self.v_WALL_pred, _, _, _, _ = self.net_uv(self.x_WALL_tf, self.y_WALL_tf)
        #self.u_CYLD_pred, self.v_CYLD_pred, _, _, _, _ = self.net_uv(self.x_CYLD_tf, self.y_CYLD_tf)
        self.u_INLET_pred, self.v_INLET_pred, _, _, _, _ = self.net_uv(self.x_INLET_tf, self.y_INLET_tf)
        self.u_internal_pred, self.v_internal_pred, self.p_internal_pred, _, _, _ = self.net_uv(self.x_internal_tf, self.y_internal_tf)
        _, _, self.p_OUTLET_pred, _, _, _ = self.net_uv(self.x_OUTLET_tf, self.y_OUTLET_tf)

        self.loss_f = tf.reduce_mean(tf.square(self.f_pred_mass)) \
                      + tf.reduce_mean(tf.square(self.f_pred_u)) \
                      + tf.reduce_mean(tf.square(self.f_pred_v)) \
                      + tf.reduce_mean(tf.square(self.f_pred_s11))\
                          + tf.reduce_mean(tf.square(self.f_pred_p)) \
                              + tf.reduce_mean(tf.square(self.f_pred_s22)) \
                                  + tf.reduce_mean(tf.square(self.f_pred_s12)) 
                        
        
                          
                      
        self.loss_WALL = tf.reduce_mean(tf.square(self.u_WALL_pred-self.u_WALL_tf)) \
                       + tf.reduce_mean(tf.square(self.v_WALL_pred-self.v_WALL_tf))
        # self.loss_CYLD = tf.reduce_mean(tf.square(self.u_CYLD_pred-self.u_CYLD_tf)) \
        #                + tf.reduce_mean(tf.square(self.v_CYLD_pred-self.v_CYLD_tf))               
        self.loss_INLET = tf.reduce_mean(tf.square(self.u_INLET_pred-self.u_INLET_tf)) \
                         + tf.reduce_mean(tf.square(self.v_INLET_pred-self.v_INLET_tf))
                         
        self.loss_internal = tf.reduce_mean(tf.square(self.u_internal_pred-self.u_internal_tf)) \
                         + tf.reduce_mean(tf.square(self.v_internal_pred-self.v_internal_tf)) \
                             + tf.reduce_mean(tf.square(self.p_internal_pred-self.p_internal_tf))
                         
        self.loss_OUTLET = tf.reduce_mean(tf.square(self.p_OUTLET_pred-self.p_OUTLET_tf))

        
        if use_internal == 0:
            
            # self.loss = self.loss_f + 2*(self.loss_CYLD + self.loss_WALL + self.loss_INLET + self.loss_OUTLET)
            self.loss = self.loss_f + 0* self.loss_internal + 2*(self.loss_WALL + self.loss_INLET + self.loss_OUTLET)
            
        else:
            
            self.loss = self.loss_f + 2*(self.loss_WALL + self.loss_INLET + self.loss_internal + self.loss_OUTLET)
        
        if debug == 0:
        
            # Optimizer for solution
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                    var_list=self.uv_weights + self.uv_biases,
                                                                    method='L-BFGS-B',
                                                                    options={'maxiter': 100000,
                                                                              'maxfun': 100000,
                                                                              'maxcor': 50,
                                                                              'maxls': 50,
                                                                              'ftol': 1*np.finfo(float).eps})
        
        elif debug == 1:
            
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                    var_list=self.uv_weights + self.uv_biases,
                                                                    method='L-BFGS-B',
                                                                    options={'maxiter': 100,
                                                                              'maxfun': 90,
                                                                              'maxcor': 50,
                                                                              'maxls': 50,
                                                                              'ftol': 1*np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
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
        
        u = psips[:,0:1]
        v = psips[:,1:2]
        p = psips[:,2:3]
        s11 = psips[:,3:4]
        s22 = psips[:,4:5]
        s12 = psips[:,5:6]
        
        
        
        return u, v, p, s11, s22, s12

    def net_f(self, x, y):

        #rho=self.rho
        #mu=self.mu
        u, v, p, s11, s22, s12 = self.net_uv(x, y)

        s11_1 = tf.gradients(s11, x)[0]
        s12_2 = tf.gradients(s12, y)[0]
        s22_2 = tf.gradients(s22, y)[0]
        s12_1 = tf.gradients(s12, x)[0]

        # Plane stress problem
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        
        #eqn 1: continuity eqn
        f_mass = (u_x + v_y)
        

        # f_u:=Sxx_x+Sxy_y
        #eqn 3
        f_u = rho*(u*u_x + v*u_y) - s11_1 - s12_2
        f_v = rho*(u*v_x + v*v_y) - s12_1 - s22_2
       
        #eqn 4
        f_s11 = -p + 2*mu*u_x - s11
        f_s22 = -p + 2*mu*v_y - s22
        f_s12 = mu*(u_y+v_x) - s12
        

        f_p = p + (s11+s22)/2

        return f_mass, f_u, f_v, f_s11, f_s22, f_s12, f_p


    def callback(self, loss):
        self.count = self.count+1
        self.loss_rec.append(loss)
        if self.count % 100 == 0:
            print('{} th iterations, Loss: {}'.format(self.count, loss))


    def train(self, iter, learning_rate):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   # self.x_CYLD_tf: self.x_CYLD, self.y_CYLD_tf: self.y_CYLD, self.u_CYLD_tf: self.u_CYLD, self.v_CYLD_tf: self.v_CYLD,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL, self.u_WALL_tf: self.u_WALL, self.v_WALL_tf: self.v_WALL,                   
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_internal_tf: self.x_internal, self.y_internal_tf: self.y_internal, self.u_internal_tf: self.u_internal, self.v_internal_tf: self.v_internal, self.p_internal_tf: self.p_internal,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET, self.p_OUTLET_tf: self.p_OUTLET,
                   self.learning_rate: learning_rate}

        #loss_CYLD = []
        loss_WALL = []
        loss_f = []
        loss_INLET = []
        loss_internal = []
        loss_OUTLET = []

        for it in range(iter):

            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 100 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e' %
                      (it, loss_value))

            #loss_CYLD.append(self.sess.run(self.loss_CYLD, tf_dict))
            loss_WALL.append(self.sess.run(self.loss_WALL, tf_dict))            
            loss_f.append(self.sess.run(self.loss_f, tf_dict))
            self.loss_rec.append(self.sess.run(self.loss, tf_dict))
            loss_INLET.append(self.sess.run(self.loss_INLET, tf_dict))
            loss_internal.append(self.sess.run(self.loss_internal, tf_dict))
            loss_OUTLET.append(self.sess.run(self.loss_OUTLET, tf_dict))

        # return loss_CYLD, loss_WALL, loss_INLET, loss_OUTLET, loss_f, self.loss
        return loss_WALL, loss_INLET, loss_internal, loss_OUTLET, loss_f, self.loss

    def train_bfgs(self):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   #self.x_CYLD_tf: self.x_CYLD, self.y_CYLD_tf: self.y_CYLD, self.u_CYLD_tf: self.u_CYLD, self.v_CYLD_tf: self.v_CYLD,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL, self.u_WALL_tf: self.u_WALL, self.v_WALL_tf: self.v_WALL,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_internal_tf: self.x_internal, self.y_internal_tf: self.y_internal, self.u_internal_tf: self.u_internal, self.v_internal_tf: self.v_internal, self.p_internal_tf: self.p_internal,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET, self.p_OUTLET_tf: self.p_OUTLET}

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
                   #self.x_CYLD_tf: self.x_CYLD, self.y_CYLD_tf: self.y_CYLD, self.u_CYLD_tf: self.u_CYLD, self.v_CYLD_tf: self.v_CYLD,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL, self.u_WALL_tf: self.u_WALL, self.v_WALL_tf: self.v_WALL,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_internal_tf: self.x_internal, self.y_internal_tf: self.y_internal, self.u_internal_tf: self.u_internal, self.v_internal_tf: self.v_internal, self.p_internal_tf: self.p_internal,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET, self.p_OUTLET_tf: self.p_OUTLET}

        loss_f = self.sess.run(self.loss_f, tf_dict)
        #loss_CYLD = self.sess.run(self.loss_CYLD, tf_dict)
        loss_WALL = self.sess.run(self.loss_WALL, tf_dict)
        loss_INLET = self.sess.run(self.loss_INLET, tf_dict)
        loss_internal = self.sess.run(self.loss_internal, tf_dict)
        loss = self.sess.run(self.loss, tf_dict)
        loss_OUTLET = self.sess.run(self.loss_OUTLET, tf_dict)

        # return loss_CYLD, loss_WALL, loss_INLET, loss_OUTLET, loss_f, loss
        return loss_WALL, loss_INLET, loss_internal, loss_OUTLET, loss_f, loss

def DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    '''
    delete points within cylinder
    '''
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst>r,:]


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

    #fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7, 4))
    #fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig, ax = plt.subplots(nrows=3, ncols=2)
    

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

    cf = ax[1, 0].scatter(x_MIXED, y_MIXED, c=v_MIXED, vmin=-0.5, vmax=0.5, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=int(s))
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

    cf = ax[2, 0].scatter(x_MIXED, y_MIXED, c=p_MIXED, vmin=-0.5, vmax=0.5, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=int(s))
    # cf = ax[2,0].tricontourf(x_MIXED[:,0], y_MIXED[:,0], p_MIXED[:,0], 
    #                          levels=np.linspace(0,2.,11), alpha=alpha, cmap='viridis')
    # cf = ax[2,0].contourf(x_MIXED, y_MIXED, p_MIXED, 
    #                          levels=np.linspace(0,2.,11), alpha=alpha, cmap='viridis')
    # cf = ax[2,0].imshow(p_MIXED, vmin=0, vmax=2, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ3 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[2,0].add_patch(circ3)
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
    ax[2, 0].set_title('Pressure (Pa)')
    fig.colorbar(cf, ax=ax[2, 0], fraction=0.046, pad=0.04)

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

    cf = ax[2, 1].scatter(x_ref, y_ref, c=p_ref, vmin=-0.5, vmax=0.5, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=s)
    #cf = ax[2, 1].tricontourf(x_ref[:,0], y_ref[:,0], p_ref[:,0], levels=np.linspace(0,2.0,11), alpha=alpha, cmap='viridis')
    # cf = ax[2, 1].contourf(x_ref, y_ref, c=p_ref, vmin=0., vmax=2., alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=s)
    # cf = ax[2,1].imshow(p_ref, vmin=0, vmax=2, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ6 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[2,1].add_patch(circ6)
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
    ax[2, 1].set_title('Pressure (Pa)')
    fig.colorbar(cf, ax=ax[2, 1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('./output/'+filename + '.png',dpi=360)
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
    

    [x_ref, y_ref, u_ref, v_ref, p_ref] = field_ref
    [x_MIXED, y_MIXED, u_MIXED, v_MIXED, p_MIXED] = field_MIXED
    [x_diff, y_diff, u_diff, v_diff, p_diff] = field_diff

    #fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(7, 4))
    fig, ax = plt.subplots(nrows=3, ncols=3)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Plot MIXED result
    cf = ax[0, 0].scatter(x_MIXED, y_MIXED, c=u_MIXED, vmin=0, vmax=1.5, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=int(s))
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

    cf = ax[2, 0].scatter(x_MIXED, y_MIXED, c=p_MIXED, vmin=-0.5, vmax=0.5, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=int(s))
    # cf = ax[2,0].imshow(p_MIXED, vmin=0, vmax=2, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ3 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[2,0].add_patch(circ3)
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
    ax[2, 0].set_title('Pressure (Pa)')
    fig.colorbar(cf, ax=ax[2, 0], fraction=0.046, pad=0.04)

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

    cf = ax[2, 1].scatter(x_ref, y_ref, c=p_ref, vmin=-0.5, vmax=0.5, alpha=alpha, edgecolors='none', cmap='viridis', marker=marker, s=s)
    # cf = ax[2,1].imshow(p_ref, vmin=0, vmax=2, extent=[xmin,xmax,ymin,ymax], 
    #                     origin='lower', cmap='viridis')
    #circ6 = plt.Circle((0,0), radius=0.5, color='w')
    #ax[2,1].add_patch(circ6)
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
    ax[2, 1].set_title('Pressure (Pa)')
    fig.colorbar(cf, ax=ax[2, 1], fraction=0.046, pad=0.04)
    
    # Plot diff result
    if diff == 'absolute':
                
        ax[0, 2].set_title('u abs diff')
        cf = ax[0, 2].scatter(x_diff, y_diff, c=u_diff, vmin=-1.5e-1, vmax=1.5e-1, alpha=alpha, 
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
        cf = ax[1, 2].scatter(x_diff, y_diff, c=v_diff, vmin=-0.5e-1, vmax=0.5e-1, alpha=alpha, 
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
                
        ax[2, 2].set_title('p abs diff')
        cf = ax[2, 2].scatter(x_diff, y_diff, c=p_diff, vmin=-0.05, vmax=0.05, alpha=alpha, 
                              edgecolors='none', cmap='viridis', marker=marker, s=int(s))
        # cf = ax[2,2].imshow(p_diff, vmin=-1e-2, vmax=1e-2, extent=[xmin,xmax,ymin,ymax], 
        #                 origin='lower', cmap='viridis')
        #circ9 = plt.Circle((0,0), radius=0.5, color='w')
        #ax[2,2].add_patch(circ9)
        
    elif diff == 'percentage':
        
        ax[2, 2].set_title('p diff (%)')
        cf = ax[2, 2].scatter(x_diff, y_diff, c=p_diff, vmin=-10, vmax=10, alpha=alpha, 
                              edgecolors='none', cmap='viridis', marker=marker, s=int(s))
        
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
    
    plt.tight_layout()
    
    #plt.savefig('./uvp_mixed_uvp.png', dpi=360)    
    plt.savefig('./output/'+filename+'.png',dpi=360)
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
    
    pi = 3.141592653
    delta = 2*pi/points
    
    theta = np.zeros((points)); p_pred = np.zeros((points))
    x_loc = np.zeros((points)); y_loc = np.zeros((points))
    
    fx = 0.; fy = 0.
    
    for i in range(points):
        
        theta[i] = i*delta
        x_loc[i] = 0.501*np.cos(theta[i]); y_loc[i] = 0.501*np.sin(theta[i])
    
    x_loc = x_loc.reshape(-1, 1); y_loc = y_loc.reshape(-1, 1)
    _, _, p_pred = model.predict(x_loc, y_loc)
    
    for i in range(points):
        
        dx = delta*np.cos(theta[i]); dy = delta*np.sin(theta[i])
        fx = fx - p_pred[i]*dx; fy = fy - p_pred[i]*dy
        
    fig = plt.figure()
    plt.plot(theta, p_pred)
    plt.savefig('./output/pressure_vs_theta.png',dpi=360)
    plt.show()
        
    return fx, fy

def pressure_forces_ref(points):
    #calculate pressure forces of circle, also plot pressure vs theta
    #use ref data
    
    dataset = np.loadtxt('uvp_openfoam1.csv',delimiter=',',usecols=range(5),skiprows=1)
    
    # [x_ref,y_ref,u_ref,v_ref,p_ref] = dataset #not working
    
    x_ref = dataset[:,0]; y_ref = dataset[:,1]
    u_ref = dataset[:,2]; v_ref = dataset[:,3]
    p_ref = dataset[:,4]
    
    #fn_u_ref = scipy.interpolate.interp2d(x_ref,y_ref,u_ref,kind='cubic')
    #fn_u_ref = scipy.interpolate.interp2d(x_ref,y_ref,u_ref,kind='linear')
    #fn_v_ref = scipy.interpolate.interp2d(x_ref,y_ref,v_ref,kind='cubic')
    #fn_p_ref = scipy.interpolate.interp2d(x_ref,y_ref,p_ref,kind='cubic')
    
    
    
    pi = 3.141592653
    delta = 2*pi/points
    
    theta = np.zeros((points)); p_pred = np.zeros((points))
    x_loc = np.zeros((points)); y_loc = np.zeros((points))
    
    fx = 0.; fy = 0.
    
    for i in range(points):
        
        theta[i] = i*delta
        x_loc[i] = 0.5*np.cos(theta[i]); y_loc[i] = 0.5*np.sin(theta[i])
        #p_pred[i] = fn_p_ref(x_loc[i],y_loc[i])
        
    p_pred = griddata((x_ref, y_ref), p_ref, (x_loc,y_loc), method='linear')
        
    for i in range(points):
       
       dx = delta*np.cos(theta[i]); dy = delta*np.sin(theta[i])
       fx = fx - p_pred[i]*dx; fy = fy - p_pred[i]*dy
        
    fig = plt.figure()
    plt.plot(theta, p_pred)
    plt.savefig('./output/pressure_vs_theta_ref.png',dpi=360)
    plt.show()
        
    return fx, fy

def viscous_forces(points,delta_dist):
    #calculate pressure forces of circle, also plot pressure vs theta
    
    pi = 3.141592653
    delta = 2*pi/points
    test_points_x = np.zeros((5,1)); test_points_y = np.zeros((5,1))
    
    theta = np.zeros((points)); p_pred = np.zeros((points))
    x_loc = np.zeros((points)); y_loc = np.zeros((points))
    
    fx = 0.; fy = 0.
    
    for i in range(points):
        
        theta[i] = i*delta
        x_loc[i] = 0.501*np.cos(theta[i]) + 1e-10 #prevent dist_x/y = 0
        y_loc[i] = 0.501*np.sin(theta[i]) + 1e-10
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
        
        fx = fx + (mu/rho)*(2*du_dx*dx + du_dy*dy + dv_dx*dy )
        fy = fy + (mu/rho)*(2*dv_dy*dy + du_dy*dx + dv_dx*dx )   
   
        
    return fx, fy

def viscous_forces_ref(points,delta_dist):
    #calculate pressure forces of circle, also plot pressure vs theta
    
    #use ref data
    
    dataset = np.loadtxt('uvp_openfoam1.csv',delimiter=',',usecols=range(5),skiprows=1)
    
    # [x_ref,y_ref,u_ref,v_ref,p_ref] = dataset #not working
    
    x_ref = dataset[:,0]; y_ref = dataset[:,1]
    u_ref = dataset[:,2]; v_ref = dataset[:,3]
    p_ref = dataset[:,4]
    
    pi = 3.141592653
    delta = 2*pi/points
    test_points_xy = np.zeros((5,2))
    
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
        
        test_points_xy[0,0] = x_loc[i]; test_points_xy[0,1] = y_loc[i]
        test_points_xy[1,0] = x1; test_points_xy[1,1] = y_loc[i]
        test_points_xy[2,0] = x2; test_points_xy[2,1] = y_loc[i]
        test_points_xy[3,0] = x_loc[i]; test_points_xy[3,1] = y1
        test_points_xy[4,0] = x_loc[i]; test_points_xy[4,1] = y2
        
    u_pred = griddata((x_ref, y_ref), u_ref, test_points_xy, method='linear')
    v_pred = griddata((x_ref, y_ref), v_ref, test_points_xy, method='linear')
    
    for i in range(points):
        
        u0 = u_pred[0]; v0 = v_pred[0]
        ui1 = u_pred[1]; vi1 = v_pred[1]
        ui2 = u_pred[2]; vi2 = v_pred[2]
        uj1 = u_pred[3]; vj1 = v_pred[3]
        uj2 = u_pred[4]; vj2 = v_pred[4]
        
        du_dx = -np.sign(x_loc[i])*(3*u0 - 4*ui1 + ui2)/(2*delta_dist)
        dv_dx = -np.sign(x_loc[i])*(3*v0 - 4*vi1 + vi2)/(2*delta_dist)
        du_dy = -np.sign(y_loc[i])*(3*u0 - 4*uj1 + uj2)/(2*delta_dist)
        dv_dy = -np.sign(y_loc[i])*(3*v0 - 4*vj1 + vj2)/(2*delta_dist)
        
        dx = delta*np.cos(theta[i]); dy = delta*np.sin(theta[i])
        
        fx = fx + (mu/rho)*(2*du_dx*dx + du_dy*dy + dv_dx*dy )
        fy = fy + (mu/rho)*(2*dv_dy*dy + du_dy*dx + dv_dx*dx )   
   
    return fx, fy

if __name__ == "__main__":
    
    shutil.rmtree('./output', ignore_errors=True)
    os.makedirs('./output')
    
    xmin = -5.0
    xmax = 9.33
    ymin = -5.0
    ymax = 5.0
    
    # xmin = -10.0
    # xmax = 20.0
    # ymin = -10.0
    # ymax = 10.0
    
    radius = 0.5
    cyld_x_center = 0.0
    cyld_y_center = 0.0
    front_refine_dist = 2.0
    back_refine_dist = 4.0
    top_bottom_refine_dist = 0.2

    # Domain bounds
    # lb = np.array([-5.0, -5.0])   #lower left x,y coord of boundary
    # ub = np.array([9.33, 5.0])  #upper right x,y coord of boundary
    
    lb = np.array([xmin, ymin])   #lower left x,y coord of boundary
    ub = np.array([xmax, ymax])  #upper right x,y coord of boundary
    
    u_ref = 1.0

    # Network configuration
    uv_layers = [2] + 8*[40] + [6]  # 2 i/p, 8 layers x 40 neurons each, 6 o/p

    # WALL = [x, y], u=v=0
    WALL_TOP = [xmin, ymax] + [xmax - xmin, 0.0] * lhs(2, 441) #lhs = Latin-Hypercube from pyDOE
    #lhs(2,441) = generate 2 variables, 441 pts each
    x_WALL_TOP = WALL_TOP[:,0:1]
    y_WALL_TOP = WALL_TOP[:,1:2]
    u_WALL_TOP = np.zeros_like(x_WALL_TOP)
    v_WALL_TOP = np.zeros_like(x_WALL_TOP)
    u_WALL_TOP[:] = u_ref
    v_WALL_TOP[:] = 0.
    #WALL_TOP = np.concatenate((WALL_TOP, u_WALL_TOP, v_WALL_TOP), 1)
    
    #[0.0, 4.1] + [11.0, 0.0]* = starting + delta * to get values within range
    WALL_BOTTOM = [xmin, ymin] + [xmax - xmin, 0.0] * lhs(2, 441)
    x_WALL_BOTTOM = WALL_BOTTOM[:,0:1]
    y_WALL_BOTTOM = WALL_BOTTOM[:,1:2]
    u_WALL_BOTTOM = np.zeros_like(x_WALL_BOTTOM)
    v_WALL_BOTTOM = np.zeros_like(x_WALL_BOTTOM)
    u_WALL_BOTTOM[:] = u_ref
    v_WALL_BOTTOM[:] = 0.
    #WALL_BOTTOM = np.concatenate((WALL_BOTTOM, u_WALL_BOTTOM, v_WALL_BOTTOM), 1)

    # INLET = [x, y, u, v]
    
    INLET = [xmin, ymin] + [0.0, ymax - ymin] * lhs(2, 201)
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
    internal, _ = train_test_split(dataset, train_size=training_size, random_state=1234) #5% of data used for training

    # INLET = [x, y], p=0 #or OUTLET?
    OUTLET = [xmax, ymin] + [0.0, ymax - ymin] * lhs(2, 201)
    x_OUTLET = OUTLET[:,0:1]
    y_OUTLET = OUTLET[:,1:2]
    p_OUTLET = np.zeros_like(y_OUTLET)
    p_OUTLET[:] = 0.
    OUTLET = np.concatenate((OUTLET, p_OUTLET), 1)

    # Cylinder surface
    r = radius
    theta = [0.0] + [2*np.pi] * lhs(1, 251)
    x_CYLD = np.multiply(r, np.cos(theta))+0.0
    y_CYLD = np.multiply(r, np.sin(theta))+0.0
    CYLD = np.concatenate((x_CYLD, y_CYLD), 1)  #x,y coord of circle
    u_CYLD = np.zeros_like(x_CYLD)
    v_CYLD = np.zeros_like(x_CYLD)
    u_CYLD[:] = 0.
    v_CYLD[:] = 0.

    WALL_x = np.concatenate((x_CYLD, x_WALL_TOP, x_WALL_BOTTOM), 0)
    WALL_y = np.concatenate((y_CYLD, y_WALL_TOP, y_WALL_BOTTOM), 0)
    WALL_u = np.concatenate((u_CYLD, u_WALL_TOP, u_WALL_BOTTOM), 0)
    WALL_v = np.concatenate((v_CYLD, v_WALL_TOP, v_WALL_BOTTOM), 0)
    WALL = np.concatenate((WALL_x, WALL_y, WALL_u, WALL_v), 1)

    # Collocation point for equation residual
    XY_c = lb + (ub - lb) * lhs(2, 40000)   #x,y coord for entire domain
    XY_c_refine = [-1.0, -1.0] + [2.0, 2.0] * lhs(2, 20000)   #add more pts between xy 0.1 -> 0.3
    XY_c = np.concatenate((XY_c, XY_c_refine), 0)
    XY_c = DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.5)   #remove pts inside circle

    #XY_c = np.concatenate((XY_c, WALL[:,0:2], OUTLET[:,0:2], INLET[:,0:2], internal[:,0:2]), 0)

    print(XY_c.shape)

    # Visualize the collocation points
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(XY_c[:,0:1], XY_c[:,1:2], marker='o', alpha=0.1 ,color='blue')
    # plt.scatter(CYLD[:,0:1], CYLD[:,1:2], marker='o', alpha=0.2 , color='yellow')
    plt.scatter(WALL[:,0:1], WALL[:,1:2], marker='o', alpha=0.2 , color='green')
    plt.scatter(OUTLET[:, 0:1], OUTLET[:, 1:2], marker='o', alpha=0.2, color='orange')
    plt.scatter(INLET[:, 0:1], INLET[:, 1:2], marker='o', alpha=0.2, color='red')
    plt.savefig('./output/domain.png',dpi=360)
    plt.show()
    plt.scatter(internal[:, 0:1], internal[:, 1:2], marker='o', alpha=0.2, color='yellow')
    plt.savefig('./output/domain_internal.png',dpi=360)
    plt.show()

    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # Train from scratch
        # model = PINN_laminar_flow(XY_c, INLET, OUTLET, WALL, uv_layers, lb, ub)

        # Load trained neural network
        model = PINN_laminar_flow(XY_c, INLET, internal, OUTLET, WALL, uv_layers, lb, ub, ExistModel = use_old_model, uvDir = 'uvNN.pickle')

        start_time = time.time()
        
        if debug == 0:
            
            if use_old_model == 1:
            
                loss_WALL, loss_INLET, loss_internal, loss_OUTLET, loss_f, loss = model.train(iter=4000, learning_rate=5e-4)
                
            else:
                
                loss_WALL, loss_INLET, loss_internal, loss_OUTLET, loss_f, loss = model.train(iter=10000, learning_rate=5e-4)
        
        elif debug == 1:
            
            loss_WALL, loss_INLET, loss_internal, loss_OUTLET, loss_f, loss = model.train(iter=60, learning_rate=5e-4)
        
        model.train_bfgs()
        print("--- %s seconds ---" % (time.time() - start_time))

        # Save neural network
        model.save_NN('uvNN.pickle')

        # Save loss history
        with open('loss_history.pickle', 'wb') as f:
            pickle.dump(model.loss_rec, f)
            
        if data_ref == 'fluent':

            # Load fluent result
            [x_ref, y_ref, u_ref, v_ref, p_ref] = preprocess(dir='../FluentReferenceMu002/FluentSol.mat')
            field_ref = [x_ref, y_ref, u_ref, v_ref, p_ref]
            
            # field_ref_csv = np.asarray(field_ref)
            # field_ref_csv = field_ref_csv[:,:,0]
            # field_ref_csv = np.transpose(field_ref_csv)
            
            # df = pd.DataFrame(field_ref_csv)
            # df.to_csv('uvp_ref.csv',index=False)
            # Cov = pd.read_csv("uvp_ref.csv", names=["x", "y", "u", "v","p"])
            # Cov.to_csv('uvp_ref.csv',index=False)
            # lines = open('uvp_ref.csv', 'r').readlines()
            # del lines[1]
            # open('uvp_ref.csv', 'w').writelines(lines)
            
        elif data_ref == 'openfoam':
            
            field_ref = np.loadtxt('uvp_openfoam1.csv',delimiter=',',usecols=range(5),skiprows=1)
            x_ref = field_ref[:,0:1]
            y_ref = field_ref[:,1:2]
            u_ref = field_ref[:,2:3]
            v_ref = field_ref[:,3:4]
            p_ref = field_ref[:,4:5]
            
            field_ref = [x_ref, y_ref, u_ref, v_ref, p_ref]
        
        # Get mixed-form PINN prediction
        x_PINN = np.linspace(xmin, xmax, 251)
        y_PINN = np.linspace(ymin, ymax, 101)
        x_PINN, y_PINN = np.meshgrid(x_PINN, y_PINN)
        x_PINN = x_PINN.flatten()[:, None]
        y_PINN = y_PINN.flatten()[:, None]
        dst = ((x_PINN-0.0)**2+(y_PINN-0.0)**2)**0.5
        x_PINN = x_PINN[dst >= 0.5]
        y_PINN = y_PINN[dst >= 0.5]
        x_PINN = x_PINN.flatten()[:, None]
        y_PINN = y_PINN.flatten()[:, None]
        u_PINN, v_PINN, p_PINN = model.predict(x_PINN, y_PINN)
        field_MIXED = [x_PINN, y_PINN, u_PINN, v_PINN, p_PINN]
        
        field_MIXED_csv = np.asarray(field_MIXED)
        field_MIXED_csv = field_MIXED_csv[:,:,0]
        field_MIXED_csv = np.transpose(field_MIXED_csv)
        df = pd.DataFrame(field_MIXED_csv)
        df.to_csv('./output/uvp_mixed.csv',index=False)
        Cov = pd.read_csv("./output/uvp_mixed.csv", names=["x", "y", "u", "v","p"])
        Cov.to_csv('./output/uvp_mixed.csv',index=False)
        lines = open('./output/uvp_mixed.csv', 'r').readlines()
        del lines[1]
        open('./output/uvp_mixed.csv', 'w').writelines(lines)
        
        #interpolate OF data to match current x,y
        u_ref_int = scipy.interpolate.griddata((x_ref.flatten(),y_ref.flatten()),u_ref.flatten() , (x_PINN,y_PINN),method='cubic')
        v_ref_int = scipy.interpolate.griddata((x_ref.flatten(),y_ref.flatten()),v_ref.flatten() , (x_PINN,y_PINN),method='cubic')
        p_ref_int = scipy.interpolate.griddata((x_ref.flatten(),y_ref.flatten()),p_ref.flatten() , (x_PINN,y_PINN),method='cubic')
        field_ref_int = [x_PINN, y_PINN, u_ref_int, v_ref_int, p_ref_int]

        # Plot the comparison of u, v, p
        postProcess('uvp_PINN_vs_OF', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, field_ref=field_ref_int, field_MIXED=field_MIXED, s=5, alpha=1)
        
        if diff == 'absolute':
            
            u_diff = u_PINN - u_ref_int
            v_diff = v_PINN - v_ref_int
            p_diff = p_PINN - p_ref_int
            
        elif diff == 'percentage':
            
            u_ref_int = u_ref_int + 1e-10 #prevent 0 divide
            v_ref_int = v_ref_int + 1e-10 #prevent 0 divide
            p_ref_int = p_ref_int + 1e-10 #prevent 0 divide
        
            u_diff = 100.*(u_PINN - u_ref_int)/u_ref_int
            v_diff = 100.*(v_PINN - v_ref_int)/v_ref_int
            p_diff = 100.*(p_PINN - p_ref_int)/p_ref_int
            
            # u_diff_xy = 100.*(u_pred_xy - u_ref_xy)/u_ref_xy
            # v_diff_xy = 100.*(v_pred_xy - v_ref_xy)/v_ref_xy
            # p_diff_xy = 100.*(p_pred_xy - p_ref_xy)/p_ref_xy
        
    field_diff = [x_PINN, y_PINN, u_diff, v_diff, p_diff]
    
    postProcess3('uvp_PINN_vs_OF_diff', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, field_ref=field_ref_int,field_MIXED=field_MIXED, field_diff=field_diff, s=5, alpha=1)
    
    fx_p,fy_p = pressure_forces(surface_pts)
    fx_v,fy_v = viscous_forces(surface_pts,delta_dist)
    
    fx = fx_p + fx_v
    fy = fy_p + fy_v
    print(fx_p,fx_v,fx)
    print(fy_p,fy_v,fy)
    
    fx_p_ref,fy_p_ref = pressure_forces_ref(surface_pts)
    
    fx_v_ref = 0; fy_v_ref = 0
    fx_v_ref,fy_v_ref = viscous_forces_ref(surface_pts,delta_dist)
    
    fx_ref = fx_p_ref + fx_v_ref
    fy_ref = fy_p_ref + fy_v_ref
    print(fx_p_ref,fx_v_ref,fx_ref)
    print(fy_p_ref,fy_v_ref,fy_ref)