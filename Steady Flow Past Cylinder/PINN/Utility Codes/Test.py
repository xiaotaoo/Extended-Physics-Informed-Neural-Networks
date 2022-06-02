import numpy as np
from pyDOE import lhs  

xlb=-5
xub=9.33
ylb=-5
yub=5
u_ref=1

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

print (WALL)