# The following license applies to this particular file.
########################################################################
# This is free and unencumbered software released into the public domain.

# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.

# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# For more information, please refer to <http://unlicense.org/>
########################################################################

import numpy as np
import math

kTol = 1e-8

def so3_to_matrix(so3):
    wx = so3[0,0]
    wy = so3[1,0]
    wz = so3[2,0]
    so3_matrix = np.array([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0]
    ])
    return so3_matrix

def so3_to_vector(so3):
    return np.array([[so3[2,1], so3[0,2], so3[1,0]]]).T

def se3_to_matrix(se3):
    result = np.empty((4,4))
    result[:3,:3] = so3_to_matrix(se3[:3,:])
    result[:3,3] = se3[3:,0]
    result[3,:] = 0
    return result

def se3_exp(se3):
    assert se3.shape == (6,1)
    # See page 10 https://ethaneade.com/lie.pdf
    # we reverse u and omega in the ordering of se3
    result = np.eye(4)
    omega_vec = se3[:3,:]
    theta_squared = np.dot(omega_vec.T, omega_vec)
    omega = so3_to_matrix(omega_vec)
    if (theta_squared < kTol): 
        # second order taylor expansion
        A = -theta_squared/6.0 + 1.0
        B = -theta_squared/24.0 + 0.5
        C = -theta_squared/120.0 + (1.0/6.0)
    else:
        theta = math.sqrt(theta_squared)
        stheta = math.sin(theta)
        ctheta = math.cos(theta)
        A = stheta / theta
        B = (-ctheta + 1.0) / theta_squared
        C = (-A + 1.0) / theta_squared

    omega_squared = omega @ omega
    v = se3[3:,:]

    result[:3,:3] += omega*A + omega_squared*B
    result[:3,3:4] = ((omega*B + omega_squared*C) @ v) + v
    return result

def se2_exp(se2):
    theta = se2[0,0]
    x = se2[1,0]
    y = se2[2,0]

    if (abs(theta) < kTol):
        return np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ])

    r_p = (x + 1j*y)/theta
    r = r_p * 1j
    rotation = np.cos(theta) + 1j*np.sin(theta)
    ep = r + (-r * rotation)
    ed = rotation

    return np.array([
        [ed.real, -ed.imag, ep.real],
        [ed.imag,  ed.real, ep.imag],
        [0,        0,       1      ]
    ])

    return out

def SE3_adj(SE3):
    result = np.empty((6,6))
    pm = so3_to_matrix(SE3[:3,3:4])
    result[:3,:3] = SE3[:3,:3]
    result[3:,3:] = SE3[:3,:3]
    result[3:,:3] = pm @ SE3[:3,:3]
    result[:3,3:].fill(0)
    return result

def SE3_inv(SE3):
    result = np.empty((4,4))
    result[:3,:3] = SE3[:3,:3].T
    result[:3,3:4] = -SE3[:3,:3].T @ SE3[:3,3:4]
    result[3,:] = [0, 0, 0, 1]
    return result

def almost_equals(a, b):
    return abs(a - b) < kTol

def SO3_log_decomposed(SO3):
    """
    Returns so3_hat, the normalized version of the so3
    vector corresponding to the input SO3, and theta, the
    magnitude.
    """
    trace = np.trace(SO3)
    is_identity = almost_equals(trace, 3)

    # Edge case: identity
    if is_identity:
        theta = 0
        omega_hat = np.array([[1, 0, 0]], dtype=float).T
        return omega_hat, theta

    # Edge case: rotation of k*PI
    if almost_equals(trace, -1):
        # print("Edge case")
        theta = np.pi
        r33 = SO3[2,2]
        r22 = SO3[1,1]
        r11 = SO3[0,0]

        if not almost_equals(1.0 + r33, 0):
            # print("Case 1")
            omega_hat = np.array([[SO3[0,2], SO3[1,2], 1+SO3[2,2]]]).T
            omega_hat /= (2 * (1 + r33))**0.5
            return omega_hat, theta

        if not almost_equals(1.0 + r22, 0):
            # print("Case 2")
            omega_hat = np.array([[SO3[0,1], 1+SO3[1,1], SO3[2,1]]]).T
            omega_hat /= (2 * (1 + r22))**0.5
            return omega_hat, theta

        # print("Case 3")
        assert almost_equals(1.0 + r33, 0)
        omega_hat = np.array([[1+SO3[0,0], SO3[1,0], SO3[2,0]]]).T
        omega_hat /= (2 * (1 + r11))**0.5
        return omega_hat, theta

    # normal case
    # htmo means Half of Trace Minus One
    htmo = 0.5 * (trace - 1)
    theta = np.arccos(htmo)
    sin_acos_htmo = (1.0 - htmo*htmo)**0.5
    omega_mat = 0.5/sin_acos_htmo * (SO3 - SO3.T)
    omega_hat = so3_to_vector(omega_mat)

    assert np.all(np.isfinite(omega_hat))
    
    return omega_hat, theta

def SO3_log(SO3):
    so3_hat, theta = SO3_log_decomposed(SO3)
    return so3_hat * theta

def barfoot_Q(se3):
    rho = so3_to_matrix(se3[3:,:])
    theta = so3_to_matrix(se3[:3,:])
    rho_theta = (rho@theta)
    rho_theta2 = (rho_theta@theta)
    theta_rho_theta = (theta@rho_theta)
    theta_rho = (theta@rho)
    theta2_rho = (theta@theta_rho)

    angle = np.linalg.norm(se3[:3,:])
    if (angle < kTol):
        angle2 = angle*angle
        angle4 = angle2*angle2
        c1 = 1.0/6 - angle2/120 + angle4/5040
        c2 = 1.0/24 - angle2/720 + angle4/40320
        c3 = 1.0/120 - angle2/2520 + angle4/120960
    else:
        angle2 = angle*angle
        angle3 = angle2*angle
        angle4 = angle2*angle2
        angle5 = angle3*angle2
        sn = np.sin(angle)
        cs = np.cos(angle)
        c1 = (angle - sn)/angle3
        c2 = -(1.0 - angle2/2 - cs)/angle4
        c3 = -0.5*(
            (1.0 - angle2/2 - cs)/angle4 -
            3.0*(angle - sn - angle3/6)/angle5)

    line1 = 0.5*rho + c1*(theta_rho + rho_theta + theta_rho_theta)
    line2 = c2*(theta2_rho + rho_theta2 - 3.0*theta_rho_theta)
    line3 = c3*(theta_rho_theta@theta + theta@theta_rho_theta)
    return line1 + line2 + line3

def SO3_left_jacobian(so3):
    theta = np.linalg.norm(so3)
    theta2 = theta*theta
    theta3 = theta2*theta
    theta4 = theta2*theta2

    theta_mat = so3_to_matrix(so3)    
    if (theta < kTol):
        coeff1 = 0.5 - theta2/24 + theta4/720
        coeff2 = 1.0/6 - theta2/120 + theta4/5040
    else:
        cs = np.cos(theta)
        sn = np.sin(theta)
        coeff1 = (1.0 - cs)/theta2
        coeff2 = (theta - sn)/theta3

    result = np.eye(3, dtype=float) + coeff1*theta_mat + coeff2*theta_mat@theta_mat
    return result

def se3_left_jacobian(se3):
    SO3_lj = SO3_left_jacobian(se3[:3,:])
    Q = barfoot_Q(se3)
    result = np.empty((6,6), dtype=float)
    result[:3,:3] = SO3_lj
    result[3:,3:] = SO3_lj
    result[:3,3:] = 0
    result[3:,:3] = Q
    return result

def x_cotx(x):
    c2 = -1.0 / 3
    c4 = -1.0 / 45
    c6 = -2.0 / 945
    c8 = -1.0 / 4725
    c10 = -2.0 / 93555
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    return 1.0 + c2 * x2 + c4 * x4 + c6 * x6 + c8 * x8 + c10 * x10

def se3_to_vector(se3_mat):
    se3 = np.empty((6,1), dtype=float)
    se3[:3,:] = so3_to_vector(se3_mat[:3,:3])
    se3[3:,:] = se3_mat[:3,3:]
    return se3

def SE3_log(SE3):
    omega_hat, theta = SO3_log_decomposed(SE3[:3,:3])
    omega = so3_to_matrix(omega_hat)

    p = SE3[:3,3:]
    omega_p = omega @ p
    v_theta = p - 0.5*theta*omega_p + (1.0 - x_cotx(theta/2)) * omega @ omega_p
    result = np.empty((6,1), float)
    result[:3,:] = omega_hat * theta
    result[3:,:] = v_theta

    assert np.all(np.isfinite(result))
    return result
    
def fix_SE3(SE3):
    Rx = SE3[:3, 0]
    Ry = SE3[:3, 1]
    Rz = SE3[:3, 2]

    Rz = np.cross(Rx, Ry)
    Ry = np.cross(Rz, Rx)

    Rx /= np.linalg.norm(Rx)
    Ry /= np.linalg.norm(Ry)
    Rz /= np.linalg.norm(Rz)

    SE3[:3, 0] = Rx
    SE3[:3, 1] = Ry
    SE3[:3, 2] = Rz

    SE3[3, :] = [0,0,0,1]

def SE2_inv(SE2):
    result = np.empty((3,3))
    result[:2,:2] = SE2[:2,:2].T
    result[:2,2:3] = -SE2[:2,:2].T @ SE2[:2,2:3]
    result[2,:] = [0, 0, 1]
    return result

def fix_SE2(SE2):
    SE2[0,1] = -SE2[1,0]
    SE2[1,1] = SE2[0,0]
    Rxy = SE2[:2,0]
    normRxy = np.linalg.norm(Rxy)
    SE2[:2,:2] /= normRxy

def check_SE2(SE2):
    R = SE2[:2,:2]
    
    Rxy = R[:2,0]
    normRxy = np.linalg.norm(Rxy)
    if abs(normRxy - 1) > kTol:
        print("Bad SE2\n", SE2)
        print("Rxy", Rxy.T)
        print("normRxy ", normRxy)
        raise RuntimeError("Bad SE2", SE2)
    
    if np.max(np.abs(R @ R.T - np.eye(2))) > kTol:
        print("Bad SE2\n", SE2)
        print("R@R.T\n", R@R.T)
        print("normRxy\n", normRxy)
        raise RuntimeError("Bad SE2", SE2)

    if abs(SE2[2,2] - 1) > kTol:
        print("Bad SE2\n", SE2)
        print("Lower corner not 1")
        raise RuntimeError("Bad SE2", SE2)

    if np.max(np.abs(SE2[2,:2])) > kTol:
        print("Bad SE2\n", SE2)
        print("Lower zeros")
        raise RuntimeError("Bad SE2", SE2)

def dSE3_adj(se3):
    # returns lim_{t \rightarrow 0} Adj[exp(se3 * t)]/t
    w_x, w_y, w_z, p_x, p_y, p_z = se3.flatten()
    S = np.array([
        [0, -w_z, w_y, 0, 0, 0],
        [w_z, 0, -w_x, 0, 0, 0],
        [-w_y, w_x, 0, 0, 0, 0],
        [0, -p_z, p_y, 0, -w_z, w_y],
        [p_z, 0, -p_x, w_z, 0, -w_x],
        [-p_y, p_x, 0, -w_y, w_x, 0]])

    return S

# SE3 flat index mappings
R_00 = 0
R_10 = 1
R_20 = 2

R_01 = 3
R_11 = 4
R_21 = 5

R_02 = 6
R_12 = 7
R_22 = 8

P_0 = 9
P_1 = 10
P_2 = 11

def SE3_times(SE3):
    """
    If the input is a pose T, and W is a 6d vector
    (an element in se3), this returns the
    linear operator L such that LW = flatten(T se3_to_matrix(W))

    The flatten() operator takes a pose T and lists out
    the entries in column major, ignoring the bottom row
    """
    
    R = SE3[:3, :3]
    return np.array([
        [       0, -R[0, 2],  R[0, 1],       0,       0,       0],
        [       0, -R[1, 2],  R[1, 1],       0,       0,       0],
        [       0, -R[2, 2],  R[2, 1],       0,       0,       0],
        [ R[0, 2],        0, -R[0, 0],       0,       0,       0],
        [ R[1, 2],        0, -R[1, 0],       0,       0,       0],
        [ R[2, 2],        0, -R[2, 0],       0,       0,       0],
        [-R[0, 1],  R[0, 0],        0,       0,       0,       0],
        [-R[1, 1],  R[1, 0],        0,       0,       0,       0],
        [-R[2, 1],  R[2, 0],        0,       0,       0,       0],
        [       0,        0,        0, R[0, 0], R[0, 1], R[0, 2]],
        [       0,        0,        0, R[1, 0], R[1, 1], R[1, 2]],
        [       0,        0,        0, R[2, 0], R[2, 1], R[2, 2]]
    ])

def dse3_left_jacobian(se3):
    # Returns 6x6x6 tensor
    # dlj such that dlj[:,:,i] is the
    # elementwise derivative of se3_left_jacobian(se3)
    # with respect to se3[i]

    w_x, w_y, w_z, p_x, p_y, p_z = se3.flatten()
    x0 = w_y**2
    x1 = w_x**2
    x2 = w_z**2
    x3 = x0 + x2
    x4 = x1 + x3
    x5 = np.sqrt(x4)
    x6 = np.sin(x5)
    x7 = x5 - x6
    x8 = x4**(-5/2)
    x9 = 3*x7*x8
    x10 = w_x*x9
    x11 = x0*x10
    x12 = x10*x2
    x13 = x4**(3/2)
    x14 = 1/x13
    x15 = 1/x5
    x16 = np.cos(x5)
    x17 = w_x*x15*x16
    x18 = w_x*x15 - x17
    x19 = x14*x18
    x20 = x0*x19
    x21 = x19*x2
    x22 = x11 + x12 - x20 - x21
    x23 = x14*x7
    x24 = w_y*x23
    x25 = w_x*w_y
    x26 = x1*x9
    x27 = w_y*x26
    x28 = -x27
    x29 = x19*x25 + x24 + x28
    x30 = w_x*w_z
    x31 = x14*x6
    x32 = x30*x31
    x33 = x16 - 1
    x34 = -x33
    x35 = x4**(-2)
    x36 = 2*x34*x35
    x37 = x30*x36
    x38 = -x32 + x37
    x39 = x29 + x38
    x40 = w_z*x23
    x41 = w_z*x26
    x42 = -x41
    x43 = x19*x30 + x40 + x42
    x44 = x25*x31
    x45 = x25*x36
    x46 = x44 - x45
    x47 = x43 + x46
    x48 = x32 - x37
    x49 = x29 + x48
    x50 = -x12
    x51 = w_x*x23
    x52 = 2*x51
    x53 = w_x**3
    x54 = x1*x19 + x52 - 3*x53*x7*x8
    x55 = -x21 - x50 - x54
    x56 = x34/x4
    x57 = w_y*x9
    x58 = x30*x57
    x59 = x56 + x58
    x60 = x1*x31 - x1*x36
    x61 = w_y*w_z*x14*x18 - x59 - x60
    x62 = -x44 + x45
    x63 = x43 + x62
    x64 = w_y*w_z
    x65 = x56 - x58
    x66 = x19*x64 + x60 + x65
    x67 = -x11
    x68 = -x20 - x54 - x67
    x69 = p_y*w_y
    x70 = 2*x69
    x71 = p_z*w_z
    x72 = 2*x71
    x73 = x70 + x72
    x74 = -x73
    x75 = p_x*x0
    x76 = 2*x75
    x77 = p_x*x2
    x78 = 2*x77
    x79 = x0/2 + x1/2 + x2/2 + x33
    x80 = -x79
    x81 = x35/2
    x82 = -x13/2 + 3*x5 - 3*x6
    x83 = x8*x82/2 - x80*x81
    x84 = p_x*w_x
    x85 = w_y*x84
    x86 = x69 + x71
    x87 = -x86
    x88 = w_y*x87
    x89 = -x88
    x90 = x85 + x89
    x91 = w_z*x69
    x92 = x71 + x84
    x93 = -x92
    x94 = w_z*x93
    x95 = -x94
    x96 = x91 + x95
    x97 = w_y*x71
    x98 = x69 + x84
    x99 = -x98
    x100 = w_y*x99
    x101 = -x100
    x102 = x101 + x97
    x103 = -x102
    x104 = w_y*x103
    x105 = w_z*x84
    x106 = w_z*x87
    x107 = -x106
    x108 = x105 + x107
    x109 = -x108
    x110 = w_z*x109
    x111 = w_y*x90 + w_z*x96 - x104 - x110
    x112 = -w_x*x15*x6 + w_x
    x113 = 2*w_x
    x114 = x4**(-3)
    x115 = x114*x80
    x116 = 5*x82/(2*x4**(7/2))
    x117 = 3*x5/2
    x118 = x8/2
    x119 = -w_x*x116 + x112*x81 + x113*x115 + x118*(-w_x*x117 + 3*w_x*x15 - 3*x17)
    x120 = p_x*w_z
    x121 = 2*x120
    x122 = -2*p_z*w_x + x121
    x123 = x35*x79
    x124 = -x122*x123
    x125 = p_x*w_y
    x126 = p_y*w_x
    x127 = x125 + x126
    x128 = x127 + x96
    x129 = p_z*x1
    x130 = p_z*x0
    x131 = x129 + x130
    x132 = x107 + x131 - 3*x91 + 2*x94
    x133 = x112*x35
    x134 = 4*x114*x79
    x135 = w_x*x134
    x136 = 2*x84
    x137 = x136 + x86
    x138 = -x137
    x139 = w_x*x69
    x140 = w_x*x93
    x141 = -x140
    x142 = x139 + x141
    x143 = -x142
    x144 = w_x*x103 + w_y*x143
    x145 = -p_z
    x146 = 2*x126
    x147 = 2*x125 - x146
    x148 = x123*x147
    x149 = p_z*w_x
    x150 = x120 + x149
    x151 = x100 + x150 - x97
    x152 = p_y*x1
    x153 = p_y*x2
    x154 = x152 + x153
    x155 = 3*p_z*w_y*w_z - 2*x100 - x154 - x89
    x156 = w_x*x71
    x157 = w_x*x99
    x158 = -x157
    x159 = x156 + x158
    x160 = -w_x*x96 - w_z*x159
    x161 = x122*x123
    x162 = -x105 + x106 + x127
    x163 = 3*p_x*w_x*w_z - 2*x106 - x131 - x95
    x164 = -w_x*x90 - w_y*x159
    x165 = 2*x23
    x166 = -p_x*x165
    x167 = -x136 - x72
    x168 = w_x*x137 - w_x*x138 + x142 + x159
    x169 = w_x*x143 - w_x*x159
    x170 = w_z*x96 - x110 - x169
    x171 = x136 + x73
    x172 = -x123*x171
    x173 = p_y*w_z
    x174 = p_z*w_y
    x175 = x173 + x174
    x176 = x159 + x175
    x177 = x75 + x77
    x178 = x141 - 3*x156 + 2*x157 + x177
    x179 = w_y*x83
    x180 = -x121*x179
    x181 = w_y*x109 + w_z*x103
    x182 = -x123*x147
    x183 = x150 + x90
    x184 = x101 + x154 - 3*x85 + 2*x88
    x185 = w_x*x109 + w_z*x143
    x186 = x123*x171
    x187 = -x139 + x140 + x175
    x188 = 3*p_y*w_x*w_y - 2*x140 - x158 - x177
    x189 = -w_y*x96 - w_z*x90
    x190 = -x136 - x70
    x191 = w_y*x90 - x104 - x169
    x192 = w_y*x15*x16
    x193 = w_y*x15 - x192
    x194 = x14*x193
    x195 = x194*x2
    x196 = x2*x57
    x197 = -x196
    x198 = 2*x24
    x199 = w_y**3
    x200 = x0*x194 + x198 - 3*x199*x7*x8
    x201 = -x195 - x197 - x200
    x202 = x194*x25 + x51 + x67
    x203 = x31*x64
    x204 = x36*x64
    x205 = -x203 + x204
    x206 = x202 + x205
    x207 = x0*x31 - x0*x36
    x208 = x194*x30 + x207 + x65
    x209 = x203 - x204
    x210 = x202 + x209
    x211 = x1*x194
    x212 = -x195 + x196 - x211 + x27
    x213 = w_z*x9
    x214 = x0*x213
    x215 = -x214
    x216 = x194*x64 + x215 + x40
    x217 = x216 + x62
    x218 = w_x*w_z*x14*x193 - x207 - x59
    x219 = x216 + x46
    x220 = -x200 - x211 - x28
    x221 = -p_y*x165
    x222 = 2*x153
    x223 = x70 + x92
    x224 = -x223
    x225 = w_y*x223 - w_y*x224 + x102 + x90
    x226 = -w_y*x15*x6 + w_y
    x227 = 2*w_y
    x228 = -w_y*x116 + x115*x227 + x118*(-w_y*x117 + 3*w_y*x15 - 3*x192) + x226*x81
    x229 = -2*p_z*w_y + 2*x173
    x230 = -x123*x229
    x231 = x226*x35
    x232 = w_y*x134
    x233 = -w_z*x146*x83
    x234 = x123*x229
    x235 = 2*x152
    x236 = w_z*x15*x16
    x237 = w_z*x15 - x236
    x238 = x14*x237
    x239 = x0*x238
    x240 = 2*x40
    x241 = w_z**3
    x242 = x2*x238 + x240 - 3*x241*x7*x8
    x243 = -x215 - x239 - x242
    x244 = x2*x31 - x2*x36
    x245 = w_x*w_y*x14*x237 - x244 - x59
    x246 = x238*x30 + x50 + x51
    x247 = x209 + x246
    x248 = x238*x25 + x244 + x65
    x249 = x1*x238
    x250 = -x242 - x249 - x42
    x251 = x197 + x238*x64 + x24
    x252 = x251 + x38
    x253 = x205 + x246
    x254 = x251 + x48
    x255 = x214 - x239 - x249 + x41
    x256 = -p_z*x165
    x257 = 2*x130
    x258 = x72 + x98
    x259 = -x258
    x260 = w_z*x258 - w_z*x259 + x108 + x96
    x261 = -w_z*x15*x6 + w_z
    x262 = 2*w_z
    x263 = -w_z*x116 + x115*x262 + x118*(-w_z*x117 + 3*w_z*x15 - 3*x236) + x261*x81
    x264 = x261*x35
    x265 = w_z*x134
    x266 = -2*x149*x179
    x267 = 2*x129
    x268 = x0*x113
    x269 = x113*x2
    x270 = x1*x227
    x271 = x270*x83
    x272 = 2*x123
    x273 = x272*x30
    x274 = -x14*x7*(w_y + x30) + x273
    x275 = x1*x262
    x276 = x275*x83
    x277 = x25*x272
    x278 = x23*(w_z - x25) + x277
    x279 = x23*(w_y - x30) + x273
    x280 = 2*x53
    x281 = -x1 + x3
    x282 = x227*x30*x83
    x283 = x1*x23 - 1/2
    x284 = -x14*x7*(w_z + x25) + x277
    x285 = 2*x199
    x286 = x2*x227
    x287 = x268*x83
    x288 = x272*x64
    x289 = -x14*x7*(w_x + x64) + x288
    x290 = x0*x23
    x291 = -x0 + x1 + x2
    x292 = x282 - 1/2
    x293 = x23*(w_x - x64) + x288
    x294 = x0*x262
    x295 = x294*x83
    x296 = x282 + 1/2
    x297 = 2*x241
    x298 = x2*x23
    x299 = x0 + x1 - x2
    x300 = x269*x83
    x301 = x286*x83

    lj_wx = np.array([
    [                                                                                                                x22,                                                                                                              x39,                                                                                                                 x47,   0,   0,   0],
    [                                                                                                                x49,                                                                                                              x55,                                                                                                                 x61,   0,   0,   0],
    [                                                                                                                x63,                                                                                                              x66,                                                                                                                 x68,   0,   0,   0],
    [                                                                   -x10*x74 + x111*x119 + x19*x74 + x83*(x76 + x78), -x10*x128 + x119*x144 + x124 + x128*x19 + x132*x133 - x132*x135 + x23*(p_y + x120) + x83*(w_y*x138 - x102 - x85), -x10*x151 + x119*x160 + x133*x155 - x135*x155 + x148 + x151*x19 + x23*(-x125 - x145) + x83*(-w_z*x137 - x105 - x96), x22, x39, x47],
    [ -x10*x162 + x119*x164 + x133*x163 - x135*x163 + x161 + x162*x19 + x23*(p_y - x120) + x83*(-w_y*x137 - 2*x85 - x89),                                                       -x10*x167 + x119*x170 + x166 + x167*x19 + x83*(x168 + x78),                                   -x10*x176 + x119*x181 + x133*x178 - x135*x178 + x137*x23 + x172 + x176*x19 + x180, x49, x55, x61],
    [-x10*x183 + x119*x185 + x133*x184 - x135*x184 + x182 + x183*x19 + x23*(p_z + x125) + x83*(w_z*x138 - 2*x105 + x106),                                -x10*x187 + x119*x189 + x133*x188 - x135*x188 + x138*x23 + x180 + x186 + x187*x19,                                                          -x10*x190 + x119*x191 + x166 + x19*x190 + x83*(x168 + x76), x63, x66, x68]])
    lj_wy = np.array([
        [                                                                                                              x201,                                                                                                                 x206,                                                                                                             x208,    0,    0,    0],
    [                                                                                                              x210,                                                                                                                 x212,                                                                                                             x217,    0,    0,    0],
    [                                                                                                              x218,                                                                                                                 x219,                                                                                                             x220,    0,    0,    0],
    [                                                         x111*x228 + x194*x74 + x221 - x57*x74 + x83*(x222 + x225), x128*x194 - x128*x57 + x132*x231 - x132*x232 + x144*x228 + x23*(p_x + x173) + x230 + x83*(w_x*x224 - w_x*x70 + x140),                                x151*x194 - x151*x57 + x155*x231 - x155*x232 + x160*x228 + x186 + x224*x23 + x233, x201, x206, x208],
    [x162*x194 - x162*x57 + x163*x231 - x163*x232 + x164*x228 + x23*(p_x - x173) + x234 + x83*(-w_x*x223 - x139 - x159),                                                                 x167*x194 - x167*x57 + x170*x228 + x83*(x222 + x235), x148 + x176*x194 - x176*x57 + x178*x231 - x178*x232 + x181*x228 + x23*(p_z + x126) + x83*(w_z*x224 - x108 - x91), x210, x212, x217],
    [                                 x172 + x183*x194 - x183*x57 + x184*x231 - x184*x232 + x185*x228 + x223*x23 + x233, x182 + x187*x194 - x187*x57 + x188*x231 - x188*x232 + x189*x228 + x23*(-x126 - x145) + x83*(-w_z*x223 - 2*x91 - x95),                                                      x190*x194 - x190*x57 + x191*x228 + x221 + x83*(x225 + x235), x218, x219, x220]])
    lj_wz = np.array([
    [                                                                                                               x243,                                                                                                              x245,                                                                                                                    x247,    0,    0,    0],
    [                                                                                                               x248,                                                                                                              x250,                                                                                                                    x252,    0,    0,    0],
    [                                                                                                               x253,                                                                                                              x254,                                                                                                                    x255,    0,    0,    0],
    [                                                         x111*x263 - x213*x74 + x238*x74 + x256 + x83*(x257 + x260),                               -x128*x213 + x128*x238 + x132*x264 - x132*x265 + x144*x263 + x172 + x23*x258 + x266, -x151*x213 + x151*x238 + x155*x264 - x155*x265 + x160*x263 + x23*(p_x - x174) + x230 + x83*(-w_x*x258 - w_x*x72 - x158), x243, x245, x247],
    [                                -x162*x213 + x162*x238 + x163*x264 - x163*x265 + x164*x263 + x186 + x23*x259 + x266,                                                     -x167*x213 + x167*x238 + x170*x263 + x256 + x83*(x260 + x267),     x161 - x176*x213 + x176*x238 + x178*x264 - x178*x265 + x181*x263 + x23*(p_y + x149) + x83*(w_y*x259 + x100 - 2*x97), x248, x250, x252],
    [-x183*x213 + x183*x238 + x184*x264 - x184*x265 + x185*x263 + x23*(p_x + x174) + x234 + x83*(w_x*x259 - x142 - x156), x124 - x187*x213 + x187*x238 + x188*x264 - x188*x265 + x189*x263 + x23*(p_y - x149) + x83*(-w_y*x258 - x90 - x97),                                                                  -x190*x213 + x190*x238 + x191*x263 + x83*(x257 + x267), x253, x254, x255]])
    lj_px = np.array([
    [                0,                           0,                        0, 0, 0, 0],
    [                0,                           0,                        0, 0, 0, 0],
    [                0,                           0,                        0, 0, 0, 0],
    [x83*(x268 + x269),                -x271 - x274,             -x276 + x278, 0, 0, 0],
    [     -x271 + x279,    -x52 + x83*(x269 + x280),  x123*x281 - x282 + x283, 0, 0, 0],
    [     -x276 - x284, -x281*x35*x79 - x282 - x283, -x52 + x83*(x268 + x280), 0, 0, 0]])

    lj_py = np.array([
    [                         0,                 0,                           0, 0, 0, 0],
    [                         0,                 0,                           0, 0, 0, 0],
    [                         0,                 0,                           0, 0, 0, 0],
    [ -x198 + x83*(x285 + x286),      -x287 - x289, -x290 - x291*x35*x79 - x292, 0, 0, 0],
    [              -x287 + x293, x83*(x270 + x286),                -x284 - x295, 0, 0, 0],
    [x290 + x291*x35*x79 - x296,       x278 - x295,   -x198 + x83*(x270 + x285), 0, 0, 0]])

    lj_pz = np.array([
    [                          0,                           0,                 0, 0, 0, 0],
    [                          0,                           0,                 0, 0, 0, 0],
    [                          0,                           0,                 0, 0, 0, 0],
    [  -x240 + x83*(x294 + x297), -x296 + x298 + x299*x35*x79,       x293 - x300, 0, 0, 0],
    [-x292 - x298 - x299*x35*x79,   -x240 + x83*(x275 + x297),      -x274 - x301, 0, 0, 0],
    [               -x289 - x300,                 x279 - x301, x83*(x275 + x294), 0, 0, 0]])

    result = np.empty((6,6,6))
    result[:,:,0] = lj_wx
    result[:,:,1] = lj_wy
    result[:,:,2] = lj_wz
    result[:,:,3] = lj_px
    result[:,:,4] = lj_py
    result[:,:,5] = lj_pz
    return result

if __name__ == "__main__":
    # T = se3_exp(np.random.uniform(-10, 10, (6,1)))
    
    w = np.random.uniform(-10, 10, (6,1))
    v = np.random.uniform(-10, 10, (6,1))
    epsilon = 1e-6
    lj = se3_left_jacobian(w)
    lj_plus = se3_left_jacobian(w + epsilon*v)

    dlj = dse3_left_jacobian(w)
    print("dlj.shape", dlj.shape)

    deriv_analytic = np.zeros((6,6))
    for i in range(6):
        deriv_analytic += dlj[:,:,i] * v[i,0]

    deriv_numeric = (lj_plus - lj)/epsilon
    # deriv_analytic = dSE3_adj_dse3(T, w)
    print("Numeric", deriv_numeric)
    print("Analytic", deriv_analytic)
    print("Diff", np.max(np.abs(deriv_numeric - deriv_analytic)))
    

    # epsilon = 1e-6
    # Adj = SE3_adj(T)
    # Adj_plus = SE3_adj(T @ se3_exp(epsilon*w))
    # dAdj_dt_numeric = (Adj_plus - Adj)/epsilon
    # dAdj_dt = dSE3_adj_dse3(T, w)
    # print("Numeric", dAdj_dt_numeric)
    # print("Analytic", dAdj_dt)
    # print("Diff", np.max(np.abs(dAdj_dt_numeric - dAdj_dt)))
