import numpy as np
from geometry import (
    se3_to_matrix,
    se3_to_vector,
    se3_exp,
    SE3_adj,
    SE3_inv,
    se3_left_jacobian)
    
from collections import namedtuple

POLY_ORDER = 5

def get_psi(L, qs, t_1, Dt):
    """
    Returns 4x4 matrix Psi, such that
    T(t_1 + Dt) ~= T(t_1)*exp(Psi), when T(s) is some frame
    that satisfies the matrix differential equation
    T'(s) = T(s)W(s) where W(s) is a twist matrix
    whose entries are specified by Chebyshev polynomials
    with coefficients in c_x, c_y, c_z

    L: Length of the beam
    qs: Length 3*POLY_ORDER array specifiing the coefficients of Chebyshev polynomial
        on domain [0, L]. The first POLY_ORDER coefficients are for the local x-twists,
        the next POLY_ORDER coefficients are for the local y-twists, and the last
        POLY_ORDER coefficients are for the local z-twists.
    t_1: The starting time of the expansion
    Dt: The ending time of the expansion
    """
    
    q_x = qs[:POLY_ORDER]
    q_y = qs[POLY_ORDER:2*POLY_ORDER]
    q_z = qs[-POLY_ORDER:]

    x0 = Dt**9
    x1 = q_z[4]/360
    x2 = Dt**2
    x3 = t_1*q_x[2]
    x4 = q_x[1]/2
    x5 = t_1**2
    x6 = 3*x5/2
    x7 = t_1**3
    x8 = 2*q_x[4]
    x9 = t_1*q_x[1]
    x10 = x5*q_x[2]
    x11 = x7*q_x[3]
    x12 = t_1**4
    x13 = x12*q_x[4]
    x14 = Dt**8
    x15 = q_z[4]/120
    x16 = t_1*q_y[3]
    x17 = q_z[4]/40
    x18 = t_1*q_y[4]
    x19 = Dt**7
    x20 = 3*q_z[4]/140
    x21 = q_z[3]/168
    x22 = t_1*q_y[2]
    x23 = q_z[4]/15
    x24 = q_y[3]*q_z[4]
    x25 = x5/10
    x26 = q_y[4]*q_z[3]
    x27 = Dt**6
    x28 = q_z[3]/48
    x29 = t_1*q_y[1]
    x30 = 3*q_z[4]/20
    x31 = q_z[3]/24
    x32 = x5*q_y[2]
    x33 = 7*q_z[4]/30
    x34 = x5*q_z[2]
    x35 = 7*q_y[4]/30
    x36 = x7*q_y[3]
    x37 = x7*q_z[3]
    x38 = Dt**5
    x39 = 3*q_z[3]/40
    x40 = q_z[2]/60
    x41 = q_z[1]/60
    x42 = 3*q_y[3]/40
    x43 = t_1*q_z[4]
    x44 = 3*x43/10
    x45 = q_z[3]/8
    x46 = q_z[1]/8
    x47 = 3*x18/10
    x48 = 2*x5/5
    x49 = x48*q_z[4]
    x50 = x34/8
    x51 = x48*q_z[1]
    x52 = q_y[2]*q_z[4]
    x53 = 13*x7/30
    x54 = q_y[4]*q_z[2]
    x55 = x12*q_z[4]
    x56 = 13*x55/40
    x57 = x12*q_y[4]
    x58 = 13*q_z[3]/40
    x59 = Dt**4
    x60 = t_1*q_x[4]
    x61 = q_x[3]/4
    x62 = q_z[2]/12
    x63 = q_y[2]/12
    x64 = t_1*q_z[3]
    x65 = x64/4
    x66 = q_z[1]/12
    x67 = x16/4
    x68 = x5/2
    x69 = x68*q_z[4]
    x70 = q_z[3]/4
    x71 = x5*x70
    x72 = q_y[3]/4
    x73 = x5*x72
    x74 = x68*q_y[4]
    x75 = q_y[1]/2
    x76 = x7*q_z[4]
    x77 = x37/6
    x78 = x36/6
    x79 = q_z[1]/2
    x80 = x7*x79
    x81 = 5*x55/12
    x82 = 5*q_z[2]/12
    x83 = t_1**5
    x84 = x83*q_z[4]
    x85 = x70*x83
    x86 = Dt**3
    x87 = t_1*q_x[3]
    x88 = q_y[1]/12
    x89 = t_1*q_z[2]
    x90 = x89/6
    x91 = x22/6
    x92 = x7/3
    x93 = x92*q_z[4]
    x94 = x92*q_y[4]
    x95 = x55/4
    x96 = x12*q_z[3]
    x97 = x12*x62
    x98 = q_z[1]/4
    x99 = x83/6
    x100 = t_1**6/12
    x101 = 2*q_y[4]
    x102 = q_x[3]*q_z[4]
    x103 = 3*q_x[3]/40
    x104 = 3*x60/10
    x105 = q_x[4]*q_z[2]
    x106 = q_z[0]/12
    x107 = x87/4
    x108 = x5*x61
    x109 = q_x[4]*q_z[0]
    x110 = x11/6
    x111 = x3/6
    x112 = q_x[1]/12
    x113 = q_x[3]*q_y[4]
    x114 = q_x[4]*q_y[3]
    x115 = 2*q_z[4]
    x116 = q_x[2]*q_y[4]
    x117 = q_x[4]*q_y[2]
    x118 = q_x[1]*q_y[4]
    x119 = q_x[2]*q_y[3]
    x120 = q_x[4]*q_y[1]
    x121 = q_y[4]/15
    x122 = q_x[4]*q_y[0]
    x123 = 7*q_x[4]/30
    x124 = q_y[3]/8
    x125 = q_y[0]/12
    return np.array([
        [                                                                                                                Dt*(x10 + x11 + x13 + x9 + q_x[0]) + x0*(x1*q_y[3] - q_y[4]*q_z[3]/360) + x14*(x15*q_y[2] + x16*x17 - x18*q_z[3]/40 - q_y[4]*q_z[2]/120) + x19*(-x18*q_z[2]/15 + x20*q_y[1] + x21*q_y[2] + x22*x23 + x24*x25 - x25*x26 - q_y[3]*q_z[2]/168 - 3*q_y[4]*q_z[1]/140) + x2*(x3 + x4 + x6*q_x[3] + x7*x8) + x27*(-x16*q_z[2]/24 - 3*x18*q_z[1]/20 + x22*x31 + x23*q_y[0] + x28*q_y[1] + x29*x30 + x32*x33 + x33*x36 - x34*x35 - x35*x37 - q_y[3]*q_z[1]/48 - q_y[4]*q_z[0]/15) + x38*(-x16*x46 + x29*x45 + x32*x45 + x39*q_y[0] + x40*q_y[1] - x41*q_y[2] - x42*q_z[0] + x44*q_y[0] - x47*q_z[0] + x49*q_y[1] - x50*q_y[3] - x51*q_y[4] + x52*x53 - x53*x54 + x56*q_y[3] - x57*x58 + q_x[4]/5) + x59*(-x22*x66 + x29*x62 - x57*x82 + x60 + x61 + x62*q_y[0] - x63*q_z[0] + x65*q_y[0] - x67*q_z[0] + x69*q_y[0] + x71*q_y[1] + x72*x84 - x73*q_z[1] - x74*q_z[0] + x75*x76 + x77*q_y[2] - x78*q_z[2] - x80*q_y[4] + x81*q_y[2] - x85*q_y[4]) + x86*(x100*x24 - x100*x26 - x32*x66 + x34*x88 + x5*x8 + x52*x99 - x54*x99 - x57*x98 + x63*x96 + x66*q_y[0] + x71*q_y[0] - x73*q_z[0] + x77*q_y[1] - x78*q_z[1] + x87 - x88*q_z[0] + x90*q_y[0] - x91*q_z[0] + x93*q_y[0] - x94*q_z[0] + x95*q_y[1] - x97*q_y[3] + q_x[2]/3)],
[Dt*(x29 + x32 + x36 + x57 + q_y[0]) + x0*(-x1*q_x[3] + q_x[4]*q_z[3]/360) + x14*(t_1*q_x[4]*q_z[3]/40 - x15*q_x[2] - x17*x87 + q_x[4]*q_z[2]/120) + x19*(t_1*q_x[4]*q_z[2]/15 - x102*x25 - x20*q_x[1] - x21*q_x[2] - x23*x3 + x5*q_x[4]*q_z[3]/10 + q_x[3]*q_z[2]/168 + 3*q_x[4]*q_z[1]/140) + x2*(x101*x7 + x22 + x6*q_y[3] + x75) + x27*(t_1*q_x[3]*q_z[2]/24 + 3*t_1*q_x[4]*q_z[1]/20 - x10*x33 - x11*x33 - x23*q_x[0] - x28*q_x[1] - x3*x31 - x30*x9 + 7*x5*q_x[4]*q_z[2]/30 + 7*x7*q_x[4]*q_z[3]/30 + q_x[3]*q_z[1]/48 + q_x[4]*q_z[0]/15) + x38*(-x10*x45 + x103*q_z[0] + x104*q_z[0] + x105*x53 + x13*x58 - x39*q_x[0] - x40*q_x[1] + x41*q_x[2] - x44*q_x[0] - x45*x9 + x46*x87 - x49*q_x[1] + x50*q_x[3] + x51*q_x[4] - x56*q_x[3] - 13*x76*q_x[2]/30 + q_y[4]/5) + x59*(x106*q_x[2] + x107*q_z[0] + x108*q_z[1] + x109*x68 + x110*q_z[2] + x13*x82 + x18 + x3*x66 - x4*x76 - x61*x84 - x62*x9 - x62*q_x[0] - x65*q_x[0] - x69*q_x[0] - x71*q_x[1] + x72 - x77*q_x[2] + x80*q_x[4] - x81*q_x[2] + x85*q_x[4]) + x86*(x10*x66 - x100*x102 + x100*q_x[4]*q_z[3] + x101*x5 + x105*x99 + x106*q_x[1] + x108*q_z[0] + x109*x92 + x110*q_z[1] + x111*q_z[0] - x112*x34 + x13*x98 + x16 - x66*q_x[0] - x71*q_x[0] - x77*q_x[1] - x90*q_x[0] - x93*q_x[0] - x95*q_x[1] - x96*q_x[2]/12 + x97*q_x[3] - x99*q_x[2]*q_z[4] + q_y[2]/3)],
[                       Dt*(t_1*q_z[1] + x34 + x37 + x55 + q_z[0]) + x0*(x113/360 - x114/360) + x14*(x116/120 - x117/120 - x60*q_y[3]/40 + x87*q_y[4]/40) + x19*(x113*x25 - x114*x25 + 3*x118/140 + x119/168 - 3*x120/140 + x121*x3 - x60*q_y[2]/15 - q_x[3]*q_y[2]/168) + x2*(x115*x7 + x6*q_z[3] + x79 + x89) + x27*(x10*x35 + x11*x35 + x121*q_x[0] - x122/15 - x123*x32 - x123*x36 + x3*q_y[3]/24 - 3*x60*q_y[1]/20 - x87*q_y[2]/24 + 3*x9*q_y[4]/20 + q_x[1]*q_y[3]/48 - q_x[3]*q_y[1]/48) + x38*(x10*x124 - x103*q_y[0] - x104*q_y[0] + x116*x53 - x117*x53 + x118*x48 - x120*x48 + x124*x9 - 13*x13*q_y[3]/40 - x32*q_x[3]/8 + x42*q_x[0] + x47*q_x[0] + 13*x57*q_x[3]/40 - x87*q_y[1]/8 + q_x[1]*q_y[2]/60 - q_x[2]*q_y[1]/60 + q_z[4]/5) + x59*(-x107*q_y[0] - x108*q_y[1] - x110*q_y[2] - x122*x68 - x125*q_x[2] - 5*x13*q_y[2]/12 - x3*x88 + x4*x7*q_y[4] + x43 + 5*x57*q_x[2]/12 + x61*x83*q_y[4] + x63*x9 + x63*q_x[0] + x67*q_x[0] - x7*x75*q_x[4] + x70 - x72*x83*q_x[4] + x73*q_x[1] + x74*q_x[0] + x78*q_x[2]) + x86*(-x10*x88 + x100*x113 - x100*x114 - x108*q_y[0] - x110*q_y[1] - x111*q_y[0] + x112*x32 + x115*x5 + x116*x99 - x117*x99 + x119*x12/12 - x12*x63*q_x[3] - x122*x92 - x125*q_x[1] - x13*q_y[1]/4 + x57*q_x[1]/4 + x64 + x73*q_x[0] + x78*q_x[1] + x88*q_x[0] + x91*q_x[0] + x94*q_x[0] + q_z[2]/3)],
[                             Dt],
[  -x23*x27 + x38*(-x39 - x44) + x59*(-x62 - x65 - x69) + x86*(-x66 - x71 - x90 - x93)],
[   x121*x27 + x38*(x42 + x47) + x59*(x63 + x67 + x74) + x86*(x73 + x88 + x91 + x94)]])

def get_dpsi_dq(L, qs, t_1, Dt):
    """Returns the 6x(POLY_ORDER*3) Jacobian
    for the psi twist with respect to the
    polynomial coefficients cs
    """

    q_x = qs[:POLY_ORDER]
    q_y = qs[POLY_ORDER:2*POLY_ORDER]
    q_z = qs[-POLY_ORDER:]

    x0 = Dt**2
    x1 = Dt*t_1 + x0/2
    x2 = Dt**3
    x3 = x2/3
    x4 = t_1**2
    x5 = Dt*x4 + t_1*x0 + x3
    x6 = Dt**4
    x7 = x6/4
    x8 = t_1**3
    x9 = t_1*x2
    x10 = Dt*x8 + 3*x0*x4/2 + x7 + x9
    x11 = Dt**5
    x12 = t_1**4
    x13 = x2*x4
    x14 = Dt*x12 + t_1*x6 + 2*x0*x8 + x11/5 + 2*x13
    x15 = Dt**6
    x16 = x15/15
    x17 = x16*q_z[4]
    x18 = t_1*q_z[4]
    x19 = 3*x18/10 + 3*q_z[3]/40
    x20 = q_z[2]/12
    x21 = q_z[3]/4
    x22 = x4*q_z[4]
    x23 = t_1*x21 + x20 + x22/2
    x24 = q_z[1]/12
    x25 = t_1/6
    x26 = x21*x4
    x27 = x8*q_z[4]
    x28 = x24 + x25*q_z[2] + x26 + x27/3
    x29 = Dt**7
    x30 = 3*x29/140
    x31 = x30*q_z[4]
    x32 = 3*x18/20 + q_z[3]/48
    x33 = q_z[3]/8
    x34 = t_1*x33 + 2*x22/5 + q_z[2]/60
    x35 = t_1*x20 + x26 + x27/2
    x36 = x12*q_z[4]
    x37 = x8/6
    x38 = x37*q_z[3] - q_z[0]/12
    x39 = x20*x4 + x36/4 + x38
    x40 = Dt**8
    x41 = x40/120
    x42 = x41*q_z[4]
    x43 = x18/15 + q_z[3]/168
    x44 = t_1/24
    x45 = 7*x22/30 + x44*q_z[3]
    x46 = 13*x27/30 + x33*x4 - q_z[1]/60
    x47 = -t_1*x24 + 5*x36/12 + x38
    x48 = q_z[3]/12
    x49 = t_1**5
    x50 = x12*x48 - x24*x4 - x25*q_z[0] + x49*q_z[4]/6
    x51 = Dt**9/360
    x52 = x51*q_z[4]
    x53 = x18*x40/40
    x54 = x22/10 - q_z[2]/168
    x55 = x44*q_z[2] - 7*x8*q_z[4]/30 + q_z[1]/48
    x56 = t_1*q_z[1]
    x57 = x4*q_z[2]
    x58 = -13*x12*q_z[4]/40 + x56/8 + x57/8 + 3*q_z[0]/40
    x59 = q_z[0]/4
    x60 = q_z[1]/4
    x61 = t_1*x59 + x37*q_z[2] + x4*x60 - x49*q_z[4]/4
    x62 = t_1**6
    x63 = x12*x20 + x37*q_z[1] + x4*x59 - x62*q_z[4]/12
    x64 = x51*q_z[3]
    x65 = t_1/40
    x66 = x65*q_z[3] + q_z[2]/120
    x67 = t_1/15
    x68 = x4/10
    x69 = x67*q_z[2] + x68*q_z[3] + 3*q_z[1]/140
    x70 = 7*x8/30
    x71 = 3*x56/20 + 7*x57/30 + x70*q_z[3] + q_z[0]/15
    x72 = 3*t_1/10
    x73 = 2*x4/5
    x74 = 13*x8/30
    x75 = 13*x12/40
    x76 = x72*q_z[0] + x73*q_z[1] + x74*q_z[2] + x75*q_z[3]
    x77 = x8/3
    x78 = x49/6
    x79 = x12*x60 + x48*x62 + x77*q_z[0] + x78*q_z[2]
    x80 = x4/2
    x81 = x8/2
    x82 = 5*x12/12
    x83 = x21*x49 + x80*q_z[0] + x81*q_z[1] + x82*q_z[2]
    x84 = x16*q_y[4]
    x85 = x72*q_y[4] + 3*q_y[3]/40
    x86 = q_y[2]/12
    x87 = q_y[3]/4
    x88 = t_1*x87 + x80*q_y[4] + x86
    x89 = x4*x87
    x90 = x25*q_y[2] + x77*q_y[4] + x89 + q_y[1]/12
    x91 = x30*q_y[4]
    x92 = 3*t_1/20
    x93 = x92*q_y[4] + q_y[3]/48
    x94 = q_y[3]/8
    x95 = t_1*x94 + x73*q_y[4] + q_y[2]/60
    x96 = t_1*x86 + x81*q_y[4] + x89
    x97 = q_y[4]/4
    x98 = x37*q_y[3] - q_y[0]/12
    x99 = x12*x97 + x4*x86 + x98
    x100 = x41*q_y[4]
    x101 = x67*q_y[4] + q_y[3]/168
    x102 = 7*x4/30
    x103 = x102*q_y[4] + x44*q_y[3]
    x104 = x4*x94 + x74*q_y[4] - q_y[1]/60
    x105 = -t_1*q_y[1]/12 + x82*q_y[4] + x98
    x106 = x12/12
    x107 = -t_1*q_y[0]/6 + x106*q_y[3] - x4*q_y[1]/12 + x78*q_y[4]
    x108 = x51*q_y[4]
    x109 = x40*x65
    x110 = x109*q_y[4]
    x111 = x68*q_y[4] - q_y[2]/168
    x112 = x44*q_y[2] - x70*q_y[4] + q_y[1]/48
    x113 = t_1/8
    x114 = x4/8
    x115 = x113*q_y[1] + x114*q_y[2] - x75*q_y[4] + 3*q_y[0]/40
    x116 = q_y[0]/4
    x117 = q_y[1]/4
    x118 = t_1*x116 + x117*x4 + x37*q_y[2] - x49*x97
    x119 = x62/12
    x120 = x116*x4 - x119*q_y[4] + x12*x86 + x37*q_y[1]
    x121 = x51*q_y[3]
    x122 = x65*q_y[3] + q_y[2]/120
    x123 = x67*q_y[2] + x68*q_y[3] + 3*q_y[1]/140
    x124 = x102*q_y[2] + x70*q_y[3] + x92*q_y[1] + q_y[0]/15
    x125 = x72*q_y[0] + x73*q_y[1] + x74*q_y[2] + x75*q_y[3]
    x126 = x117*x12 + x119*q_y[3] + x77*q_y[0] + x78*q_y[2]
    x127 = x49*x87 + x80*q_y[0] + x81*q_y[1] + x82*q_y[2]
    x128 = x16*q_x[4]
    x129 = x72*q_x[4] + 3*q_x[3]/40
    x130 = q_x[2]/12
    x131 = q_x[3]/4
    x132 = t_1*x131 + x130 + x80*q_x[4]
    x133 = q_x[1]/12
    x134 = x131*x4
    x135 = x133 + x134 + x25*q_x[2] + x77*q_x[4]
    x136 = x30*q_x[4]
    x137 = x92*q_x[4] + q_x[3]/48
    x138 = x113*q_x[3] + x73*q_x[4] + q_x[2]/60
    x139 = t_1*x130 + x134 + x81*q_x[4]
    x140 = x37*q_x[3] - q_x[0]/12
    x141 = x12*q_x[4]/4 + x130*x4 + x140
    x142 = x41*q_x[4]
    x143 = x67*q_x[4] + q_x[3]/168
    x144 = x102*q_x[4] + x44*q_x[3]
    x145 = x114*q_x[3] + x74*q_x[4] - q_x[1]/60
    x146 = -t_1*x133 + x140 + x82*q_x[4]
    x147 = x106*q_x[3] - x133*x4 - x25*q_x[0] + x78*q_x[4]
    x148 = x51*q_x[4]
    x149 = x109*q_x[4]
    x150 = x68*q_x[4] - q_x[2]/168
    x151 = x44*q_x[2] - 7*x8*q_x[4]/30 + q_x[1]/48
    x152 = x113*q_x[1] + x114*q_x[2] - 13*x12*q_x[4]/40 + 3*q_x[0]/40
    x153 = q_x[0]/4
    x154 = q_x[1]/4
    x155 = t_1*x153 + x154*x4 + x37*q_x[2] - x49*q_x[4]/4
    x156 = x12*x130 + x153*x4 + x37*q_x[1] - x62*q_x[4]/12
    x157 = x51*q_x[3]
    x158 = x65*q_x[3] + q_x[2]/120
    x159 = x67*q_x[2] + x68*q_x[3] + 3*q_x[1]/140
    x160 = x102*q_x[2] + x70*q_x[3] + x92*q_x[1] + q_x[0]/15
    x161 = x72*q_x[0] + x73*q_x[1] + x74*q_x[2] + x75*q_x[3]
    x162 = x119*q_x[3] + x12*x154 + x77*q_x[0] + x78*q_x[2]
    x163 = x131*x49 + x80*q_x[0] + x81*q_x[1] + x82*q_x[2]
    x164 = x2/12
    x165 = x6/12 + x9/6
    x166 = t_1*x7 + 3*x11/40 + x13/4
    x167 = x11*x72 + x16 + x3*x8 + x6*x80
    return np.array([
        [        Dt,          x1,   x5,x10,     x14,      x11*x19 + x17 + x2*x28 + x23*x6,       x11*x34 + x15*x32 + x2*x39 + x31 + x35*x6,        x11*x46 + x15*x45 + x2*x50 + x29*x43 + x42 + x47*x6,       -x11*x58 - x15*x55 - x2*x63 + x29*x54 + x52 + x53 - x6*x61,       -x11*x76 - x15*x71 - x2*x79 - x29*x69 - x40*x66 - x6*x83 - x64,    -x11*x85 - x2*x90 - x6*x88 - x84,     -x11*x95 - x15*x93 - x2*x99 - x6*x96 - x91, -x100 - x101*x29 - x103*x15 - x104*x11 - x105*x6 - x107*x2, -x108 + x11*x115 - x110 - x111*x29 + x112*x15 + x118*x6 + x120*x2,  x11*x125 + x121 + x122*x40 + x123*x29 + x124*x15 + x126*x2 + x127*x6],
        [-x11*x19 - x17 - x2*x28 - x23*x6, -x11*x34 - x15*x32 - x2*x39 - x31 - x35*x6,      -x11*x46 - x15*x45 - x2*x50 - x29*x43 - x42 - x47*x6,        x11*x58 + x15*x55 + x2*x63 - x29*x54 - x52 - x53 + x6*x61,         x11*x76 + x15*x71 + x2*x79 + x29*x69 + x40*x66 + x6*x83 + x64,    Dt,    x1,    x5,x10,    x14, x11*x129 + x128 + x132*x6 + x135*x2, x11*x138 + x136 + x137*x15 + x139*x6 + x141*x2,  x11*x145 + x142 + x143*x29 + x144*x15 + x146*x6 + x147*x2, -x11*x152 + x148 + x149 - x15*x151 + x150*x29 - x155*x6 - x156*x2, -x11*x161 - x15*x160 - x157 - x158*x40 - x159*x29 - x162*x2 - x163*x6],
    [ x11*x85 + x2*x90 + x6*x88 + x84,  x11*x95 + x15*x93 + x2*x99 + x6*x96 + x91, x100 + x101*x29 + x103*x15 + x104*x11 + x105*x6 + x107*x2, x108 - x11*x115 + x110 + x111*x29 - x112*x15 - x118*x6 - x120*x2, -x11*x125 - x121 - x122*x40 - x123*x29 - x124*x15 - x126*x2 - x127*x6, -x11*x129 - x128 - x132*x6 - x135*x2, -x11*x138 - x136 - x137*x15 - x139*x6 - x141*x2, -x11*x145 - x142 - x143*x29 - x144*x15 - x146*x6 - x147*x2, x11*x152 - x148 - x149 + x15*x151 - x150*x29 + x155*x6 + x156*x2, x11*x161 + x15*x160 + x157 + x158*x40 + x159*x29 + x162*x2 + x163*x6,   Dt,   x1,    x5, x10,     x14],
    [0,0,    0,  0,       0,     0,     0,     0,  0,      0,    0,    0,     0,   0,       0],
    [0,0,    0,  0,       0,     0,     0,     0,  0,      0,    0,-x164, -x165,        -x166,   -x167],
        [0,0,    0,  0,       0,     0,  x164,  x165,        x166,   x167,    0,    0,     0,   0,       0]])

def get_psi_segments(L, qs, num_segments):
    segments = []
    dsegment_dqs = []

    current_l = 0
    segment_length = L/num_segments
    for i in range(num_segments):
        segment = get_psi(L, qs, current_l, segment_length)
        dsegment_dq = get_dpsi_dq(L, qs, current_l, segment_length)
        segments.append(segment)
        dsegment_dqs.append(dsegment_dq)
        current_l += segment_length

    return segments, dsegment_dqs

def get_endpoint(psis, dpsi_dqs):
    exp_psis = [se3_exp(seg) for seg in psis]

    tail = np.eye(4)
    tails = []
    for exp_seg in exp_psis[::-1]:
        tail = exp_seg @ tail
        tails.append(tail)

    tails.reverse()

    dfinal_dq = np.zeros((6, 3*POLY_ORDER))
    for i in range(len(psis)):
        tail = tails[i]
        psi = psis[i]
        dpsi_dq = dpsi_dqs[i]
        dfinal_dq += SE3_adj(SE3_inv(tail)) @ se3_left_jacobian(psi) @ dpsi_dq

    return tails[0], dfinal_dq

def get_dendpoint_dtheta(endpoint):
    # endpoint -> exp(twist*theta) @ endpoint
    return SE3_adj(SE3_inv(endpoint)) @ np.array([[1, 0, 0, 0, 0, 0]]).T


if __name__ == "__main__":
    import time
    
    print("OK")
    np.set_printoptions(precision=4)

    length = 3.2

    cs = np.random.uniform(-1, 1, POLY_ORDER*3)
    # cs = np.zeros((POLY_ORDER*3,))

    t1 = 0.5
    t2 = 1.2345

    standard_total = 0
    repeats = 1000
    for i in range(repeats):
        start = time.monotonic()
        psi = get_psi(length, cs, t1, t2)
        end = time.monotonic()
        standard_total += end - start
    print("Standard time", standard_total)

    psi = get_psi(length, cs, t1, t2)

    epsilon = 1e-5
    dcs = np.random.uniform(-1,1,POLY_ORDER*3)
    print("dcs", dcs)

    psi_plus = get_psi(length, cs+epsilon*dcs, t1, t2)
    print("psi_plus\n", psi_plus.flatten())

    delta_psi = psi_plus - psi
    print("delta_psi\n", delta_psi.flatten())

    psi_prime_numeric = (psi_plus - psi)/epsilon
    print("psi' numeric", psi_prime_numeric.flatten())

    dpsi_dq = get_dpsi_dq(length, cs, t1, t2)
    psi_prime_analytic = dpsi_dq @ dcs
    print("psi' analytic", psi_prime_analytic.flatten())

    print("="*20)

    psis, dpsi_dqs = get_psi_segments(length, cs, 10)
    psis_plus, _ = get_psi_segments(length, cs + epsilon*dcs, 10)
    endpoint, dendpoint_dq = get_endpoint(psis, dpsi_dqs)
    endpoint_plus, _ = get_endpoint(psis_plus, _)

    print("endpoint", endpoint)
    print("dendpoint_dq", dendpoint_dq)

    print((dendpoint_dq @ dcs.reshape(15,1)).shape)

    print("endpoint' numeric", (endpoint_plus - endpoint)/epsilon)
    print("endpoint' analytic", endpoint @ se3_to_matrix(dendpoint_dq @ dcs.reshape(15,1)))

    print("DONE")
    
