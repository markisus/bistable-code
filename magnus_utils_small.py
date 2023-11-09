import numpy as np
from geometry import *
    
from collections import namedtuple

POLY_ORDER = 3

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

    x0 = Dt**2
    x1 = t_1*q_x[2]
    x2 = t_1*q_x[1]
    x3 = t_1**2
    x4 = x3*q_x[2]
    x5 = Dt**5
    x6 = q_z[2]/60
    x7 = Dt**4
    x8 = q_z[2]/12
    x9 = q_z[0]/12
    x10 = t_1*q_y[1]
    x11 = t_1*q_y[2]
    x12 = q_z[1]/12
    x13 = Dt**3
    x14 = q_y[1]/12
    x15 = t_1*q_z[2]
    x16 = x15/6
    x17 = x11/6
    x18 = x3*q_z[2]
    x19 = x3*q_y[2]
    x20 = x1/6
    x21 = q_x[1]/12
    x22 = q_y[2]/12
    x23 = q_y[0]/12
    return np.array([
        [                    Dt*(x2 + x4 + q_x[0]) + x0*(x1 + q_x[1]/2) + x13*(-x12*x19 + x12*q_y[0] + x14*x18 - x14*q_z[0] + x16*q_y[0] - x17*q_z[0] + q_x[2]/3) + x5*(x6*q_y[1] - q_y[2]*q_z[1]/60) + x7*(x10*x8 - x11*x12 + x8*q_y[0] - x9*q_y[2])],
        [Dt*(x10 + x19 + q_y[0]) + x0*(x11 + q_y[1]/2) + x13*(x12*x4 - x12*q_x[0] - x16*q_x[0] - x18*x21 + x20*q_z[0] + x9*q_x[1] + q_y[2]/3) + x5*(-x6*q_x[1] + q_x[2]*q_z[1]/60) + x7*(t_1*q_x[2]*q_z[1]/12 - x2*x8 - x8*q_x[0] + q_x[2]*q_z[0]/12)],
        [  Dt*(t_1*q_z[1] + x18 + q_z[0]) + x0*(x15 + q_z[1]/2) + x13*(-x14*x4 + x14*q_x[0] + x17*q_x[0] + x19*x21 - x20*q_y[0] - x23*q_x[1] + q_z[2]/3) + x5*(q_x[1]*q_y[2]/60 - q_x[2]*q_y[1]/60) + x7*(-x1*x14 + x2*x22 + x22*q_x[0] - x23*q_x[2])],
        [                                                                              Dt],
        [                                                        x13*(-x12 - x16) - x7*x8],
        [                                                      x13*(x14 + x17) + x22*x7]])


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
    x3 = t_1**2
    x4 = Dt*x3 + t_1*x0 + x2/3
    x5 = Dt**4
    x6 = x5/12
    x7 = x6*q_z[2]
    x8 = q_z[1]/12
    x9 = t_1/6
    x10 = x8 + x9*q_z[2]
    x11 = q_z[0]/12
    x12 = x3/12
    x13 = -x11 + x12*q_z[2]
    x14 = Dt**5/60
    x15 = t_1*x7 + x14*q_z[2]
    x16 = x14*q_z[1]
    x17 = t_1*x8 + x11
    x18 = x3*x8 + x9*q_z[0]
    x19 = x6*q_y[2]
    x20 = q_y[1]/12
    x21 = x20 + x9*q_y[2]
    x22 = q_y[0]/12
    x23 = x12*q_y[2] - x22
    x24 = t_1*x19 + x14*q_y[2]
    x25 = x14*q_y[1]
    x26 = t_1*x20 + x22
    x27 = x20*x3 + x9*q_y[0]
    x28 = x6*q_x[2]
    x29 = q_x[1]/12
    x30 = x29 + x9*q_x[2]
    x31 = q_x[0]/12
    x32 = x12*q_x[2] - x31
    x33 = t_1*x28 + x14*q_x[2]
    x34 = x14*q_x[1]
    x35 = t_1*x29 + x31
    x36 = x29*x3 + x9*q_x[0]
    x37 = x2/12
    x38 = x2*x9 + x6
    return np.array([
    [          Dt,            x1,                     x4,   x10*x2 + x7,  x13*x2 + x15, -x16 - x17*x5 - x18*x2, -x19 - x2*x21, -x2*x23 - x24,  x2*x27 + x25 + x26*x5],
    [-x10*x2 - x7, -x13*x2 - x15,  x16 + x17*x5 + x18*x2,            Dt,            x1,                     x4,  x2*x30 + x28,  x2*x32 + x33, -x2*x36 - x34 - x35*x5],
    [x19 + x2*x21,  x2*x23 + x24, -x2*x27 - x25 - x26*x5, -x2*x30 - x28, -x2*x32 - x33,  x2*x36 + x34 + x35*x5,            Dt,            x1,                     x4],
    [           0,             0,                      0,             0,             0,                      0,             0,             0,                      0],
    [           0,             0,                      0,             0,             0,                      0,             0,          -x37,                   -x38],
    [           0,             0,                      0,             0,           x37,                    x38,             0,             0,                      0]])

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

def get_endpoint_(psis, dpsi_dqs):
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

def get_endpoint(psis, dpsi_dqs, compute_hess=False):
    assert dpsi_dqs[0].shape[0] == 6
    num_params = dpsi_dqs[0].shape[1]

    exp_psis = [se3_exp(seg) for seg in psis]

    tail = np.eye(4)
    tails = []
    for exp_seg in exp_psis[::-1]:
        tail = exp_seg @ tail
        tails.append(tail)
    tails.reverse()

    # tails[i] holds exp(psis[i])exp(psis[i+1])...exp(psis[N-1])
    dfinal_dq = np.zeros((6, num_params))
    for i in range(len(psis)):
        tail = tails[i]
        psi = psis[i]
        dpsi_dq = dpsi_dqs[i]
        dfinal_dq += SE3_adj(SE3_inv(tail)) @ se3_left_jacobian(psi) @ dpsi_dq
    
    d2final_dq2 = np.zeros((6, num_params, num_params))
    dfinal_dq_sanity = np.zeros((6, num_params))
    if compute_hess:
        adj_invs = []    
        for exp_seg in exp_psis:
            adj_invs.append(SE3_adj(SE3_inv(exp_seg)))
        # adj_invs[i] = adj(SE3_inv(se3_exp(psis[i])))

        adj_inv_tails = []
        for tail in tails:
            adj_inv_tails.append(SE3_adj(SE3_inv(tail)))
        # adj_inv_tails[i] = adj(se3_inv(exp(psis[i])exp(psis[i+1])...exp(psis[M-1])))

        f_prev = None #np.zeros((6, 6, num_params))
        for i in reversed(range(len(psis))):
            # print(f"i={i}")
            # print("psi", psis[i].flatten())
            # print("dpsi_dq", dpsi_dqs[i])
            adj_inv = adj_invs[i]
            dadj_inv = np.zeros((6, 6, num_params))
            for k in range(num_params):
                ek = dpsi_dqs[i][:,k]
                da = -dSE3_adj(se3_left_jacobian(-psis[i]) @ ek) @ adj_inv
                dadj_inv[:,:,k] = da

            # print("dadj_inv[0,:,:]", dadj_inv[0,:,:])
            if i == len(psis) - 1:
                # print("base case")
                f_curr = dadj_inv
            else:
                # print("non base case")
                f_curr = np.zeros((6, 6, num_params))
                for k in range(num_params):
                    f_curr[:,:,k] = prev[:,:,k] @ adj_inv + adj_inv_tails[i+1] @ dadj_inv[:,:,k]

            psi = psis[i]
            dpsi_dq = dpsi_dqs[i]
            lj_psi = se3_left_jacobian(psi)
            dlj_psi = dse3_left_jacobian(psi)
            acc = np.zeros((6, num_params, num_params))
            for k in range(num_params):
                # print("num params", num_params)
                # print("f_curr.shape", f_curr.shape)
                # print("dlj_psi", dlj_psi.shape)
                djl_psi_k = np.einsum("ijd,d", dlj_psi, dpsi_dq[:,k].flatten())
                gk = f_curr[:,:,k] @ lj_psi @ dpsi_dq + adj_inv_tails[i] @ djl_psi_k @ dpsi_dq
                     # 6x6           6x6      6xN       6x6                6x6         6xN
                     #..............6xN..............   ..............6xN...............................
                acc[:,:,k] = gk

            # #### acc sanity check
            # # # f_curr[:,:,k] = d/dk Ad(inv(tail[i]))
            # eps = 1e-6
            # k = 0
            # dq = np.zeros((num_params,1))
            # dq[k] = 1
            # tail_plus = np.eye(4)
            # for i_ in range(i, len(psis)):
            #     tail_plus = tail_plus @ se3_exp(psis[i_] + dpsi_dqs[i_] @ dq * eps)
            # # print("tail", tails[i])
            # # print("tail_plus", tail_plus)
            # adj_inv_tail_plus = SE3_adj(SE3_inv(tail_plus))
            # # print("adj_inv_tail\n", adj_inv_tails[i])
            # # print("adj_inv_tail_alt", SE3_adj(SE3_inv(tails[i])))
            # # print("adj_inv_tail_plus\n", adj_inv_tail_plus)
            # dadj_inv_tail_k = (adj_inv_tail_plus - adj_inv_tails[i])/eps
            # # print("dadj_inv_tail_k numeric\n", dadj_inv_tail_k)
            # # print("f_curr\n", f_curr[:,:,k])

            # lj_psi_plus = se3_left_jacobian(psi + dpsi_dq @ dq * eps)

            # t_plus = adj_inv_tail_plus @ lj_psi_plus @ dpsi_dq
            # t_curr = adj_inv_tails[i] @ lj_psi @ dpsi_dq

            # # print("t_plus - t_curr", t_plus - t_curr)
            # # print("deriv n", (t_plus - t_curr).flatten()/eps)
            # # print("acc shape", acc.shape)
            # # print("acc", acc.flatten())

            # print("adding", t_curr.flatten(), "to dfinal_dqsanity, i=", i)
            # # print("\ttail", tails[i].flatten())
            # # print("\tleftj", lj_psi_plus.flatten())
            # # print("\tadjinvtail", adj_inv_tails[i].flatten())
            # print("\tdpsi_dq", dpsi_dq.flatten())
            # dfinal_dq_sanity += t_curr
            # # ### end sanity check
            
            d2final_dq2 += acc
            prev = f_curr

    if compute_hess:
        # print("d2final_dq2", d2final_dq2.flatten())
        # print("dfinal_dq_sanity", dfinal_dq_sanity.flatten())
        # # sanity check
        # eps = 1e-4
        # k = 0
        # dq = np.zeros((num_params,1))
        # dq[k] = 1

        # psis_plus = []
        # for i in range(len(psis)):
        #     psi_plus = psis[i] + dpsi_dqs[i] @ dq * eps
        #     psis_plus.append(psi_plus)

        # # for psi, psi_plus in zip(psis, psis_plus):
        # #     print("psi", psi.flatten())
        # #     print("psi_plus", psi_plus.flatten())
        
        # tails_plus = []
        # for i in range(len(psis)):
        #     tail_plus = np.eye(4)
        #     for i_ in range(i, len(psis)):
        #         tail_plus = tail_plus @ se3_exp(psis_plus[i_])
        #     tails_plus.append(tail_plus)

        # # for tail, tail_plus in zip(tails, tails_plus):
        # #     print("tail", tail.flatten())
        # #     print("tail_plus", tail_plus.flatten())


        # print("FINAL DPSI_DQS")
        # for dpsi_dq in dpsi_dqs:
        #     print("\t", dpsi_dq.flatten())

        # dfinal_dq_plus = np.zeros((6, num_params))                
        # for i in range(len(psis)):
        #     tail_plus = tails_plus[i]
        #     psi_plus = psis_plus[i]
        #     dpsi_dq = dpsi_dqs[i]
        #     dfinal_dq_plus_inc = SE3_adj(SE3_inv(tail_plus)) @ se3_left_jacobian(psi_plus) @ dpsi_dq
        #     print("adding ", dfinal_dq_plus_inc.flatten(), " to dfinal_dq_plus, i=", i)
        #     # print("\ttail", tail_plus.flatten())
        #     # print("\tleftj", se3_left_jacobian(psi_plus).flatten())
        #     # print("\tadjinvtail", SE3_adj(SE3_inv(tail_plus)).flatten())
        #     print("\tdpsi_dq", dpsi_dq.flatten())
        #     dfinal_dq_plus += SE3_adj(SE3_inv(tail_plus)) @ se3_left_jacobian(psi_plus) @ dpsi_dq

        # delta = (dfinal_dq_plus - dfinal_dq)
        # print("delta", delta.flatten())

        # diff_numeric = (dfinal_dq_plus - dfinal_dq)/eps
        # # print("dif", delta.flatten())
        # print("dif numeric", diff_numeric.flatten())
        # print("dif a", d2final_dq2.flatten())
        
        # # ### end sanity check
        return tails[0], dfinal_dq, d2final_dq2

    return tails[0], dfinal_dq

def get_dendpoint_dtheta(endpoint):
    # endpoint -> exp(twist*theta) @ endpoint
    return SE3_adj(SE3_inv(endpoint)) @ np.array([[1, 0, 0, 0, 0, 0]]).T

# def test_func(se3):
#     result = SE3_adj(se3_exp(-se3)) @ se3_left_jacobian(se3)
#     # result = SE3_adj(se3_exp(-se3)) 
#     # result = se3_left_jacobian(se3)
#     return result

# def test_func2(se3):
#     # result = SE3_adj(se3_exp(-se3)) @ se3_left_jacobian(se3)
#     result = SE3_adj(se3_exp(-se3)) 
#     # result = se3_left_jacobian(se3)
#     return result

# # helper function
# def sum_adj_string(psis, dpsi_dqs):
#     exp_psis = [se3_exp(seg) for seg in psis]

#     tail = np.eye(4)
#     tails = []

#     for exp_seg in exp_psis[::-1]:
#         tail = exp_seg @ tail
#         tails.append(tail)
#     tails.reverse()
#     # tails[i] = exp(psis[i]) exp(psis[i+1]) ... exp(psis[M-1])

#     adj_invs = []    
#     for exp_seg in exp_psis:
#         adj_invs.append(SE3_adj(SE3_inv(exp_seg)))
#     # adj_invs[i] = adj(SE3_inv(se3_exp(psis[i])))

#     adj_inv_tails = []
#     for tail in tails:
#         adj_inv_tails.append(SE3_adj(SE3_inv(tail)))
#     # adj_inv_tails[i] = adj(se3_inv(exp(psis[i])exp(psis[i+1])...exp(psis[M-1])))

#     # compute derivative
#     num_params = dpsi_dqs[0].shape[1]

#     dresult_dq = np.zeros((6, 6, num_params))
#     prev = np.zeros((6, 6, num_params))
#     for i in reversed(range(len(psis))):
#         # print(f"i={i}")
#         # print("psi", psis[i].flatten())
#         # print("dpsi_dq", dpsi_dqs[i])
#         adj_inv = adj_invs[i]
#         dadj_inv = np.zeros((6, 6, num_params))
#         for k in range(num_params):
#             ek = dpsi_dqs[i][:,k]
#             da = -dSE3_adj(se3_left_jacobian(-psis[i]) @ ek) @ adj_inv
#             dadj_inv[:,:,k] = da

#         # print("dadj_inv[0,:,:]", dadj_inv[0,:,:])

#         if i == len(psis) - 1:
#             # print("base case")
#             acc = dadj_inv
#         else:
#             # print("non base case")
#             acc = np.zeros((6, 6, num_params))
#             for k in range(num_params):
#                 acc[:,:,k] = prev[:,:,k] @ adj_inv + adj_inv_tails[i+1] @ dadj_inv[:,:,k]

#         dresult_dq += acc
#         prev = acc

#     result = sum(adj_inv_tails)

#     # print("dresult_dq[0,:,:]\n", dresult_dq[0,:,:])
#     return result, dresult_dq


# def dSE3_adj_se3_exp_neg(se3, ds_dp):
#     assert ds_dp.shape[0] == 6
#     num_params = ds_dp.shape[1]

#     result = np.zeros((6,6, num_params))

#     SE3_adj_se3_exp_neg_se3 = SE3_adj(se3_exp(-se3))    
#     for k in range(num_params):
#         ek = ds_dp[:,k]
#         da = -dSE3_adj(se3_left_jacobian(-se3) @ ek) @ SE3_adj_se3_exp_neg_se3
#         result[:,:,k] = da

#     return result

# def deriv_test_func(se3, ds_dp):
#     # the matrix derivative with respect to the kth component
#     # d_dk [SE3_adj(se3_exp(-se3)) @ se3_left_jacobian(se3)] = 
#     # d_dk [SE3_adj(se3_exp(-se3))] @ se3_left_jacobian(se3) + SE3_adj(se3_exp(-se3)) @ d_dk [se3_left_jacobian(se3)] =
#     # -dSE3_adj(Jl[-se3] ek) SE3_adj(exp(-se3)) @ se3_left_jacobian(se3) + SE3_adj(se3_exp(-se3)) @ d_dk [se3_left_jacobian(se3)] =
#     # -dSE3_adj(Jl[-se3] ek) SE3_adj(exp(-se3)) @ se3_left_jacobian(se3) + SE3_adj(se3_exp(-se3)) @  dse3_left_jacobian(se3)[k,:,:]

#     assert ds_dp.shape[0] == 6
#     num_params = ds_dp.shape[1]

#     result = np.zeros((6,6,num_params))
#     a = SE3_adj(se3_exp(-se3))    
#     b = se3_left_jacobian(se3)

#     dse3_left_jac = dse3_left_jacobian(se3)
#     SE3_adj_se3_exp_neg_se3 = SE3_adj(se3_exp(-se3))

#     dSE3_adj_se3_exp_neg_result = dSE3_adj_se3_exp_neg(se3, ds_dp)

#     for k in range(num_params):
#         ek = ds_dp[:,k]
#         # da = -dSE3_adj(se3_left_jacobian(-se3) @ ek) @ SE3_adj_se3_exp_neg_se3
#         da = dSE3_adj_se3_exp_neg_result[k,:,:]
#         db = np.einsum("ijl,l", dse3_left_jac, ek.flatten())
#         increment = a @ db + da @ b
#         result[:,:,k] = increment

#     return result

# def deriv_test_func2(se3, ds_dp):
#     # the matrix derivative with respect to the kth component
#     # d_dk [SE3_adj(se3_exp(-se3)) @ se3_left_jacobian(se3)] = 
#     # d_dk [SE3_adj(se3_exp(-se3))] @ se3_left_jacobian(se3) + SE3_adj(se3_exp(-se3)) @ d_dk [se3_left_jacobian(se3)] =
#     # -dSE3_adj(Jl[-se3] ek) SE3_adj(exp(-se3)) @ se3_left_jacobian(se3) + SE3_adj(se3_exp(-se3)) @ d_dk [se3_left_jacobian(se3)] =
#     # -dSE3_adj(Jl[-se3] ek) SE3_adj(exp(-se3)) @ se3_left_jacobian(se3) + SE3_adj(se3_exp(-se3)) @  dse3_left_jacobian(se3)[k,:,:]

#     assert ds_dp.shape[0] == 6
#     num_params = ds_dp.shape[1]

#     result = np.zeros((6,6,num_params))
#     a = SE3_adj(se3_exp(-se3))    
#     b = se3_left_jacobian(se3)

#     dse3_left_jac = dse3_left_jacobian(se3)
#     SE3_adj_se3_exp_neg_se3 = SE3_adj(se3_exp(-se3))

#     dSE3_adj_se3_exp_neg_result = dSE3_adj_se3_exp_neg(se3, ds_dp)

#     for k in range(num_params):
#         ek = ds_dp[:,k]
#         # da = -dSE3_adj(se3_left_jacobian(-se3) @ ek) @ SE3_adj_se3_exp_neg_se3
#         da = dSE3_adj_se3_exp_neg_result[:,:,k]
#         # db = np.einsum("lij,l", dse3_left_jac, ek.flatten())
#         increment = da
#         result[:,:,k] = increment

#     return result


if __name__ == "__main__":
    import time
    
    print("OK")
    np.set_printoptions(precision=4, suppress=True)

    length = 3.2
    eps = 1e-6
    num_params = 1

    dqs = []
    dpsi_dqs = []
    psis = []
    psis_plus = []

    dq = np.random.uniform(-1,1, (num_params,1))    
    # dq = np.zeros((num_params,1))
    # dq[0] = 1
    for i in range(10):
        psi = np.random.uniform(-1,1, (6,1))
        dpsi_dq = np.random.uniform(-1,1, (6,num_params))
        # dpsi_dq = np.eye(6)
        dpsi = dpsi_dq @ dq
        dqs.append(dq)
        psis.append(psi)
        dpsi_dqs.append(dpsi_dq)
        psis_plus.append(psi + eps*dpsi)

    endpoint, dendpoint_dq, d2endpoint_dq2 = get_endpoint(psis, dpsi_dqs, compute_hess=True)
    endpoint_plus, dendpoint_dq_plus = get_endpoint(psis_plus, dpsi_dqs, compute_hess=False)

    deriv_n = (dendpoint_dq_plus - dendpoint_dq)/eps
    deriv_aa = np.einsum("ijk,k", d2endpoint_dq2, dq.flatten())

    print("Deriv n\n", deriv_n)
    print("Deriv a\n", deriv_aa)

    print("DONE")
    
