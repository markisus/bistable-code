from matplotlib import pyplot as plt
import numpy as np
# import cvxpy as cp
import scipy as sp
# import open3d as o3d
import time
from magnus_utils_small import *
# from magnus_utils_tiny import *
# from magnus_utils import *
from geometry import *
import functools

MAGNUS_SEGMENTS = 6
angle = np.pi/8
c = np.cos(angle)
s = np.sin(angle)

initial_frame = np.array([
        c, -s, 0, 0,
        s, c, 0, 0.1,
        0, 0, 1, 0,
        0, 0, 0, 1]).reshape((4,4), order='C')

alpha_x = 1
alpha_y = 1
alpha_z = 10

def get_energy_matrix(L):
    if POLY_ORDER == 5:
        return get_energy_matrix_5(L)
    elif POLY_ORDER == 3:
        return get_energy_matrix_3(L)
    elif POLY_ORDER == 2:
        return get_energy_matrix_2(L)    
    raise RuntimeError("Unsupported poly order")

def get_energy_matrix_2(L):
    x0 = L**2/2
    return np.array([
        [ L,     x0],
        [x0, L**3/3]])

def get_energy_matrix_5(L):
    x0 = L**2/2
    x1 = L**3/3
    x2 = L**4/4
    x3 = L**5/5
    x4 = L**6/6
    x5 = L**7/7
    x6 = L**8/8
    return np.array([
    [ L, x0, x1, x2,     x3],
    [x0, x1, x2, x3,     x4],
    [x1, x2, x3, x4,     x5],
    [x2, x3, x4, x5,     x6],
    [x3, x4, x5, x6, L**9/9]
    ])

def get_energy_matrix_3(L):
    x0 = L**2/2
    x1 = L**3/3
    x2 = L**4/4
    return np.array([
    [ L, x0,     x1],
    [x0, x1,     x2],
    [x1, x2, L**5/5]])

def gradient_descent(length, current_qs):
    current_params = current_qs
    current_objective, current_dobjective_dq = get_objective(length, current_params)

    multiplier = 1e-11
    stuck_count = 0

    iteration = 0
    while True:
        iteration += 1
        # try to take a step in gradient direction
        dq = -multiplier * current_dobjective_dq

        new_params = current_params + dq
        new_objective, new_dobjective_dq = get_objective(length, new_params)

        if new_objective < current_objective:
            stuck_count = 0
            multiplier *= 1.1

            current_objective = new_objective
            current_params = new_params
            current_dobjective_dq = new_dobjective_dq
            multiplier = min(1e-1, multiplier)
        else:
            # try taking smaller step
            stuck_count += 1
            multiplier *= 0.5
            
        # print("Current cost = ", current_cost)
        if (iteration) % 100 == 0:
            print("Current objective = ", current_objective, "multiplier=", multiplier, "stuck=", stuck_count)
            print("\tparams = ", current_params)

        if stuck_count > 20:
            print("Too long without improvement")
            break

    return current_params

def get_o3d_geometries(beam_problem, params):
    length = beam_problem['length']
    magnus_segments = beam_problem['magnus_segments']
    initial_frame = beam_problem['initial_frame']

    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)

    frames = get_frames(magnus_segments, length, initial_frame, params)
    frames_viz = []
    for frame in frames:
        frame_viz = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
        frame_viz.transform(frame)
        frames_viz.append(frame_viz)

    cylinders = []
    for frame in subdivide_frames(magnus_segments, length, initial_frame, params, length/20):
        c = o3d.geometry.TriangleMesh.create_coordinate_frame(0.025)
        # c = o3d.geometry.TriangleMesh.create_cylinder(radius=0.06, height=length/20*1.1)
        # c.rotate(np.array([
        #     0, 0, 1,
        #     1, 0, 0,
        #     0, 1, 0
        # ]).reshape((3,3), order='C'))
        # c.compute_triangle_normals()
        # c.compute_vertex_normals()
        c.transform(frame)
        cylinders.append(c)

    return frames_viz + [origin_frame] + cylinders

def get_frames(magnus_segments, length, initial_frame, qs):
    ts = []
    l = 0
    T = initial_frame
    ts.append(T)
    # print("l=", l, "Txyz", T[:3,3].T)
    for i in range(magnus_segments):
        Psi = get_psi(length, qs, l, length/magnus_segments)
        T = T @ se3_exp(Psi)
        ts.append(T)
        l += length/magnus_segments
        # print("l=", l, "Txyz", T[:3,3].T)
    return ts

def subdivide_frames(magnus_segments, length, initial_frame, qs, delta_l):
    current_l = 0

    current_frame = initial_frame
    yield current_frame
    # print("current_l", current_l, "xyz", current_frame[:3,3].T)

    prev_changepoint = 0
    next_changepoint = length/magnus_segments
    changepoint_frame = initial_frame

    while current_l < length:
        # draw the segment from current_l to current_l + delta_l
        next_l = min(current_l + delta_l, next_changepoint)
        T = changepoint_frame @ se3_exp(
            get_psi(length, qs, prev_changepoint, next_l - prev_changepoint))
        # print("current_l", next_l, "xyz", T[:3,3].T)
        yield T

        if next_l == next_changepoint:
            prev_changepoint = next_changepoint
            next_changepoint = min(next_changepoint + length/magnus_segments, length)
            changepoint_frame = T

        current_l = next_l

def F(magnus_segments, length, alpha_x, alpha_y, alpha_z, initial_frame, theta, qs, lambd):
    psis, dpsi_dqs = get_psi_segments(length, qs, magnus_segments)
    endpoint, dendpoint_dq, d2endpoint_dq2 = get_endpoint(psis, dpsi_dqs, compute_hess = True)

    eps = 1e-6
    dqs = np.random.uniform(-1, 1, 3*POLY_ORDER)
    psis_plus, dpsi_dqs_plus = get_psi_segments(length, qs + dqs*eps, magnus_segments)
    endpoint_plus, dendpoint_dq_plus = get_endpoint(psis_plus, dpsi_dqs_plus, compute_hess = False)

    # # ##### SANITY

    # def get(dqs):
    #     psis, dpsi_dqs = get_psi_segments(length, qs + dqs, magnus_segments)
    #     endpoint, dendpoint_dq = get_endpoint(psis, dpsi_dqs, compute_hess = False)
    #     return endpoint, dendpoint_dq

    # # psis_plus, dpsi_dqs_plus = get_psi_segments(length, qs + dqs*eps, magnus_segments)
    # # psis_plus_half, dpsi_dqs_plus_half = get_psi_segments(length, qs + dqs*eps/2, magnus_segments)
    # # psis_minus, dpsi_dqs_minus = get_psi_segments(length, qs - dqs*eps, magnus_segments)
    # # psis_minus_half, dpsi_dqs_minus_half = get_psi_segments(length, qs - dqs*eps/2, magnus_segments)

    # endpoint, dendpoint_dq = get(dqs*0)
    # endpoint_plus, dendpoint_dq_plus = get(dqs*eps)
    # endpoint_minus, dendpoint_dq_minus = get(-dqs*eps)
    
    # # analytic first derivatives, numeric second derivatives
    # dendpoint_dt_plus = endpoint_plus@se3_to_matrix(dendpoint_dq_plus@dqs.reshape((-1,1)))
    # dendpoint_dt = endpoint@se3_to_matrix(dendpoint_dq@dqs.reshape((-1,1)))
    # d2endpoint_dt2_n = (dendpoint_dt_plus - dendpoint_dt)/eps
    # print("d2endpoint_dt2_n", d2endpoint_dt2_n)

    # # central difference
    # d2endpoint_dt2_cd = (endpoint_plus - 2*endpoint + endpoint_minus)/eps**2
    # print("d2endpoint_dt2_cd", d2endpoint_dt2_cd)

    # analytic second derivative
    # T'         =   T @ M L dq
    # T''        =   T'@ M L dq + T @ d/dt[M L dq]
    # print("dendpoint_dq shape", dendpoint_dq.shape)
    # print("d2endpoint_dq2 shape", d2endpoint_dq2.shape)
    # print("dendpoint_dq(t)", dendpoint_dq)

    # print("dendpoint_dq'(t) numeric", (dendpoint_dq_plus - dendpoint_dq)/eps)
    # print("dendpoint_dq'(t) analytic", np.einsum("ijk,k", d2endpoint_dq2, dqs))

    # d__dendpoint_dq__t = np.einsum("ijk,k", d2endpoint_dq2, dqs) @ dqs.reshape((-1,1))
    # d__dendpoint_dq__t_n = (dendpoint_dq_plus@dqs.reshape((-1,1)) - dendpoint_dq@dqs.reshape((-1,1)))/eps
    # print("d__dendpoint_dq__t", d__dendpoint_dq__t)
    # print("d__dendpoint_dq__t_n", d__dendpoint_dq__t_n)
    # Tprime = endpoint@se3_to_matrix(dendpoint_dq@dqs.reshape((-1,1)))

    # Tprimeprime = endpoint@(se3_to_matrix(dendpoint_dq@dqs.reshape((-1,1))) @ se3_to_matrix(dendpoint_dq@dqs.reshape((-1,1))) + \
    #                         se3_to_matrix(np.einsum("ijk,k", d2endpoint_dq2, dqs) @ dqs.reshape((-1,1))))


    K = get_energy_matrix(length)

    angle = np.pi/8
    c = np.cos(angle)
    s = np.sin(angle)
    base_frame = np.array([
        c, -s, 0, 0,
        s, c, 0, 0.1,
        0, 0, 1, 0,
        0, 0, 0, 1]).reshape((4,4), order='C')
    rotator = se3_exp(np.array([theta, 0, 0, 0, 0, 0]).reshape((-1,1)))

    initial_frame = base_frame @ rotator
    T = initial_frame @ endpoint
    T_plus = initial_frame @ endpoint_plus

    dT_dq = np.zeros((4,4,3*POLY_ORDER))
    for k in range(3*POLY_ORDER):
        dT_dq[:,:,k] = T @ se3_to_matrix(dendpoint_dq[:,k])

    dT_dq_plus = np.zeros((4,4,3*POLY_ORDER))
    for k in range(3*POLY_ORDER):
        dT_dq_plus[:,:,k] = T_plus @ se3_to_matrix(dendpoint_dq_plus[:,k])

    # print("SANITY")
    # print("dendpoint_dt n", (endpoint_plus - endpoint)/eps)
    # print("dendpoint_dt a", endpoint @ se3_to_matrix(dendpoint_dq @ dqs))
    # print("dT_dt n", initial_frame @ (endpoint_plus - endpoint)/eps)
    # print("dT_dt n alt", (T_plus - T)/eps)
    # print("dT_dt a", T @ se3_to_matrix(dendpoint_dq @ dqs))
    # print("dT_dt alt", np.einsum("ijk,k", dT_dq, dqs))

    d2T_dq2 = np.zeros((4,4,3*POLY_ORDER,3*POLY_ORDER))
    for k1 in range(3*POLY_ORDER):
        for k2 in range(3*POLY_ORDER):
            d2T_dq2[:,:,k1,k2] = T @ (
                se3_to_matrix(dendpoint_dq[:,k2].reshape((-1,1))) @ se3_to_matrix(dendpoint_dq[:,k1].reshape((-1,1))) + \
                se3_to_matrix(d2endpoint_dq2[:,k2,k1].reshape((-1,1))))

    dT_dt_plus = dT_dq_plus @ dqs
    # print("dT_dt_plus", dT_dt_plus)
    dT_dt = dT_dq @ dqs
    # print("d2T_dt2 n", (dT_dt_plus - dT_dt)/eps)
    # print("d2T_dt2 a", (d2T_dq2 @ dqs) @ dqs)

    # SE3_times_endpoint = SE3_times(endpoint)
    # dendpoint_dq = SE3_times_endpoint @ dtwistendpoint_dq
    # d2endpoint_dq2 = np.einsum("ij, jkl", SE3_times_endpoint, d2twistendpoint_dq2)

    roll = T[1,2]
    height = T[1,3]

    roll_plus = T_plus[1,2]
    height_plus = T_plus[1,3]

    droll_dq = dT_dq[1,2,:]
    dheight_dq = dT_dq[1,3,:]

    droll_dq_plus = dT_dq_plus[1,2,:]
    dheight_dq_plus = dT_dq_plus[1,3,:]

    # print("droll_dt a", droll_dq @ dqs)
    # print("dheight_dt a", dheight_dq @ dqs)
    # print("droll_dt n", (T_plus[1,2] - roll)/eps)
    # print("dheight_dt n", (T_plus[1,3] - height)/eps)

    d2roll_dq2 = d2T_dq2[1,2,:,:]
    d2height_dq2 = d2T_dq2[1,3,:,:]
    # print("d2height_dq2", d2height_dq2)

    # print("droll_dt_plus", droll_dq_plus @ dqs)
    # print("dheight_dt_plus", dheight_dq_plus @ dqs)

    # print("d2roll_dt2 n", (droll_dq_plus @ dqs - droll_dq @ dqs)/eps)
    # print("d2roll_dt2 a", (d2roll_dq2 @ dqs) @ dqs)
    # print("d2height_dt2 n", (dheight_dq_plus @ dqs - dheight_dq @ dqs)/eps)
    # print("d2height_dt2 a", (d2height_dq2 @ dqs) @ dqs)

    qxs = qs[:POLY_ORDER].reshape((-1,1))
    qys = qs[POLY_ORDER:2*POLY_ORDER].reshape((-1,1))
    qzs = qs[-POLY_ORDER:].reshape((-1,1))

    dqxs = dqs[:POLY_ORDER].reshape((-1,1))
    dqys = dqs[POLY_ORDER:2*POLY_ORDER].reshape((-1,1))
    dqzs = dqs[-POLY_ORDER:].reshape((-1,1))

    objective = alpha_x * qxs.T @ K @ qxs + \
        alpha_y * qys.T @ K @ qys  + \
        alpha_z * qzs.T @ K @ qzs

    objective_plus = alpha_x * (qxs + eps*dqxs).T @ K @ (qxs + eps*dqxs) + \
        alpha_y * (qys + eps*dqys).T @ K @ (qys + eps*dqys)  + \
        alpha_z * (qzs + eps*dqzs).T @ K @ (qzs + eps*dqzs)

    objective = objective.item()
    objective_plus = objective_plus.item()

    dE_dqxs = 2*(alpha_x * qxs.T @ K).flatten()
    dE_dqys = 2*(alpha_y * qys.T @ K).flatten() 
    dE_dqzs = 2*(alpha_z * qzs.T @ K).flatten() 
    dE_dqs = np.concatenate((dE_dqxs, dE_dqys, dE_dqzs))

    

    print("dE_dt numeric", (objective_plus - objective)/eps)
    print("dE_dt a", dE_dqs @ dqs)

    dE_dqxs_plus = 2*(alpha_x * (qxs+ dqxs*eps).T @ K).flatten()
    dE_dqys_plus = 2*(alpha_y * (qys+ dqys*eps).T @ K).flatten() 
    dE_dqzs_plus = 2*(alpha_z * (qzs+ dqzs*eps).T @ K).flatten() 
    dE_dqs_plus = np.concatenate((dE_dqxs_plus, dE_dqys_plus, dE_dqzs_plus))

    print("d2E_dt2 n", (dE_dqs_plus @ dqs - dE_dqs @ dqs)/eps)


    objective_hess = np.zeros((3*POLY_ORDER, 3*POLY_ORDER))
    objective_hess[:POLY_ORDER,:POLY_ORDER] = 2*alpha_x*K
    objective_hess[POLY_ORDER:2*POLY_ORDER,POLY_ORDER:2*POLY_ORDER] = 2*alpha_y*K
    objective_hess[-POLY_ORDER:,-POLY_ORDER:] = 2*alpha_z*K

    print("d2E_dt2 a", (objective_hess @ dqs) @ dqs)

    G = np.zeros((2 + 3*POLY_ORDER))
    G[0] = roll
    G[1] = height
    G[2:] = dE_dqs + droll_dq*lambd[0] + dheight_dq*lambd[1]

    G_plus = np.zeros((2 + 3*POLY_ORDER))
    G_plus[0] = roll_plus
    G_plus[1] = height_plus
    G_plus[2:] = dE_dqs_plus + droll_dq_plus*lambd[0] + dheight_dq_plus*lambd[1]

    print("dG_dt n", (G_plus - G)/eps)

    H = np.zeros((2 + 3*POLY_ORDER, 2 + 3*POLY_ORDER))
    H[0, :3*POLY_ORDER] = droll_dq
    H[1, :3*POLY_ORDER] = dheight_dq
    H[2:, -2] = droll_dq.T
    H[2:, -1] = dheight_dq.T
    H[2:,:-2] = objective_hess + d2roll_dq2 * lambd[0] + d2height_dq2 * lambd[1]

    print("dG_dt a", (H @ np.concatenate((dqs, [0,0]))))
    return G, H

def make_beam_problem(magnus_segments, length, alpha_x, alpha_y, alpha_z, initial_frame):
    K = get_energy_matrix(length)
    initial_frame = initial_frame.copy()

    # LRU cache is a hack to avoid double computation
    # during optimization, since the scipy SLSQP
    # interface does not let you supply value and jacobian
    # at the same time
    @functools.lru_cache
    def get_endpoint_cached(qs):
        psis, dpsi_dqs = get_psi_segments(length, qs, magnus_segments)
        return get_endpoint(psis, dpsi_dqs)
    
    @functools.lru_cache
    def constraint_and_jac(qs):
        # when the constraint value = 0
        # the terminal frame has 0-roll and 0-height
        endpoint, dendpoint_dq = get_endpoint_cached(qs)
        endpoint = initial_frame @ endpoint
        roll = endpoint[1,2]
        height = endpoint[1,3]

        dendpoint_dq = SE3_times(endpoint) @ dendpoint_dq
        droll_dq = dendpoint_dq[R_12,:]
        dheight_dq = dendpoint_dq[P_1,:]
        return np.array([roll, height]), np.array([droll_dq, dheight_dq])

    def dconstraint_dtheta(qs):
        endpoint, _ = get_endpoint_cached(tuple(qs))
        dendpoint_dtheta = (initial_frame @ endpoint) @ se3_to_matrix(get_dendpoint_dtheta(endpoint))# SE3_adj(SE3_inv(endpoint)) @ np.array([[1, 0, 0, 0, 0, 0]]).T
        droll_dtheta = dendpoint_dtheta[1,2]
        dheight_dtheta = dendpoint_dtheta[1,3]
        return np.array([droll_dtheta, dheight_dtheta])

    def constraint(qs):
        # tuple is used to make qs hashable
        c, j = constraint_and_jac(tuple(qs))
        return c

    def constraint_jac(qs):
        # tuple is used to make qs hashable
        c, j = constraint_and_jac(tuple(qs))
        return j

    def objective(qs):
        # objective(qs) = qxs.T alpha_x*K qxs + qys.T alpha_y*K qys + qzs.T alpha_z*K qys
        #
        # objective(qs + dqs) = (qxs + dqxs).T alpha_x * K * (qxs + dqxs) + ...
        #                     
        #                     = objective(qs) + 2 * qxs.T alpha_x * K * dqxs  + ...
        #                     
        # dobjective_dqs(qs)  =  2 * qxs.T alpha_x * K + 2 * qys.T alpha_y * K + 2 * qzs.T alpha_z * K

        qxs = qs[:POLY_ORDER].reshape((-1,1))
        qys = qs[POLY_ORDER:2*POLY_ORDER].reshape((-1,1))
        qzs = qs[-POLY_ORDER:].reshape((-1,1))

        objective = alpha_x * qxs.T @ K @ qxs + \
            alpha_y * qys.T @ K @ qys  + \
            alpha_z * qzs.T @ K @ qzs

        objective = objective.item()

        dE_dqxs = 2*(alpha_x * qxs.T @ K).flatten()
        dE_dqys = 2*(alpha_y * qys.T @ K).flatten() 
        dE_dqzs = 2*(alpha_z * qzs.T @ K).flatten() 
        dE_dqs = np.concatenate((dE_dqxs, dE_dqys, dE_dqzs))

        # print("dE_dqs", dE_dqs.shape)

        dobjective_dq = dE_dqs

        return objective, dobjective_dq

    def objective_hess(qs=None):
        hess = np.zeros((3*POLY_ORDER, 3*POLY_ORDER))
        hess[:POLY_ORDER,:POLY_ORDER] = alpha_x*K
        hess[POLY_ORDER:2*POLY_ORDER,POLY_ORDER:2*POLY_ORDER] = alpha_y*K
        hess[-POLY_ORDER:,-POLY_ORDER:] = alpha_z*K
        return hess

    def get_final_frame(qs):
        endpoint, _ = get_endpoint_cached(tuple(qs))
        return initial_frame @ endpoint

    return {
        'objective': objective,
        'constraint': constraint,
        'constraint_jac': constraint_jac,
        'dconstraint_dtheta': dconstraint_dtheta,
        'length': length,
        'initial_frame': initial_frame,
        'magnus_segments': magnus_segments,
        'get_final_frame': get_final_frame,
        'objective_hess': objective_hess,
    }

def solve_beam_problem(beam_problem, initial_guess):
    return sp.optimize.minimize(beam_problem['objective'], initial_guess, jac=True,
                                constraints={'type': 'eq',
                                             'fun':beam_problem['constraint'],
                                             'jac':beam_problem['constraint_jac'],
                                             # 'jac':None
                                             },
                                options={'maxiter':1000})



if __name__ == "__main__":
    np.random.seed(3)
    np.set_printoptions(precision=8, suppress=False)
    
    length = 0.34
    # soln = np.zeros(POLY_ORDER*3)
    qs = np.random.uniform(-1, 1, POLY_ORDER*3) 
    dqs = np.random.uniform(-1, 1, POLY_ORDER*3) 

    lambd = np.random.uniform(-1, 1, 2) 
    dlambd = np.random.uniform(-1, 1, 2) * 0

    theta = 0.0
    eps = 1e-6

    G, H = F(MAGNUS_SEGMENTS, length, alpha_x, alpha_y, alpha_z, initial_frame, theta, qs, lambd)
    G_plus, _ = F(MAGNUS_SEGMENTS, length, alpha_x, alpha_y, alpha_z, initial_frame, theta, qs + eps*dqs, lambd + eps*dlambd)

    print("G", G)
    print("G_plus", G_plus)
    print("dG/dt n", (G_plus - G)/eps)

    daux = np.concatenate((dqs, dlambd)).reshape(-1,1)
    dG_dt = H @ daux
    print("dG/dt a", dG_dt.flatten())
    print("H", H)


    # beam_problem = make_beam_problem(MAGNUS_SEGMENTS, length, alpha_x, alpha_y, alpha_z, initial_frame)
    # print("SVD")
    # U, S, V = sp.linalg.svd(beam_problem['objective_hess']())
    # print("S", S)
    # print("U", U.shape)
    # print("V", V.shape)
    # # print("absmax", np.max(np.abs(U.T - V)))

    # soln[POLY_ORDER*2-1] = -1 # curving into +z axis

    # solns = []

    # # thetas = np.concatenate([np.deg2rad(np.array([60, 30, 0, -30, -34])), np.linspace(np.deg2rad(-35), np.deg2rad(-50), 30)])
    # thetas = np.concatenate([np.deg2rad(np.array([60, 30, 0, -25])), np.linspace(np.deg2rad(-26), np.deg2rad(-40), 30)])

    # dE_dthetas = []
    # dE_dthetas_numeric = []
    # Es = []

    # last_E = None
    # last_theta = None
    # dg_dthetas = []
    # lams = []
    # singular_values = []

    # for theta in thetas:
    #     print("Theta = ", np.rad2deg(theta), " deg")
    #     rotator = se3_exp(np.array([theta, 0, 0, 0, 0, 0]).reshape((-1,1)))
    #     beam_problem = make_beam_problem(MAGNUS_SEGMENTS, length, alpha_x, alpha_y, alpha_z, initial_frame @ rotator)
    #     start = time.monotonic()
    #     result = solve_beam_problem(beam_problem, soln)
    #     if not result.success:
    #         raise RuntimeError("Uh Oh!")
    #     end = time.monotonic()
    #     print("Optimization time", end-start)
    #     print(result)
    #     soln = result.x
    #     print(soln)
    #     print("Final constrant", beam_problem['constraint'](soln))

    #     dg_dw = beam_problem['constraint_jac'](soln)
    #     _, dE_dw = beam_problem['objective'](soln)
    #     dg_dtheta = beam_problem['dconstraint_dtheta'](soln)
    #     dg_dthetas.append(dg_dtheta.flatten())

    #     lam = sp.linalg.solve(dg_dw @ dg_dw.T, dg_dw @ dE_dw.T)
    #     lams.append(lam.flatten())

    #     dE_dtheta = (-lam.T @ dg_dtheta).item() * -1 # minus 1 because we are going backwards
    #     dE_dthetas.append(dE_dtheta)
    #     print("dg_dtheta", dg_dtheta.flatten())
    #     print("dE_dtheta", dE_dtheta)

    #     P = np.empty((3*POLY_ORDER, 3))
    #     P[:,0] = dE_dw
    #     P[:,1:] = dg_dw.T

    #     U, S, V = sp.linalg.svd(P)
    #     singular_values.append(S)
    #     # print("P", P)
    #     # print("dE_dw", dE_dw)
    #     # print("dg_dw", dg_dw)

    #     E = result.fun
    #     Es.append(E)

    #     if last_E is not None:
    #         # print("E", E)
    #         # print("last_E", last_E)
    #         dE_numeric = E - last_E
    #         dE_dtheta_numeric = dE_numeric/(theta - last_theta) *-1
    #         print("dE_dtheta_numeric", dE_dtheta_numeric)
    #         dE_dthetas_numeric.append(dE_dtheta_numeric)
    #     else:
    #         # some arbitrary seed
    #         dE_dthetas_numeric.append(dE_dtheta)

    #     last_E = E
    #     last_theta = theta
        
    #     solns.append((theta, beam_problem, soln.copy()))

    # singular_values = np.log(np.array(singular_values))
    # plt.plot(np.rad2deg(thetas), singular_values[:,0])
    # plt.plot(np.rad2deg(thetas), singular_values[:,1])
    # plt.plot(np.rad2deg(thetas), singular_values[:,2])
    # plt.show()

    # fig, axs = plt.subplots(2)
    # i = 0
    # for _, _, soln in solns:
    #     i += 1
    #     ws = np.linspace(0, length, 100)
    #     cs = soln
    #     qx = cs[:POLY_ORDER]
    #     qy = cs[POLY_ORDER:2*POLY_ORDER]
    #     qz = cs[-POLY_ORDER:]

    #     if (qy[0] > 7):
    #         break

        
    #     axs[0].plot(ws, sum(qx[i]*ws**i for i in range(POLY_ORDER)), '--', c='red', alpha=0.1+0.9*i/len(solns), linewidth=0.5)
    #     axs[0].plot(ws, sum(qy[i]*ws**i for i in range(POLY_ORDER)), '--', c='green', alpha=0.1+0.9*i/len(solns), linewidth=0.5)
    #     axs[0].plot(ws, sum(qz[i]*ws**i for i in range(POLY_ORDER)), '--', c='blue', alpha=0.1+0.9*i/len(solns), linewidth=0.5)

    #     # derivatives
    #     axs[1].plot(ws, sum(i*qx[i]*ws**(i-1) for i in range(1,POLY_ORDER)), '--', c='red', alpha=0.1+0.9*i/len(solns), linewidth=0.5)
    #     axs[1].plot(ws, sum(i*qy[i]*ws**(i-1) for i in range(1,POLY_ORDER)), '--', c='green', alpha=0.1+0.9*i/len(solns), linewidth=0.5)
    #     axs[1].plot(ws, sum(i*qz[i]*ws**(i-1) for i in range(1,POLY_ORDER)), '--', c='blue', alpha=0.1+0.9*i/len(solns), linewidth=0.5)

    # # plot the last convex curves
    # _, _, soln = solns[i-2]
    # ws = np.linspace(0, length, 100)
    # cs = soln
    # qx = cs[:POLY_ORDER]
    # qy = cs[POLY_ORDER:2*POLY_ORDER]
    # qz = cs[-POLY_ORDER:]

    # axs[0].plot(ws, sum(qx[i]*ws**i for i in range(POLY_ORDER)), c='red', alpha=0.1+0.9*i/len(solns), linewidth=1.0)
    # axs[0].plot(ws, sum(qy[i]*ws**i for i in range(POLY_ORDER)), c='green', alpha=0.1+0.9*i/len(solns), linewidth=1.0)
    # axs[0].plot(ws, sum(qz[i]*ws**i for i in range(POLY_ORDER)), c='blue', alpha=0.1+0.9*i/len(solns), linewidth=1.0)

    # # derivatives
    # axs[1].plot(ws, sum(i*qx[i]*ws**(i-1) for i in range(1,POLY_ORDER)), c='red', alpha=0.1+0.9*i/len(solns), linewidth=1.0)
    # axs[1].plot(ws, sum(i*qy[i]*ws**(i-1) for i in range(1,POLY_ORDER)), c='green', alpha=0.1+0.9*i/len(solns), linewidth=1.0)
    # axs[1].plot(ws, sum(i*qz[i]*ws**(i-1) for i in range(1,POLY_ORDER)), c='blue', alpha=0.1+0.9*i/len(solns), linewidth=1.0)
    
    # plt.show()

    # lams = np.array(lams)        

    # plt.plot(np.rad2deg(thetas), dE_dthetas)
    # # plt.plot(np.rad2deg(thetas), dE_dthetas_numeric, '--')
    # plt.plot(np.rad2deg(thetas), Es)
    # plt.plot(np.rad2deg(thetas), lams[:,0])
    # plt.plot(np.rad2deg(thetas), lams[:,1])

    # plt.legend(["dE/dtheta", "E", "lam0", "lam1"])
    # plt.xlim([-50, -30])
    # # plt.ylim([-5000, 5000])
    # plt.show()

    # lams /= 1000
    # dg_dthetas = np.array(dg_dthetas)
    # plt.plot(lams[:,0], lams[:,1])
    # plt.plot(dg_dthetas[:,0], dg_dthetas[:,1])
    # plt.legend(["lam", "dg/dtheta"])
    # plt.show()


    # vis = o3d.visualization.Visualizer()
    # vis.create_window()


    # dE_dthetas = []
    # thetas = []

    # last_E = None
    # last_theta = None

    # for theta, beam_problem, soln in solns:
    #     print("theta", np.rad2deg(theta))

    #     dg_dw = beam_problem['constraint_jac'](soln)
    #     _, dE_dw = beam_problem['objective'](soln)
    #     dg_dtheta = beam_problem['dconstraint_dtheta'](soln)

    #     lam = sp.linalg.solve(dg_dw @ dg_dw.T, dg_dw @ dE_dw.T)
    #     dE_dtheta = (-lam.T @ dg_dtheta).item()
    #     dE_dthetas.append(dE_dtheta)
    #     thetas.append(theta)

    #     print("dE_dtheta", dE_dtheta)

    #     E, _ = beam_problem['objective'](soln)
    #     if last_E is not None:
    #         # print("E", E)
    #         # print("last_E", last_E)
    #         dE_numeric = E - last_E
    #         dE_dtheta_numeric = dE_numeric/(theta - last_theta)
    #         print("dE_dtheta_numeric", dE_dtheta_numeric)

    #     last_E = E
    #     last_theta = theta
        
    #     vis.clear_geometries()

    #     # geometry is the point cloud used in your animaiton
    #     for geo in get_o3d_geometries(beam_problem, soln):
    #         vis.add_geometry(geo)

    #     # now modify the points of your geometry
    #     # you can use whatever method suits you best, this is just an example
    #     vis.poll_events()
    #     vis.update_renderer()

    #     # plot first solution
    #     ws = np.linspace(0, length, 100)
    #     cs = soln
    #     qx = cs[:POLY_ORDER]
    #     qy = cs[POLY_ORDER:2*POLY_ORDER]
    #     qz = cs[-POLY_ORDER:]

    #     plt.title(f"Theta={np.rad2deg(theta)}")

    #     plt.plot(ws, sum(qx[i]*ws**i for i in range(POLY_ORDER)), c='red')
    #     plt.plot(ws, sum(qy[i]*ws**i for i in range(POLY_ORDER)), c='green')
    #     plt.plot(ws, sum(qz[i]*ws**i for i in range(POLY_ORDER)), c='blue')
    #     plt.legend(["x", "y", "z"])
    #     plt.show()


