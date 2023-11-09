from matplotlib import pyplot as plt
import numpy as np
import cvxpy as cp
import scipy as sp
import open3d as o3d
import time
# from magnus_utils_small import *
# from magnus_utils_tiny import *
from magnus_utils import *
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
    np.random.seed(0)

    
    
    length = 0.34
    soln = np.zeros(POLY_ORDER*3)

    beam_problem = make_beam_problem(MAGNUS_SEGMENTS, length, alpha_x, alpha_y, alpha_z, initial_frame)
    print("SVD")
    U, S, V = sp.linalg.svd(beam_problem['objective_hess']())
    print("S", S)
    print("U", U.shape)
    print("V", V.shape)
    # print("absmax", np.max(np.abs(U.T - V)))

    soln[POLY_ORDER*2-1] = -1 # curving into +z axis

    solns = []

    # thetas = np.concatenate([np.deg2rad(np.array([60, 30, 0, -30, -34])), np.linspace(np.deg2rad(-35), np.deg2rad(-50), 30)])
    thetas = np.concatenate([np.deg2rad(np.array([60, 30, 0, -25])), np.linspace(np.deg2rad(-26), np.deg2rad(-40), 30)])

    dE_dthetas = []
    dE_dthetas_numeric = []
    Es = []

    last_E = None
    last_theta = None
    dg_dthetas = []
    lams = []
    singular_values = []

    for theta in thetas:
        print("Theta = ", np.rad2deg(theta), " deg")
        rotator = se3_exp(np.array([theta, 0, 0, 0, 0, 0]).reshape((-1,1)))
        beam_problem = make_beam_problem(MAGNUS_SEGMENTS, length, alpha_x, alpha_y, alpha_z, initial_frame @ rotator)
        start = time.monotonic()
        result = solve_beam_problem(beam_problem, soln)
        if not result.success:
            raise RuntimeError("Uh Oh!")
        end = time.monotonic()
        print("Optimization time", end-start)
        print(result)
        soln = result.x
        print(soln)
        print("Final constrant", beam_problem['constraint'](soln))

        dg_dw = beam_problem['constraint_jac'](soln)
        _, dE_dw = beam_problem['objective'](soln)
        dg_dtheta = beam_problem['dconstraint_dtheta'](soln)
        dg_dthetas.append(dg_dtheta.flatten())

        lam = sp.linalg.solve(dg_dw @ dg_dw.T, dg_dw @ dE_dw.T)
        lams.append(lam.flatten())

        dE_dtheta = (-lam.T @ dg_dtheta).item() * -1 # minus 1 because we are going backwards
        dE_dthetas.append(dE_dtheta)
        print("dg_dtheta", dg_dtheta.flatten())
        print("dE_dtheta", dE_dtheta)

        P = np.empty((3*POLY_ORDER, 3))
        P[:,0] = dE_dw
        P[:,1:] = dg_dw.T

        U, S, V = sp.linalg.svd(P)
        singular_values.append(S)
        # print("P", P)
        # print("dE_dw", dE_dw)
        # print("dg_dw", dg_dw)

        E = result.fun
        Es.append(E)

        if last_E is not None:
            # print("E", E)
            # print("last_E", last_E)
            dE_numeric = E - last_E
            dE_dtheta_numeric = dE_numeric/(theta - last_theta) *-1
            print("dE_dtheta_numeric", dE_dtheta_numeric)
            dE_dthetas_numeric.append(dE_dtheta_numeric)
        else:
            # some arbitrary seed
            dE_dthetas_numeric.append(dE_dtheta)

        last_E = E
        last_theta = theta
        
        solns.append((theta, beam_problem, soln.copy()))

    singular_values = np.log(np.array(singular_values))
    plt.plot(np.rad2deg(thetas), singular_values[:,0])
    plt.plot(np.rad2deg(thetas), singular_values[:,1])
    plt.plot(np.rad2deg(thetas), singular_values[:,2])
    plt.show()

    fig, axs = plt.subplots(2)
    i = 0
    for _, _, soln in solns:
        i += 1
        ws = np.linspace(0, length, 100)
        cs = soln
        qx = cs[:POLY_ORDER]
        qy = cs[POLY_ORDER:2*POLY_ORDER]
        qz = cs[-POLY_ORDER:]

        if (qy[0] > 7):
            break

        
        axs[0].plot(ws, sum(qx[i]*ws**i for i in range(POLY_ORDER)), '--', c='red', alpha=0.1+0.9*i/len(solns), linewidth=0.5)
        axs[0].plot(ws, sum(qy[i]*ws**i for i in range(POLY_ORDER)), '--', c='green', alpha=0.1+0.9*i/len(solns), linewidth=0.5)
        axs[0].plot(ws, sum(qz[i]*ws**i for i in range(POLY_ORDER)), '--', c='blue', alpha=0.1+0.9*i/len(solns), linewidth=0.5)

        # derivatives
        axs[1].plot(ws, sum(i*qx[i]*ws**(i-1) for i in range(1,POLY_ORDER)), '--', c='red', alpha=0.1+0.9*i/len(solns), linewidth=0.5)
        axs[1].plot(ws, sum(i*qy[i]*ws**(i-1) for i in range(1,POLY_ORDER)), '--', c='green', alpha=0.1+0.9*i/len(solns), linewidth=0.5)
        axs[1].plot(ws, sum(i*qz[i]*ws**(i-1) for i in range(1,POLY_ORDER)), '--', c='blue', alpha=0.1+0.9*i/len(solns), linewidth=0.5)

    # plot the last convex curves
    _, _, soln = solns[i-2]
    ws = np.linspace(0, length, 100)
    cs = soln
    qx = cs[:POLY_ORDER]
    qy = cs[POLY_ORDER:2*POLY_ORDER]
    qz = cs[-POLY_ORDER:]

    axs[0].plot(ws, sum(qx[i]*ws**i for i in range(POLY_ORDER)), c='red', alpha=0.1+0.9*i/len(solns), linewidth=1.0)
    axs[0].plot(ws, sum(qy[i]*ws**i for i in range(POLY_ORDER)), c='green', alpha=0.1+0.9*i/len(solns), linewidth=1.0)
    axs[0].plot(ws, sum(qz[i]*ws**i for i in range(POLY_ORDER)), c='blue', alpha=0.1+0.9*i/len(solns), linewidth=1.0)

    # derivatives
    axs[1].plot(ws, sum(i*qx[i]*ws**(i-1) for i in range(1,POLY_ORDER)), c='red', alpha=0.1+0.9*i/len(solns), linewidth=1.0)
    axs[1].plot(ws, sum(i*qy[i]*ws**(i-1) for i in range(1,POLY_ORDER)), c='green', alpha=0.1+0.9*i/len(solns), linewidth=1.0)
    axs[1].plot(ws, sum(i*qz[i]*ws**(i-1) for i in range(1,POLY_ORDER)), c='blue', alpha=0.1+0.9*i/len(solns), linewidth=1.0)
    
    plt.show()

    lams = np.array(lams)        

    plt.plot(np.rad2deg(thetas), dE_dthetas)
    # plt.plot(np.rad2deg(thetas), dE_dthetas_numeric, '--')
    plt.plot(np.rad2deg(thetas), Es)
    plt.plot(np.rad2deg(thetas), lams[:,0])
    plt.plot(np.rad2deg(thetas), lams[:,1])

    plt.legend(["dE/dtheta", "E", "lam0", "lam1"])
    plt.xlim([-50, -30])
    # plt.ylim([-5000, 5000])
    plt.show()

    lams /= 1000
    dg_dthetas = np.array(dg_dthetas)
    plt.plot(lams[:,0], lams[:,1])
    plt.plot(dg_dthetas[:,0], dg_dthetas[:,1])
    plt.legend(["lam", "dg/dtheta"])
    plt.show()


        

    # theta = np.deg2rad(60)
    # dtheta = 1e-4

    # beam_problem0 = make_beam_problem(
    #     MAGNUS_SEGMENTS, length, alpha_x, alpha_y, alpha_z, initial_frame @ se3_exp(np.array([theta, 0, 0, 0, 0, 0]).reshape((-1,1))))
    # soln0 = solve_beam_problem(beam_problem0, soln).x

    # beam_problem1 = make_beam_problem(
    #     MAGNUS_SEGMENTS, length, alpha_x, alpha_y, alpha_z, initial_frame @ se3_exp(np.array([theta + dtheta, 0, 0, 0, 0, 0]).reshape((-1,1))))

    # c0 = beam_problem0['constraint'](soln0)
    # c1 = beam_problem1['constraint'](soln0)

    # endpoint0 = beam_problem0['get_final_frame'](tuple(soln0))
    # endpoint1 = beam_problem1['get_final_frame'](tuple(soln0))

    # print("endpoint0\n", endpoint0)
    # print("endpoint1\n", endpoint1)

    # dendpoint_dtheta = get_dendpoint_dtheta(endpoint0)

    # dendpoint_dtheta_expected = (endpoint0 @ se3_exp(get_dendpoint_dtheta(SE3_inv(initial_frame) @ endpoint0) * dtheta) - endpoint0)/dtheta
    # dendpoint_dtheta_expected = (endpoint0 @ se3_to_matrix(get_dendpoint_dtheta(SE3_inv(initial_frame) @ endpoint0)))
    # dendpoint_dtheta_actual = (endpoint1 - endpoint0)/dtheta

    # print("move\n", dendpoint_dtheta_actual)
    # print("expected\n", dendpoint_dtheta_expected)

    # dconstraint_dtheta = (c1 - c0)/dtheta
    # dconstraint_dtheta_analytic = beam_problem0['dconstraint_dtheta'](soln0)
    # print("dconstraint_dtheta", dconstraint_dtheta)
    # print("dconstraint_dtheta analytic", dconstraint_dtheta_analytic)
    # print("ratio", dconstraint_dtheta / dconstraint_dtheta_analytic)

    # soln1 = solve_beam_problem(beam_problem1, soln0).x
    # dw = (soln1 - soln0).reshape((-1,1))
    # dconstraint_dw = beam_problem0['constraint_jac'](soln0)

    # print("dconstraint_dw", dconstraint_dw)
    # print("dw/dtheta", dw.T/dtheta)
    # print("dcons/dw * dw/dtheta", (dconstraint_dw @ dw/dtheta).T)
    # print("dcons/dtheta", dconstraint_dtheta)

    # Let g: R^n -> R^2 be the constraint function
    # Let w_eq(theta) be the equilibrium parameters, given boundary condition theta
    # 
    # dg/dw * dw_eq + dg/dtheta dtheta = 0 (w_eq must remain on constraint manifold as theta changes)
    # dg/dw * w_eq' = -dg/dtheta
    #
    # At optimality we have the lagrange condition
    # lam^T dg/dw = dE/dw. Then
    # 
    # lam^T dg/dw * w_eq'= -lam^T dg/dtheta
    # dE/dw * w_eq' = -lam^T dg/dtheta
    # dE/dtheta_eq = -lam^T dg/dtheta
    #
    # This is units of torque. When positive, it takes work
    # to move to an adjacent equilibrium point.
    # When >= 0, it takes no work to move the equilibrium.
    #
    # To solve for lam, we write
    # dg/dw^T lam = dE/dw^T
    # dg/dw dg/dw^T lam = dg/dw dE/dw^T
    # lam = (dg/dw dg/dw^T)^{-1} dg/dw dE/dw^T
    #

    vis = o3d.visualization.Visualizer()
    vis.create_window()



    dE_dthetas = []
    thetas = []

    last_E = None
    last_theta = None

    for theta, beam_problem, soln in solns:
        print("theta", np.rad2deg(theta))

        dg_dw = beam_problem['constraint_jac'](soln)
        _, dE_dw = beam_problem['objective'](soln)
        dg_dtheta = beam_problem['dconstraint_dtheta'](soln)

        lam = sp.linalg.solve(dg_dw @ dg_dw.T, dg_dw @ dE_dw.T)
        dE_dtheta = (-lam.T @ dg_dtheta).item()
        dE_dthetas.append(dE_dtheta)
        thetas.append(theta)

        print("dE_dtheta", dE_dtheta)

        E, _ = beam_problem['objective'](soln)
        if last_E is not None:
            # print("E", E)
            # print("last_E", last_E)
            dE_numeric = E - last_E
            dE_dtheta_numeric = dE_numeric/(theta - last_theta)
            print("dE_dtheta_numeric", dE_dtheta_numeric)

        last_E = E
        last_theta = theta
        
        vis.clear_geometries()

        # geometry is the point cloud used in your animaiton
        for geo in get_o3d_geometries(beam_problem, soln):
            vis.add_geometry(geo)

        # now modify the points of your geometry
        # you can use whatever method suits you best, this is just an example
        vis.poll_events()
        vis.update_renderer()

        # plot first solution
        ws = np.linspace(0, length, 100)
        cs = soln
        qx = cs[:POLY_ORDER]
        qy = cs[POLY_ORDER:2*POLY_ORDER]
        qz = cs[-POLY_ORDER:]

        plt.title(f"Theta={np.rad2deg(theta)}")

        plt.plot(ws, sum(qx[i]*ws**i for i in range(POLY_ORDER)), c='red')
        plt.plot(ws, sum(qy[i]*ws**i for i in range(POLY_ORDER)), c='green')
        plt.plot(ws, sum(qz[i]*ws**i for i in range(POLY_ORDER)), c='blue')
        plt.legend(["x", "y", "z"])
        plt.show()


