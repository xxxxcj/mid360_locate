import numpy as np
from scipy.linalg import logm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

def umeyama_alignment(x: np.ndarray, y: np.ndarray, with_scale: bool = False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        raise Exception("data matrices must have the same shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        raise Exception("Degenerate covariance rank, Umeyama alignment is not possible")

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c

def read_tum(file_path):
    poses = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split(' ')
            if len(data) == 8:  # Assuming TUM format with 8 columns
                pose = np.array([float(x) for x in data])
                poses.append(pose)
    poses = np.array(poses).T
    return poses

def binary_search(poses:np.ndarray, timestamps:float, t_max_diff:float=0.1, t_offset:float=0.0):
    left, right = 0, poses.shape[1]-1
    timestamps += t_offset
    while left <= right:
        mid = (left + right) // 2
        if poses[0, mid] > timestamps:
            right = mid - 1
        elif poses[0, mid] == timestamps:
            return mid
        else:
            left = mid + 1

    if right < 0 or left >= poses.shape[1] or \
        abs(timestamps - poses[0,right]) > t_max_diff or \
        abs(timestamps - poses[0,left]) > t_max_diff:
        return -1
    elif timestamps < poses[0,right]:
        return right
    elif timestamps > poses[0,left]:
        return left
    else:
        return right if timestamps - poses[0,right] <= poses[0,left] - timestamps else left

def tum2matrix(arr):
    # 提取位置和四元数
    pos = arr[1:4]
    pos[2] = 0
    quat = arr[4:]

    # 将四元数转换为旋转矩阵
    r = Rotation.from_quat(quat).as_euler('xyz')
    r[0] = 0
    r[1] = 0
    rot_matrix = Rotation.from_euler('xyz', r).as_matrix()

    # 构建变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rot_matrix
    transform_matrix[:3, 3] = pos

    return transform_matrix

def matrix2euler(R):
    # 先计算 pitch（绕 y 轴的旋转角度）和 roll（绕 x 轴的旋转角度）
    pitch = -np.arcsin(R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])

    # 计算 yaw（绕 z 轴的旋转角度）
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw])

def get_correspondence_tum(est: np.ndarray, ref:np.ndarray, t_max_diff:float=0.01, t_offset:float=0.0):
    est_idx, ref_idx = [], []
    for est_id, t in enumerate(est[0]):
        ref_id = binary_search(ref, t, t_max_diff, t_offset)
        if ref_id != -1:
            est_idx.append(est_id)
            ref_idx.append(ref_id)

    return est[:,est_idx], ref[:,ref_idx]

def get_relative_correspondence(est: np.ndarray, ref:np.ndarray, frequency:float=15, t_duration:float=1.0):
    step = int(t_duration * frequency)
    est_relative = []
    ref_relative = []
    T_est_last, T_ref_last = None, None
    for i in range(0, est.shape[1], step):
        T_est = tum2matrix(est[:,i])
        T_ref = tum2matrix(ref[:,i])
        if i != 0:
            est_relative.append(np.linalg.inv(T_est) @ T_est_last)
            ref_relative.append(np.linalg.inv(T_ref) @ T_ref_last)
        T_est_last, T_ref_last = T_est, T_ref
    return est_relative, ref_relative

def get_time_offset(est_tum_traj:np.ndarray, ref_tum_traj:np.ndarray, t_max_diff:float=0.1, offset_step:float=0.1, window:float=1):
    t_offset = 0
    max_corr = -5000
    for n in range(-int(window/2/offset_step), int(window/2/offset_step) + 1):
        est_tum, ref_tum = get_correspondence_tum(est_tum_traj, ref_tum_traj, t_max_diff, n * offset_step)
        est_relative, ref_relative = get_relative_correspondence(est_tum, ref_tum)

        R_est_norm = []
        R_ref_norm = []
        for T_est, T_ref in zip(est_relative, ref_relative):
            R_est_norm.append(np.linalg.norm(logm(T_est[:3,:3])))
            R_ref_norm.append(np.linalg.norm(logm(T_ref[:3,:3])))

        R_est_norm = np.array(R_est_norm)
        R_ref_norm = np.array(R_ref_norm)

        corr = np.corrcoef(R_est_norm, R_ref_norm)[0,1]
        if corr > max_corr:
            max_corr = corr
            t_offset = n * offset_step
    return t_offset, max_corr

def get_ext_and_coordinate_T(est_tum_traj:np.ndarray, ref_tum_traj:np.ndarray, t_max_diff:float=0.1, t_offset:float=0., max_iter:int=10, err_eps:float=1e-4):
    est_tum, ref_tum = get_correspondence_tum(est_tum_traj, ref_tum_traj, t_max_diff, t_offset)

    Test = []
    Tref = []
    for i in range(est_tum.shape[1]):
        Test.append(tum2matrix(est_tum[:,i]))
        Tref.append(tum2matrix(ref_tum[:,i]))

    T_w = np.eye(4)  # world coordinate Transformation
    T_e = np.eye(4)  # extrinsics
    t_err_list = []
    R_err_list = []

    t_err_mean_last = 5000.
    R_err_mean_last = 3.14
    for i in range(est_tum.shape[1]):
        t_err_list.append(np.linalg.norm(Test[i][0:3,3] - Tref[i][0:3,3]))
        R_err = Rotation.from_matrix(Test[i][0:3, 0:3]).as_euler('xyz') - Rotation.from_matrix(Tref[i][0:3, 0:3]).as_euler('xyz')
        R_err_list.append(np.linalg.norm(R_err))
    print(f"Initial Trans Error: {np.mean(t_err_list):.4f} [m]")
    print(f"Initial Angle Error: {np.mean(R_err_list)/np.pi * 180:.2f} [deg]")

    for n_iter in range(max_iter):
        t_est_np = []
        t_ref_np = []
        for i in range(est_tum.shape[1]):
            t_est_np.append(Test[i][0:3,3])
            t_ref_np.append(Tref[i][0:3,3])
        t_est_np = np.stack(t_est_np, axis=1)
        t_ref_np = np.stack(t_ref_np, axis=1)

        R, t, _ = umeyama_alignment(t_est_np, t_ref_np)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t
        T_w = T@T_w

        for i in range(est_tum.shape[1]):
            Test[i] = T@Test[i]

        Test_inv = []
        Tref_inv = []
        for i in range(est_tum.shape[1]):
            Test_inv.append(np.linalg.inv(Test[i]))
            Tref_inv.append(np.linalg.inv(Tref[i]))
            
        t_est_np = []
        t_ref_np = []
        for i in range(est_tum.shape[1]):
            t_est_np.append(Test_inv[i][0:3,3])
            t_ref_np.append(Tref_inv[i][0:3,3])
        t_est_np = np.stack(t_est_np, axis=1)
        t_ref_np = np.stack(t_ref_np, axis=1)

        R, t, _ = umeyama_alignment(t_ref_np, t_est_np)
        T2= np.eye(4)
        T2[:3,:3] = R
        T2[:3,3] = t

        # T2 = np.linalg.inv(T2)
        for i in range(est_tum.shape[1]):
            Test[i] = Test[i]@T2
        T_e = T_e @ T2

        t_err_list.clear()
        R_err_list.clear()
        for i in range(est_tum.shape[1]):
            t_err_list.append(np.linalg.norm(Test[i][0:3,3] - Tref[i][0:3,3]))
            R_err = Rotation.from_matrix(Test[i][0:3, 0:3]).as_euler('xyz') - Rotation.from_matrix(Tref[i][0:3, 0:3]).as_euler('xyz')
            R_err_list.append(np.linalg.norm(R_err))
        print(f"Iter: {n_iter:>3d} Distance: {np.mean(t_err_list):.4f}[m] Angle:{np.mean(R_err_list)/np.pi * 180:.2f}[deg]")

        if (abs(np.mean(t_err_list)-t_err_mean_last) < err_eps and abs(np.mean(R_err_list)-R_err_mean_last) < err_eps):
            break
        t_err_mean_last = np.mean(t_err_list)
        R_err_mean_last = np.mean(R_err_list)

    return T_w, T_e, np.mean(t_err_list), np.mean(R_err_list)/np.pi * 180

if __name__ == "__main__":

    ref_file_path = '/home/ubuntu/Projects/Livox_ws/src/mid360_locate/tum/localization_result_20240321.txt'
    ref_tum_traj = read_tum(ref_file_path)

    est_file_path = "/home/ubuntu/Projects/Livox_ws/src/mid360_locate/tum/test_slam.txt"
    est_tum_traj = read_tum(est_file_path)

    t_offset , max_corr = get_time_offset(est_tum_traj, ref_tum_traj, 0.1, 0.1, 2)

    print(t_offset, max_corr)

    T_w, T_e, t_err, R_err = get_ext_and_coordinate_T(est_tum_traj, ref_tum_traj, 0.1, t_offset, 100)

    print(T_w)
    print(T_e)
