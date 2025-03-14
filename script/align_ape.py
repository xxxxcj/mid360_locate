import os

import numpy as np
from scipy.linalg import logm
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
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
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

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
    with open(file_path, "r") as file:
        for line in file:
            data = line.strip().split(" ")
            if len(data) == 8:  # Assuming TUM format with 8 columns
                pose = np.array([float(x) for x in data])
                poses.append(pose)
    poses = np.array(poses).T
    return poses


def binary_search(
    poses: np.ndarray, timestamps: float, t_max_diff: float = 0.1, t_offset: float = 0.0
):
    left, right = 0, poses.shape[1] - 1
    timestamps += t_offset
    while left <= right:
        mid = (left + right) // 2
        if poses[0, mid] > timestamps:
            right = mid - 1
        elif poses[0, mid] == timestamps:
            return mid
        else:
            left = mid + 1

    if (
        right < 0
        or left >= poses.shape[1]
        or abs(timestamps - poses[0, right]) > t_max_diff
        or abs(timestamps - poses[0, left]) > t_max_diff
    ):
        return -1
    elif timestamps < poses[0, right]:
        return right
    elif timestamps > poses[0, left]:
        return left
    else:
        return (
            right
            if timestamps - poses[0, right] <= poses[0, left] - timestamps
            else left
        )


def tum2matrix(arr):
    # 提取位置和四元数
    pos = arr[1:4]
    # pos[2] = 0
    quat = arr[4:]

    # 将四元数转换为旋转矩阵
    r = Rotation.from_quat(quat).as_euler("xyz")
    # r[0] = 0
    # r[1] = 0
    rot_matrix = Rotation.from_euler("xyz", r).as_matrix()

    # 构建变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rot_matrix
    transform_matrix[:3, 3] = pos

    return transform_matrix

def timestamp_match(A, B, max_diff=0.1):
    """
    使用匈牙利算法找到两个时间戳序列的最优匹配对，使时间误差最小。
    
    参数:
    A: List[float]，第一个时间戳序列
    B: List[float]，第二个时间戳序列
    
    返回:
    matches: List[Tuple[float, float]]，最优匹配对
    """
    # 构造代价矩阵
    m, n = len(A), len(B)
    cost_matrix = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            error = abs(A[i] - B[j])
            cost_matrix[i][j] = error if error < max_diff else 2*max_diff

    # 匈牙利算法求解
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    valid_mask = (cost_matrix[row_ind, col_ind] < max_diff)
    # print(valid_mask)
    # print(cost_matrix[row_ind, col_ind] )
    # for i,j in zip(row_ind, col_ind):
    #     print(i,j,cost_matrix[i,j],abs(A[i] - B[j]))
    valid_rows = row_ind[valid_mask]
    valid_cols = col_ind[valid_mask]

    return valid_rows, valid_cols

def get_correspondence_tum(
    est: np.ndarray, ref: np.ndarray, t_max_diff: float = 0.01, t_offset: float = 0.0
):
    # est_idx, ref_idx = [], []
    # for est_id, t in enumerate(est[0]):
    #     ref_id = binary_search(ref, t, t_max_diff, t_offset)
    #     if ref_id != -1:
    #         est_idx.append(est_id)
    #         ref_idx.append(ref_id)

    est_idx, ref_idx = timestamp_match(est[0,:]+t_offset, ref[0,:], t_max_diff)

    return est[:, est_idx], ref[:, ref_idx]


def get_relative_correspondence(
    est: np.ndarray, ref: np.ndarray, frequency: float = 10, t_duration: float = 1.0
):
    step = int(t_duration * frequency)
    est_relative = []
    ref_relative = []
    T_est_last, T_ref_last = None, None
    for i in range(0, est.shape[1], step):
        T_est = tum2matrix(est[:, i])
        T_ref = tum2matrix(ref[:, i])
        if i != 0:
            est_relative.append(np.linalg.inv(T_est_last) @ T_est)
            ref_relative.append(np.linalg.inv(T_ref_last) @ T_ref)
        T_est_last, T_ref_last = T_est, T_ref
    return est_relative, ref_relative


def get_time_offset(
    est_tum_traj: np.ndarray,
    ref_tum_traj: np.ndarray,
    t_max_diff: float = 0.1,
    offset_step: float = 0.1,
    window: float = 1,
):
    t_offset = 0
    max_corr = -5000
    for n in range(-int(window / 2 / offset_step), int(window / 2 / offset_step) + 1):
        est_tum, ref_tum = get_correspondence_tum(
            est_tum_traj, ref_tum_traj, t_max_diff, n * offset_step
        )

        est_relative, ref_relative = get_relative_correspondence(est_tum, ref_tum)

        R_est_norm = []
        R_ref_norm = []
        for T_est, T_ref in zip(est_relative, ref_relative):
            R_est_norm.append(np.linalg.norm(logm(T_est[:3, :3])))
            R_ref_norm.append(np.linalg.norm(logm(T_ref[:3, :3])))

        R_est_norm = np.array(R_est_norm)
        R_ref_norm = np.array(R_ref_norm)

        corr = np.corrcoef(R_est_norm, R_ref_norm)[0, 1]
        # corr = np.linalg.norm(R_est_norm-R_ref_norm)
        if corr > max_corr:
            max_corr = corr
            t_offset = n * offset_step
        print(f"offset: {n * offset_step:.4}\t corr: {corr:.4}\t pair: {est_tum.shape[1]}")
    return t_offset, max_corr


def rotation_error_angle(R1, R2):
    # Compute relative rotation matrix
    R_error = np.dot(R1.T, R2)
    # Compute the trace of the relative rotation matrix
    trace = np.trace(R_error)
    # Compute the angle using arccos, with numerical stability
    x = (trace - 1) / 2
    x = np.clip(x, -1, 1)  # Ensure x is in [-1, 1]
    angle_rad = np.arccos(x)  # Angle in radians
    return angle_rad


def get_ext_and_coordinate_T(
    est_tum_traj: np.ndarray,
    ref_tum_traj: np.ndarray,
    t_max_diff: float = 0.1,
    t_offset: float = 0.0,
    max_iter: int = 10,
    err_eps: float = 1e-5,
    verbose=False
):
    est_tum, ref_tum = get_correspondence_tum(
        est_tum_traj, ref_tum_traj, t_max_diff, t_offset
    )

    Test = []
    Tref = []
    for i in range(est_tum.shape[1]):
        Test.append(tum2matrix(est_tum[:, i]))
        Tref.append(tum2matrix(ref_tum[:, i]))

    T_w = np.eye(4)  # world coordinate Transformation
    T_e = np.eye(4)  # extrinsics
    t_err_list = []
    R_err_list = []

    t_err_mean_last = 5000.0
    R_err_mean_last = 3.14
    for i in range(est_tum.shape[1]):
        t_err_list.append(np.linalg.norm(Test[i][0:3, 3] - Tref[i][0:3, 3]))
        R_err = Rotation.from_matrix(Test[i][0:3, 0:3]).as_euler(
            "xyz"
        ) - Rotation.from_matrix(Tref[i][0:3, 0:3]).as_euler("xyz")
        R_err_list.append(np.linalg.norm(R_err))

    if verbose:
        print(f"Find {est_tum.shape[1]} Pairs")
        print(f"Initial Trans Error: {np.mean(t_err_list):.4f} [m]")
        print(f"Initial Angle Error: {np.mean(R_err_list)/np.pi * 180:.2f} [deg]")

    for n_iter in range(max_iter):
        t_est_np = []
        t_ref_np = []
        for i in range(est_tum.shape[1]):
            t_est_np.append(Test[i][0:3, 3])
            t_ref_np.append(Tref[i][0:3, 3])
        t_est_np = np.stack(t_est_np, axis=1)
        t_ref_np = np.stack(t_ref_np, axis=1)

        R, t, _ = umeyama_alignment(t_est_np, t_ref_np)  # y=T@x
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        T_w = T @ T_w

        for i in range(est_tum.shape[1]):
            Test[i] = T @ Test[i]

        Test_inv = []
        Tref_inv = []
        for i in range(est_tum.shape[1]):
            Test_inv.append(np.linalg.inv(Test[i]))
            Tref_inv.append(np.linalg.inv(Tref[i]))

        t_est_np = []
        t_ref_np = []
        for i in range(est_tum.shape[1]):
            t_est_np.append(Test_inv[i][0:3, 3])
            t_ref_np.append(Tref_inv[i][0:3, 3])
        t_est_np = np.stack(t_est_np, axis=1)
        t_ref_np = np.stack(t_ref_np, axis=1)

        R, t, _ = umeyama_alignment(t_ref_np, t_est_np)
        T2 = np.eye(4)
        T2[:3, :3] = R
        T2[:3, 3] = t
        T2 = np.linalg.inv(T2)
        T_e = T_e @ T2

        for i in range(est_tum.shape[1]):
            Tref[i] = Tref[i] @ T2

        t_err_list.clear()
        R_err_list.clear()
        for i in range(est_tum.shape[1]):
            t_err_list.append(np.linalg.norm(Test[i][0:3, 3] - Tref[i][0:3, 3], ord=2))
            # R_err = Rotation.from_matrix(Test[i][0:3, 0:3]).as_euler("xyz") - \
            #     Rotation.from_matrix(Tref[i][0:3, 0:3]).as_euler("xyz")
            R_err_list.append(rotation_error_angle(Test[i][0:3, 0:3], Tref[i][0:3, 0:3]))
        
        if verbose:
            print(
                f"Iter: {n_iter:>3d} Distance: {np.mean(t_err_list):.4f}[m] Angle:{np.mean(R_err_list)/np.pi * 180:.4f}[deg]"
            )

        if abs(np.mean(t_err_list) - t_err_mean_last) < err_eps or \
            abs(np.mean(R_err_list) - R_err_mean_last) < err_eps:
            break
        t_err_mean_last = np.mean(t_err_list)
        R_err_mean_last = np.mean(R_err_list)

    return T_w, T_e, np.mean(t_err_list), np.mean(R_err_list) / np.pi * 180, est_tum.shape[1]


def transform_tum_poses(file_path, T_w, T_e, file_path_new):
    poses = []
    with open(file_path, "r") as file:
        for line in file:
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split()
                pose = np.array([float(x) for x in parts])
                poses.append((pose[0], tum2matrix(pose)))

    transformed_poses = []
    for pose in poses:
        timestamp = pose[0]
        transformed_pose = T_w @ pose[1] @ T_e
        transformed_poses.append((timestamp, transformed_pose))

    with open(file_path_new, "w") as file:
        for pose in transformed_poses:
            timestamp = pose[0]
            rotation_matrix = pose[1][:3, :3]
            t = pose[1][:3, 3]

            # 将旋转矩阵转换为四元数
            r = Rotation.from_matrix(rotation_matrix)
            q = r.as_quat()  # x, y, z, w
            file.write(
                "{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    timestamp, t[0], t[1], t[2], q[0], q[1], q[2], q[3]
                )
            )


if __name__ == "__main__":

    est_file_path = "/home/ubuntu/Projects/AVP-RGBD-Shared/src/localization/result/20241214-z03-camera-1-all.txt"
    # est_file_path = "/home/ubuntu/Projects/Livox_ws/src/mid360_locate/tum/20241214-z03-1-semantic.txt"
    est_tum_traj = read_tum(est_file_path)

    ref_file_path = "/home/ubuntu/Projects/Livox_ws/src/mid360_locate/tum/20241214_1.txt"
    ref_tum_traj = read_tum(ref_file_path)

    # t_offset , max_corr = get_time_offset(est_tum_traj, ref_tum_traj, 0.05, 0.01, 1)
    # print(t_offset, max_corr)


    duration = 0.2
    step = 0.005
    t_offset = 0.
    best_t_err = float('inf')
    best_R_err = 0
    T_w, T_e = None, None
    best_pair_num = 0
    for i in range(-int(duration/2/step), int(duration/2/step)):
        # t_offset = i*step  # .35  #.40
        # T_w @ T_est = T_ref @ T_e
        T_w_step, T_e_step, t_err, R_err, pairs_num = get_ext_and_coordinate_T(est_tum_traj, ref_tum_traj, 0.02, i*step, 100, verbose=False)
        print(f"offset: {i * step:.4}\t t_err: {t_err:.4}\t R_err: {R_err:.4f}\t Pairs:{pairs_num}")

        if t_err < best_t_err:
            best_t_err = t_err
            best_R_err = R_err
            t_offset = i*step
            T_w = T_w_step
            T_e = T_e_step
            best_pair_num = pairs_num
    
    print("offset", t_offset, "t_err: ", best_t_err, "R_err: ", best_R_err, "Pairs:", pairs_num)
    print(T_w)
    print(T_e)

    read_path = ref_file_path
    save_path = read_path.replace(".txt", "_transformed.txt")
    print(save_path)
    transform_tum_poses(read_path, np.linalg.inv(T_w), T_e, save_path)

    os.system(
        f"evo_ape tum {est_file_path} {save_path} -a --t_max_diff=0.02 --t_offset={-t_offset}  --plot_mode xy -r trans_part --save_results {est_file_path[:-4]}-trans.zip --save_plot {est_file_path[:-4]}-trans.png"
    )

    os.system(
        f"evo_ape tum {est_file_path} {save_path} -a --t_max_diff=0.02 --t_offset={-t_offset}  --plot_mode xy -r angle_deg --save_results {est_file_path[:-4]}-rot.zip --save_plot {est_file_path[:-4]}-rot.png"
    )

    # os.system(
    #     f"evo_ape tum {est_file_path} {save_path} --t_max_diff=0.05 --t_offset={-t_offset}  --plot_mode xy -r trans_part"
    # )

    # os.system(
    #     f"evo_ape tum {est_file_path} {save_path} --t_max_diff=0.05 --t_offset={-t_offset}  --plot_mode xy -r angle_deg"
    # )
