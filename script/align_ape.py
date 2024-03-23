import numpy as np
from scipy.linalg import logm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

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
    quat = arr[4:]

    # 将四元数转换为旋转矩阵
    r = Rotation.from_quat(quat)
    rot_matrix = r.as_matrix()

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

def get_correspondence(est: np.ndarray, ref:np.ndarray, t_max_diff:float=0.01, t_offset:float=0.0):
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


ref_file_path = '/home/ubuntu/Projects/Livox_ws/src/mid360_locate/tum/localization_result_20240321.txt'
ref = read_tum(ref_file_path)

est_file_path = "/home/ubuntu/Projects/Livox_ws/src/mid360_locate/tum/test_slam.txt"
est = read_tum(est_file_path)

# t_offset = 0
# max_corr = -5000
# offset_setp = 0.1
# for n in range(-10, 10):
#     est_pair, ref_pair = get_correspondence(est, ref, 0.1, n * offset_setp)
#     est_relative, ref_relative = get_relative_correspondence(est_pair, ref_pair)

#     R_est_norm = []
#     R_ref_norm = []
#     for T_est, T_ref in zip(est_relative, ref_relative):
#         R_est_norm.append(np.linalg.norm(logm(T_est[:3,:3])))
#         R_ref_norm.append(np.linalg.norm(logm(T_ref[:3,:3])))

#     R_est_norm = np.array(R_est_norm)
#     R_ref_norm = np.array(R_ref_norm)

#     corr = np.corrcoef(R_est_norm, R_ref_norm)[0,1]
#     if corr > max_corr:
#         max_corr = corr
#         t_offset = n * offset_setp

# print(t_offset, max_corr)

t_offset = -0.5
est_pair, ref_pair = get_correspondence(est, ref, 0.1, t_offset)
est_relative, ref_relative = get_relative_correspondence(est_pair, ref_pair)

R_est_norm = []
R_ref_norm = []
theta_est = []
theta_ref = []
R_est = []
R_ref = []
for T_est, T_ref in zip(est_relative, ref_relative):
    theta_est.append(logm(T_est[:3,:3]))
    theta_ref.append(logm(T_ref[:3,:3]))
    R_est_norm.append(np.linalg.norm(logm(T_est[:3,:3])))
    R_ref_norm.append(np.linalg.norm(logm(T_ref[:3,:3])))
    R_est.append(T_est[:3,:3])
    R_ref.append(T_ref[:3,:3])

x = range(len(R_est_norm))
plt.plot(x, R_est_norm, color='red', label='est')
plt.plot(x, R_ref_norm, color='blue', label='ref')
plt.legend()
# plt.show()

theta_est = np.concatenate(theta_est, axis=1)
theta_ref = np.concatenate(theta_ref, axis=1)

Rbi, _, _ = umeyama_alignment(theta_est, theta_ref)

print(Rbi)
Rbi = Rotation.from_matrix(Rbi)

print()
print(est_relative[100][0:3,0:3])
print(ref_relative[100][0:3, 0:3])
print(Rbi.as_matrix() @ ref_relative[100][0:3, 0:3] @ Rbi.inv().as_matrix())
print(Rbi.as_euler('xyz'))

qwv_all = []
Rwb_all = []
Rvi_all = []
for i in range(est_pair.shape[1]):
    Rwb = Rotation.from_quat(est_pair[4:8,i])
    Rwb_all.append(Rwb.as_matrix())
    Rvi = Rotation.from_quat(ref_pair[4:8,i])
    Rvi_all.append(Rvi.as_matrix())
    qwv_all.append((Rwb * Rbi * Rvi.inv()).as_quat())
    if qwv_all[-1][-1] < 0:
        qwv_all[-1] *= -1
# qwv = np.stack(qwv)
qwv = np.mean(qwv_all, axis=0)
Rvw = Rotation.from_quat(qwv).inv().as_matrix()

b_all = []
for i in range(est_pair.shape[1]):
    b_all.append(Rvw @ est_pair[1:4,i] - ref_pair[1:4,i])
b = np.concatenate(b_all, axis=0)

I = np.tile(np.eye(3), (len(Rvi_all), 1))
A = np.concatenate(Rvi_all, axis=0)
AI = np.concatenate([A, I], axis=1)
x = np.linalg.inv(AI.T @ AI) @ AI.T @ b
print(Rvw @ x[3:])
print(Rotation.from_matrix(Rvw).as_quat())
print(qwv_all[0])