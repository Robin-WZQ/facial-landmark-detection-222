import cv2
import numpy as np

def getTform(landmark, target_size, warp_type = "DL_LANDMARK"):
    tmp_landmark = np.array(landmark).copy()
    tmp_landmark = tmp_landmark.reshape(-1, 2)
    tform = np.zeros((2, 3))
    
    template_std_points_save = np.array([
                    0.4125, 0.508125,
                    0.5875, 0.508125,
                    0.5, 0.6175,
                    0.4371, 0.7022,
                    0.5628, 0.7022])


    template_std_points_landmark = np.array([
            0.375     , 0.46875   ,
            0.625     , 0.46875   ,
            0.5       , 0.625     ,
            0.41015625, 0.74609375,
            0.58984375, 0.74609375])

    if warp_type == "DL_SAVE":
        std_points = template_std_points_save * target_size
    elif warp_type == "DL_LANDMARK":
        std_points = template_std_points_landmark * target_size

    tmp_landmark = tmp_landmark.reshape(-1)
    assert (tmp_landmark.shape[0] == std_points.shape[0]), "ERROR, Landmark shape != std_points shape !"
    if warp_type == "DL_VIDEO" or warp_type == "DL_VOX":
        tmp_landmark = tmp_landmark[:1690]
        std_points = std_points[:1690]
    points_num = int(std_points.shape[0] / 2)

    sum_x = 0
    sum_y = 0
    sum_u = 0
    sum_v = 0
    sum_xx_yy = 0
    sum_ux_vy = 0
    sum_vx_uy = 0

    for c in range(points_num):
        x_off = c * 2
        y_off = x_off + 1
        tmp_x = float(tmp_landmark[c * 2])
        tmp_y = float(tmp_landmark[c * 2 + 1])
        sum_x += tmp_x
        sum_y += tmp_y
        sum_u += std_points[x_off]
        sum_v += std_points[y_off]
        sum_xx_yy += tmp_x * tmp_x + \
                        tmp_y * tmp_y
        sum_ux_vy += tmp_x * std_points[x_off] + \
                        tmp_y * std_points[y_off]
        sum_vx_uy += std_points[y_off] * tmp_x - \
                        std_points[x_off] * tmp_y

    if sum_xx_yy < 1e-16:
        return None

    q = sum_u - sum_x * sum_ux_vy * 1. / sum_xx_yy \
                + sum_y * sum_vx_uy * 1. / sum_xx_yy
    p = sum_v - sum_y * sum_ux_vy * 1. / sum_xx_yy \
                - sum_x * sum_vx_uy * 1. / sum_xx_yy

    r = points_num - (sum_x * sum_x + sum_y * sum_y) * 1. / sum_xx_yy

    if r < 1e-16 and r > -1e-16:
        return None

    a = (sum_ux_vy - sum_x * q * 1. / r - sum_y * p * 1. / r) * 1. / sum_xx_yy
    b = (sum_vx_uy + sum_y * q * 1. / r - sum_x * p * 1. / r) * 1. / sum_xx_yy
    c = q * 1. / r
    d = p * 1. / r

    tform[0][0] = a
    tform[0][1] = -b
    tform[0][2] = c
    tform[1][0] = b
    tform[1][1] = a
    tform[1][2] = d

    return tform

def getTform_scale(landmark, target_size, warp_type = "DL_LANDMARK", scale=1):
    tmp_landmark = np.array(landmark).copy()
    tmp_landmark = tmp_landmark.reshape(-1, 2)
    tform = np.zeros((2, 3))
    
    template_std_points_save = np.array([
                    0.4125, 0.508125,
                    0.5875, 0.508125,
                    0.5, 0.6175,
                    0.4371, 0.7022,
                    0.5628, 0.7022])


    template_std_points_landmark = np.array([
            0.375     , 0.46875   ,
            0.625     , 0.46875   ,
            0.5       , 0.625     ,
            0.41015625, 0.74609375,
            0.58984375, 0.74609375])
    
    if warp_type == "DL_SAVE":
        std_points = template_std_points_save * target_size
    elif warp_type == "DL_LANDMARK":
        std_points = template_std_points_landmark * target_size
    
    if not scale == 1:
        std_points += (target_size*(scale - 1)/2.0)

    tmp_landmark = tmp_landmark.reshape(-1)
    assert (tmp_landmark.shape[0] == std_points.shape[0]), "ERROR, Landmark shape != std_points shape !"
    if warp_type == "DL_VIDEO" or warp_type == "DL_VOX":
        tmp_landmark = tmp_landmark[:1690]
        std_points = std_points[:1690]
    points_num = int(std_points.shape[0] / 2)

    sum_x = 0
    sum_y = 0
    sum_u = 0
    sum_v = 0
    sum_xx_yy = 0
    sum_ux_vy = 0
    sum_vx_uy = 0

    for c in range(points_num):
        x_off = c * 2
        y_off = x_off + 1
        tmp_x = float(tmp_landmark[c * 2])
        tmp_y = float(tmp_landmark[c * 2 + 1])
        sum_x += tmp_x
        sum_y += tmp_y
        sum_u += std_points[x_off]
        sum_v += std_points[y_off]
        sum_xx_yy += tmp_x * tmp_x + \
                        tmp_y * tmp_y
        sum_ux_vy += tmp_x * std_points[x_off] + \
                        tmp_y * std_points[y_off]
        sum_vx_uy += std_points[y_off] * tmp_x - \
                        std_points[x_off] * tmp_y

    if sum_xx_yy < 1e-16:
        return None

    q = sum_u - sum_x * sum_ux_vy * 1. / sum_xx_yy \
                + sum_y * sum_vx_uy * 1. / sum_xx_yy
    p = sum_v - sum_y * sum_ux_vy * 1. / sum_xx_yy \
                - sum_x * sum_vx_uy * 1. / sum_xx_yy

    r = points_num - (sum_x * sum_x + sum_y * sum_y) * 1. / sum_xx_yy

    if r < 1e-16 and r > -1e-16:
        return None

    a = (sum_ux_vy - sum_x * q * 1. / r - sum_y * p * 1. / r) * 1. / sum_xx_yy
    b = (sum_vx_uy + sum_y * q * 1. / r - sum_x * p * 1. / r) * 1. / sum_xx_yy
    c = q * 1. / r
    d = p * 1. / r

    tform[0][0] = a
    tform[0][1] = -b
    tform[0][2] = c
    tform[1][0] = b
    tform[1][1] = a
    tform[1][2] = d

    return tform
