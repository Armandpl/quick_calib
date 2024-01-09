import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# cam intrinsinc from c2k19 (eon)
FOCAL_LEN = 910
IMAGE_W = 1164
IMAGE_H = 874

K = np.array(
    [
        [FOCAL_LEN, 0, IMAGE_W / 2],
        [0, FOCAL_LEN, IMAGE_H / 2],
        [0, 0, 1],
    ]
).astype(np.float32)

# eon is based on the leeco pro 3/OnePlus 3T, sensor is Sony IMX298
# 4672x3520 effective px and 1.12um per px = 5.23264x3.9424mm
SENSOR_W = 5.23264
SENSOR_H = 3.9424


def get_vp_from_calib(pitch, yaw):
    # Step 1: Reconstruct r3 from yaw and pitch
    r3 = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])

    # Step 2: Get p_infinity by multiplying r3 with K
    p_infinity = np.dot(K, r3)

    # Step 3: Calculate the homogeneous coordinates (u_i, v_i) from p_infinity
    u_i, v_i, _ = p_infinity / p_infinity[2]

    return u_i, v_i
