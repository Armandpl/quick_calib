import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# cam intrinsinc from c2k19 (eon)
FOCAL_LEN = 910.0
IMAGE_W = 1164
IMAGE_H = 874

K = np.array(
    [
        [FOCAL_LEN, 0., IMAGE_W / 2.],
        [0., FOCAL_LEN, IMAGE_H / 2.],
        [0., 0., 1.],
    ]
).astype(np.float32)

K_inv = np.linalg.inv(K)

# eon is based on the leeco pro 3/OnePlus 3T, sensor is Sony IMX298
# 4672x3520 effective px and 1.12um per px = 5.23264x3.9424mm
SENSOR_W = 5.23264
SENSOR_H = 3.9424


def get_vp_from_calib(pitch, yaw):
    pitch = -pitch
    # Step 1: Reconstruct r3 from yaw and pitch
    r3 = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])

    # Step 2: Get p_infinity by multiplying r3 with K
    p_infinity = np.dot(K, r3)

    # Step 3: Calculate the homogeneous coordinates (u_i, v_i) from p_infinity
    u_i, v_i, _ = p_infinity / p_infinity[2]

    return u_i, v_i


def get_calib_from_vp(vp):
  vp_norm = normalize(vp)
  yaw_calib = np.arctan(vp_norm[0])
  pitch_calib = -np.arctan(vp_norm[1]*np.cos(yaw_calib))
  # TODO should be, this but written
  # to be compatible with meshcalib and
  # get_view_frame_from_road_fram
  #pitch_calib = -np.arctan(vp_norm[1]*np.cos(yaw_calib))
  roll_calib = 0
  return roll_calib, pitch_calib, yaw_calib


def normalize(img_pts):
  # normalizes image coordinates
  # accepts single pt or array of pts
  img_pts = np.array(img_pts)
  input_shape = img_pts.shape
  img_pts = np.atleast_2d(img_pts)
  img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0],1))))
  img_pts_normalized = K_inv.dot(img_pts.T).T
  img_pts_normalized[(img_pts < 0).any(axis=1)] = np.nan
  return img_pts_normalized[:,:2].reshape(input_shape)