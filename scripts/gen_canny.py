from pathlib import Path

import cv2
import kornia
import numpy as np
import torch
from kornia.geometry.conversions import convert_points_from_homogeneous
from kornia.geometry.linalg import transform_points
from tqdm import trange

from quick_calib.camera import FOCAL_LEN, IMAGE_H, IMAGE_W
from quick_calib.orientation import euler2rot

# device/mesh : x->forward, y-> right, z->down
# view : x->right, y->down, z->forward
device_frame_from_view_frame = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
view_frame_from_device_frame = device_frame_from_view_frame.T


def compute_EE(roll, pitch, yaw, tx, ty, tz):
    """compute camera extrinsic matrix from pose tx, ty, tz is translation from origin in world
    coords roll, pitch, yaw is camera orientation in world coords."""
    RC = euler2rot(np.array([roll, pitch, yaw]))
    RC = np.einsum("jk,ik->ij", view_frame_from_device_frame, RC)
    C = np.array([tx, ty, tz])
    R = RC.T
    T = -R @ C
    E = np.eye(4)
    # E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = T
    return E.astype(np.float32)


def make_lane_lines(nb=4, len=500, width=3.65):
    """Returns a list of lane lines on the road plane.

    Each lane line is a tuple of two points representing the start and end of the line. Half of the
    lines are on the left side of the road, half on the right side. They are spaced by width. So
    the first right lane is just width/2 to the right of the origin. Lanes are paralel to the x
    axis (forward). They are of length len.
    """
    lane_lines = []
    for i in range(nb):
        y = -width / 2 - width * (nb // 2 - 1) + i * width
        x_start = 0
        x_end = len
        lane_lines.append(((x_start, y), (x_end, y)))
    return lane_lines


def make_lane_lines_vertices(lane_lines, line_width=0.15):
    """Takes in lane lines from make_lane_lines and returns a list of 4 vertices for each line."""
    lane_line_vertices = []
    for line in lane_lines:
        vertices = []
        (x1, y1), (x2, y2) = line
        vertices.append((x1, y1 - line_width / 2))
        vertices.append((x1, y1 + line_width / 2))
        vertices.append((x2, y2 + line_width / 2))
        vertices.append((x2, y2 - line_width / 2))
        lane_line_vertices.append(vertices)

    return lane_line_vertices


def make_image(roll, pitch, yaw, tx, ty, tz):
    EE2 = compute_EE(
        roll,
        pitch,
        yaw,
        tx,
        ty,
        tz,
    )

    image = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)

    K2 = np.array(
        [
            [FOCAL_LEN, 0, IMAGE_W / 2, 0],
            [0, FOCAL_LEN, IMAGE_H // 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ).astype(np.float32)

    camera = kornia.geometry.camera.pinhole.PinholeCamera(
        intrinsics=torch.from_numpy(K2).unsqueeze(0),
        extrinsics=torch.from_numpy(EE2).unsqueeze(0),
        height=torch.tensor([IMAGE_H]),
        width=torch.tensor([IMAGE_W]),
    )

    lane_lines = make_lane_lines(nb=6)
    for vertices in make_lane_lines_vertices(lane_lines):
        points_to_project = np.array(vertices)
        points_to_project = np.concatenate(
            [points_to_project, np.zeros((points_to_project.shape[0], 1))], axis=1
        )
        points_to_project = torch.from_numpy(points_to_project).to(torch.float32)
        P = camera.intrinsics @ camera.extrinsics
        homogeneous_points = transform_points(P, points_to_project)

        # if points are behind the camera, discard them
        homogeneous_points = homogeneous_points[homogeneous_points[..., 2] > 0]

        points_to_plot = convert_points_from_homogeneous(homogeneous_points)
        # append last point to close the polygon
        points_to_plot = torch.cat([points_to_plot, points_to_plot[0].unsqueeze(0)], dim=0)
        # convert to int
        points_to_plot = points_to_plot.int()

        for i in range(points_to_plot.shape[0] - 1):
            p1 = points_to_plot[i].tolist()
            p2 = points_to_plot[i + 1].tolist()
            cv2.line(image, p1, p2, (255, 255, 255), 1)

    return image


def gen(output_dir: Path, max_pitch, max_yaw, max_ty, max_tz, nb_images):
    output_dir.mkdir(exist_ok=False)

    # gen labels
    # each row is pitch, yaw, ty, tz
    # for each row each value is randomly choosen between -max and max
    labels = np.random.uniform(
        low=[-max_pitch, -max_yaw, -max_ty, -max_tz],
        high=[max_pitch, max_yaw, max_ty, max_tz],
        size=(nb_images, 4),
    )
    np.save(output_dir / "labels.npy", labels)

    # 'hack' so that camera is in a place where we don't get too many glitches
    # glithes happend when points are behind the camera
    # for now we just don't show those points which hides the full line
    # TODO fix it
    init_tx = -1.2
    init_tz = -2.0
    init_ty = 0.0

    for idx in trange(nb_images):
        pitch, yaw, ty, tz = labels[idx]
        image = make_image(0, pitch, yaw, init_tx, init_ty + ty, init_tz + tz)
        cv2.imwrite(str(output_dir / f"{idx}.png"), image)


def viz():
    forward = -1.2
    down = -2.0
    right = 0
    pitch = 0.0
    yaw = 0.0
    roll = 0

    while True:
        cv2.imshow("synth road", make_image(roll, pitch, yaw, forward, right, down))

        # Listen for key presses
        key = cv2.waitKey(0) & 0xFF

        increment_m = 0.5
        increment_rad = np.deg2rad(10)

        # Update the shift values based on arrow key presses
        if key == ord("z"):  # Quit the program
            break
        elif key == ord("w"):  # up
            down += increment_m
        elif key == ord("s"):  # down
            down -= increment_m
        elif key == ord("a"):  # left
            right -= increment_m
        elif key == ord("d"):  # right
            right += increment_m
        elif key == ord("q"):  # forward
            forward += increment_m
        elif key == ord("e"):  # backward
            forward -= increment_m
        elif key == ord("u"):  # roll clockwise
            roll += increment_rad
        elif key == ord("o"):  # roll counterclockwise
            roll -= increment_rad
        elif key == ord("i"):  # pitch up
            pitch += increment_rad
        elif key == ord("k"):  # pitch down
            pitch -= increment_rad
        elif key == ord("j"):  # yaw left
            yaw -= increment_rad
        elif key == ord("l"):  # yaw right
            yaw += increment_rad

    cv2.destroyAllWindows()


DEBUG = False
OUTPUT_DIR = Path("../data/synth_road")
NB_IMAGES = 500
MAX_PITCH = np.deg2rad(20)
MAX_YAW = np.deg2rad(20)
MAX_TY = 2.5
MAX_TZ = 1.5

if __name__ == "__main__":
    if DEBUG:
        viz()
    else:
        gen(OUTPUT_DIR, MAX_PITCH, MAX_YAW, MAX_TY, MAX_TZ, NB_IMAGES)
