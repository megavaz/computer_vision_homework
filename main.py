from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


RNG = np.random.default_rng()


def generate_random_points(num_points: int) -> dict:
    heights = RNG.uniform(0, 1, num_points)
    phis = RNG.uniform(0, 2 * np.pi, num_points)
    coordinates = {i: [h, p] for i, (h, p) in enumerate(zip(heights, phis))}
    return coordinates


def rotate(phi: float, num_points: int, direction: str = "clock") -> list:
    rot_phi = np.linspace(phi, phi + 2 * np.pi, num_points, endpoint=False)
    rot_phi = list(map(lambda a: a if a < 2 * np.pi else a - 2 * np.pi, rot_phi))
    if direction == "counter":
        rot_phi = rot_phi[::-1]
    return rot_phi


def generate_rotation(
    num_point_per_full_rotation: int, direction: str, coordinates: dict
) -> dict:
    rotational_coordinates = {}
    for point_num, point_coords in coordinates.items():
        rotational_coordinates[point_num] = [
            point_coords[0],
            rotate(point_coords[1], num_point_per_full_rotation, direction),
        ]
    return rotational_coordinates


def projection(point_coords: list, radius: float, distance_to_camera: float) -> list:
    z = point_coords[0]
    x_y = []
    for phi in point_coords[1]:
        x_y.append((radius * np.cos(phi), -radius * np.sin(phi)))
    x_y_proj = []
    for x, y in x_y:
        x_proj = -y / distance_to_camera
        y_proj = (0.5 - z) / (distance_to_camera - x) + 0.5
        x_y_proj.append((x_proj, y_proj))
    return x_y_proj


def generate_projected_coordinates(
    radius: float, distance_to_camera: float, rotational_coordinates: dict
) -> dict:
    projected_coordinates = {}
    for point_num, point_coords in rotational_coordinates.items():
        projected_coordinates[point_num] = projection(
            point_coords, radius, distance_to_camera
        )
    return projected_coordinates


def generate_normalized_coordinates(
    min_x: float, min_y: float, scale: float, shift_x: int, shift_y: int
) -> Any:
    def generate(projected_coordinates):
        normalized_coordinates = (
            int(round((projected_coordinates[0] - min_x) * scale) + shift_x),
            int(round((projected_coordinates[1] - min_y) * scale) + shift_y),
        )
        return normalized_coordinates

    return generate


def generate_arrays(
    projected_coordinates: dict, shape: tuple, volume: bool = False
) -> list:
    max_x = 0
    min_x = 0
    max_y = 0.5
    min_y = 0.5

    for point in projected_coordinates.values():
        for item in point:
            if item[0] < min_x:
                min_x = item[0]
            if item[0] > max_x:
                max_x = item[0]
            if item[1] < min_y:
                min_y = item[1]
            if item[1] > max_y:
                max_y = item[1]

    scale_x = (shape[1] - 30) / (max_x - min_x)
    scale_y = (shape[0] - 30) / (max_y - min_y)
    scale = min(scale_x, scale_y)

    normalized_coordinates = {}

    shift_x = round((shape[1] - (max_x - min_x) * scale) / 2)
    shift_y = round((shape[0] - (max_y - min_y) * scale) / 2)

    for point_number, point_coordinates in projected_coordinates.items():
        normalized_coordinates[point_number] = list(
            map(
                generate_normalized_coordinates(min_x, min_y, scale, shift_x, shift_y),
                point_coordinates,
            )
        )

    frames = [np.zeros(shape) for i in range(len(normalized_coordinates[0]))]
    for point_number, point_coordinates in normalized_coordinates.items():
        for i, coord in enumerate(point_coordinates):
            frames[i][coord[1] : coord[1] + 1, coord[0] : coord[0] + 1] = 1
    if volume:
        frames = list(map(lambda a: np.repeat(a[..., np.newaxis], 3, axis=-1), frames))
    return frames


def append_coordinates(c1: dict, c2: dict) -> dict:
    shift = len(c1)
    for im_num, img in c2.items():
        c1[im_num + shift] = img
    return c1


def show_animation(frames: list):
    f, ax = plt.subplots()
    ims = []
    for img in frames:
        ims.append([ax.imshow(img, animated=True)])

    ani = animation.ArtistAnimation(f, ims, 16, blit=True)
    # ani.save("res.gif", fps=30)
    plt.show()


def pick_random_point(frame):
    y_s, x_s = np.where(frame == 1)
    i = RNG.integers(0, len(y_s), endpoint=False)
    return y_s[i], x_s[i]


def find_closest(frame, y, x, shape=(20, 20), motion_vector=(0, 0)):
    y += motion_vector[0]
    x += motion_vector[1]
    area = frame[
        y - int(shape[0] // 2) : y + int(shape[0] // 2),
        x - int(shape[1] // 2) : x + int(shape[1] // 2),
    ]
    y_s, x_s = np.where(area[..., 0] == 1)
    min_distance = shape[0] ** 2 + shape[1] ** 2
    for y_cand, x_cand in zip(y_s, x_s):
        distance = (y_cand - int(shape[0] // 2)) ** 2 + (
            x_cand - int(shape[1] // 2)
        ) ** 2
        if distance < min_distance:
            closest_y, closest_x = y_cand, x_cand
            min_distance = distance

    try:
        distance_vector = (
            closest_y - int(shape[0] // 2),
            closest_x - int(shape[1] // 2),
        )
    except UnboundLocalError:
        return find_closest(
            frame,
            y - motion_vector[0],
            x - motion_vector[1],
            (shape[0] + 10, shape[1] + 10),
            motion_vector,
        )
    motion_vector = (
        motion_vector[0] + distance_vector[0],
        motion_vector[1] + distance_vector[1],
    )
    return y + distance_vector[0], x + distance_vector[1], motion_vector


def track_point(frames):
    y_start, x_start = pick_random_point(frames[0][..., 0])
    y_s, x_s = [y_start], [x_start]
    last_y = y_start
    last_x = x_start
    motion_vector = (0, 0)
    motion_vectors = [motion_vector]
    for frame in frames[1:]:
        if len(motion_vectors) > 3:
            c = 0.3
            motion_vector = (
                motion_vector[0] + int(c * (motion_vector[0] - motion_vectors[-2][0])),
                motion_vector[1] + int(c * (motion_vector[1] - motion_vectors[-2][1])),
            )
        last_y, last_x, motion_vector = find_closest(
            frame, last_y, last_x, shape=(20, 20), motion_vector=motion_vector
        )
        y_s.append(last_y)
        x_s.append(last_x)
        motion_vectors.append(motion_vector)
    for i, (y, x) in enumerate(zip(y_s, x_s)):
        frames[i][y, x, 1:] = 0
    return frames, motion_vectors, y_start / frames[0].shape[0] < 0.5


def measure_angular_speed(motion_vectors):
    horizontal_directions = []
    for vector in motion_vectors:
        if vector[1] != 0:
            horizontal_directions.append(vector[1] / np.abs(vector[1]))
        else:
            horizontal_directions.append(0)
    horizontal_directions = np.array(horizontal_directions)
    total_frames_in_one_direction = np.maximum(np.count_nonzero(horizontal_directions == 1), np.count_nonzero(horizontal_directions == -1)) + (np.count_nonzero(horizontal_directions == 0) - 1) / 2
    return np.pi / total_frames_in_one_direction, horizontal_directions


def find_direction(motion_vectors, horizontal_directions, upper_half):
    vertical_directions = []
    for vector in motion_vectors:
        if vector[0] != 0:
            vertical_directions.append(vector[0] / np.abs(vector[0]))
        else:
            vertical_directions.append(0)
    vertical_directions = np.array(vertical_directions)
    reached_start_of_left_motion = False
    for i in range(1, len(horizontal_directions)):
        if reached_start_of_left_motion:
            if vertical_directions[i] == 1:
                if upper_half:
                    print('Rotation is counter-clockwise!')
                else:
                    print('Rotation is clockwise!')
                return
            elif vertical_directions[i] == -1:
                if upper_half:
                    print('Rotation is clockwise!')
                else:
                    print('Rotation is counter-clockwise!')
                return
            else:
                continue

        else:
            if horizontal_directions[i] == 0 and horizontal_directions[i+1] == -1 and horizontal_directions[i+2] == -1:
                reached_start_of_left_motion = True


def scale_dots(frames, scale_factor=4):
    for i, f in enumerate(frames):
        coordinates = np.where(f[..., 0] == 1)
        for (y, x) in zip(coordinates[0], coordinates[1]):
            frames[i][y: y + scale_factor, x: x + scale_factor] = f[y,x]
    return frames


if __name__ == "__main__":
    coordinates = generate_random_points(60)
    rotational_coordinates = generate_rotation(120, "clock", coordinates)
    projected_coordinates = generate_projected_coordinates(
        1.2, 3, rotational_coordinates
    )

    coordinates = generate_random_points(50)
    rotational_coordinates = generate_rotation(120, "counter", coordinates)
    projected_coordinates_2 = generate_projected_coordinates(
        0.6, 3, rotational_coordinates
    )

    projected_coordinates = append_coordinates(
        projected_coordinates, projected_coordinates_2
    )
    frames_tracked = generate_arrays(projected_coordinates, (240, 360), volume=True)
    for i in range(1):
        frames_tracked, motion_vectors, upper_half = track_point(frames_tracked)
        

    angular_speed, horizontal_directions = measure_angular_speed(motion_vectors)
    print('angular speed is {} r/f or {} d/f'.format(angular_speed, angular_speed * 180 / np.pi))
    find_direction(motion_vectors, horizontal_directions, upper_half)

    frames_tracked = scale_dots(frames_tracked)
    show_animation(frames_tracked)
