from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def generate_random_points(num_points: int) -> dict:
    rng = np.random.default_rng()
    heights = rng.uniform(0, 1, num_points)
    phis = rng.uniform(0, 2 * np.pi, num_points)
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


def generate_arrays(projected_coordinates: dict, shape: tuple) -> list:
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

    scale_x = (shape[1] - 10) / (max_x - min_x)
    scale_y = (shape[0] - 10) / (max_y - min_y)
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
            frames[i][coord[1] : coord[1] + 5, coord[0] : coord[0] + 5] = 1

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
    ani.save('res.mp4', fps=60)
    # plt.show()


if __name__ == "__main__":
    coordinates = generate_random_points(100)
    rotational_coordinates = generate_rotation(120, "counter", coordinates)
    projected_coordinates = generate_projected_coordinates(
        0.3, 2, rotational_coordinates
    )

    coordinates = generate_random_points(150)
    rotational_coordinates = generate_rotation(120, "clock", coordinates)
    projected_coordinates_2 = generate_projected_coordinates(
        0.6, 2, rotational_coordinates
    )

    projected_coordinates = append_coordinates(
        projected_coordinates, projected_coordinates_2
    )
    frames = generate_arrays(projected_coordinates, (480, 640))

    show_animation(frames)
