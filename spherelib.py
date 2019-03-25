import numpy as np
import math
import pyshtools
from typing import Tuple, Union

"""
WE ASSUME A RIGHT-HANDED COORDINATE SYSTEM X-Y-Z.
This means that the right thumb points along the Z-axis in the positive Z-direction and the curl of the other fingers
represents a motion from the X-axis to the Y-axis.

SPHERE PARAMETRIZATION:
Let us consider a point (x, y, z) in our right-handed coordinate system:
1. theta is the angle, in [0, pi], from the positive Z-semiaxis to the segment connecting the origin (0, 0, 0) to the
   point (x, y, z).
2. phi is the angle, in [0, 2*pi], between the negative Y-semiaxis and the segment connecting (0, 0, 0) to (x, y, 0),
   with the rotation proceeding counter-clockwise when observed from the positive Z-semiaxis.

ROTATIONS:
A rotation keeps the world fixed and rotates the axis of the coordinate system.
A rotation of the coordinate system by alpha degrees around the X-axis corresponds to a counter-clockwise rotation
observed from the positive X-semiaxis. The same holds true for the Y-axis and the Z-axis.
"""


def pol2eu(theta: np.array, phi: np.array, radius: np.array) -> Tuple[np.array, np.array, np.array]:
    """
    It turns the (theta, phi, rad) polar coordinate into the (x, y, z) Euclidean coordinate.

    Args:
        theta: elevation coordinate (N,).
        phi: azimuth coordinate (N,).
        radius: radius (N,).

    Returns:
        x: x-axis value (N,).
        y: y-axis value (N,).
        z: z-axis value (N,).
    """

    # Pre-compute values used multiple times.
    radius_sin_theta = radius * np.sin(theta)

    x = - radius_sin_theta * np.sin(phi)
    y = - radius_sin_theta * np.cos(phi)
    z = + radius * np.cos(theta)
    # x, y, z are (N,) np.arrays.

    return x, y, z


def eu2pol(x: np.array, y: np.array, z: np.array) -> Tuple[np.array, np.array, np.array]:
    """
    It turns the Euclidean coordinate (x, y, z) to the polar coordinate (theta, phi, rad).

    Args:
        x: x-axis value (N,).
        y: y-axis value (N,).
        z: z-axis value (N,).

    Returns:
        theta: elevation coordinate (N,).
        phi: azimuth coordinate (N,).
        radius: radius (N,).
    """

    # Compute the norm of the vector (x, y, z).
    radius = np.sqrt((x ** 2) + (y ** 2) + (z ** 2))

    theta = np.arccos(z / radius)
    phi = np.arctan2(- x, - y)
    # np.arctan2 returns an angle in [-pi,pi], while we want phi to be in [0,2*pi]. Therefore ...
    mask = (phi < 0)
    phi[mask] = (2 * np.pi) + phi[mask]
    # theta_new, phi_new are (N,) np.arrays.

    return theta, phi, radius


def project(theta: np.array, phi: np.array, depth: np.array,
            rot_mtx_a: np.array, t_vec_a: np.array, rad_a: float,
            rot_mtx_b: np.array, t_vec_b: np.array, rad_b: float,
            grad=False) -> Union[Tuple[np.array, np.array], Tuple[np.array, np.array, np.array, np.array]]:
    """
    It projects the coordinate (theta, phi) in the sphere A to the sphere B. It computes the derivative of each
    component, if required.
    The rotation matrices and the translation vectors refer to a common world coordinate system.

    Args:
        theta: elevation coordinate (N,) in the sphere A.
        phi: azimuth coordinate (N,) in the sphere A.
        depth: depth values (N,) associated to the coordinate (theta, phi) in the sphere A.
        rot_mtx_a: rotation matrix (3, 3) from the world coordinate system to the one of the sphere A.
        t_vec_a: translation vector (3,) from the coordinate system of the sphere A to the world one.
        rad_a: radius of the sphere A.
        rot_mtx_b: rotation matrix (3, 3) from the world coordinate system to the one of the sphere B.
        t_vec_b: translation vector (3,) from the coordinate system of the sphere B to the world one.
        rad_b: radius of the sphere B.
        grad: True if the derivatives are required, False by default.

    Returns:
        theta_proj: the corresponding elevation (N,) on the sphere B.
        phi_proj: the corresponding azimuth (N,) on the sphere B.
        d_theta_proj: the corresponding elevation derivative (N,).
        d_phi_proj: the corresponding azimuth derivative (N,).
    """

    # Compute the 3D coordinate of theta and phi on the sphere A.
    x, y, z = pol2eu(theta, phi, rad_a)
    f = np.stack((x, y, z), axis=0)

    # Compute auxiliary variables.
    a = rot_mtx_b.dot(rot_mtx_a.T) * (rad_b / rad_a)
    b = (- rot_mtx_b.dot(rot_mtx_a.T).dot(t_vec_a) + t_vec_b) * rad_b
    c = (2 / rad_a) * (t_vec_b.dot(rot_mtx_b).dot(rot_mtx_a.T) - t_vec_a)
    d = t_vec_a.dot(t_vec_a) + t_vec_b.dot(t_vec_b) - \
        (2 * t_vec_b.dot(rot_mtx_b).dot(rot_mtx_a.T).dot(t_vec_a))

    # Compute the projection on the sphere B in 3D coordinates.
    a_f = a.dot(f)
    c_f = c.dot(f)
    num = (a_f * depth) + b[:, None]
    den_sq = (depth ** 2) + (c_f * depth) + d
    f_proj = num / np.sqrt(den_sq)

    # Compute the projection on the sphere B in polar coordinate.
    theta_proj, phi_proj, _ = eu2pol(f_proj[0, :], f_proj[1, :], f_proj[2, :])
    # theta_proj, phi_proj are (N,) np.arrays.

    # Note that the radius returned by eu2pol, and here ignored, should be equal to rad_b.

    if grad:

        # Compute the gradient of the elevation component of the projection.
        d_theta_proj = ((- 1) / np.sqrt(((rad_b ** 2) * den_sq) - (num[2, :] ** 2))) * \
                       (a_f[2, :] - ((num[2, :] * (depth + 0.5 * c_f)) / den_sq))

        # Compute the gradient of the azimuth component of the projection.
        d_phi_proj = ((a_f[0, :] * b[1]) - (a_f[1, :] * b[0])) / ((num[1, :] ** 2) + (num[0, :] ** 2))

        return theta_proj, phi_proj, d_theta_proj, d_phi_proj

    else:

        return theta_proj, phi_proj


def surface(theta_a: np.array, theta_b: np.array, phi_a: np.array, phi_b: np.array, rad: float) -> np.array:
    """
    It returns the surface area of the patch with vertices (theta_a, phi_a), (theta_a, phi_b),
    (theta_b, phi_b), and (theta_b, phi_a) with theta_a <= theta_b and phi_a <= phi_b.

    Args:
        theta_a: smaller elevation coordinate (N,).
        theta_b: larger elevation coordinate (N,).
        phi_a: smaller azimuth coordinate (N,).
        phi_b: larger azimuth coordinate (N,).
        rad: sphere radius.

    Returns:
        the surface area enclosed by the four input vertices (N,).
    """

    area = (rad**2) * (np.cos(theta_a) - np.cos(theta_b)) * (phi_b - phi_a)

    return area


def rotation_mtx(alpha: Tuple, order: Tuple) -> np.array:
    """
    It creates a rotation matrix that implements a rotation of the coordinate system X-Y-Z (the scene does not move).
    The rotation consists of three sequential rotation around the X-axis, the Y-axis, and the Z-axis, in the order
    specified by the user.

    THE ROTATION IS PERFORMED COUNTER-CLOCKWISE WHEN WATCHING AN AXIS FROM ITS POSITIVE SEMI-AXIS !!!
    THIS CONVENTION IS NOT RELATED TO THE POLAR PARAMETRIZATION !!!

    Args:
        alpha: 3-tuple with alpha[0], alpha[1], and alpha[3] the angles of the rotation around the X-axis, Y-axis,
               and Z-axis, respectively.
        order: 3-tuple specifying the order in which the rotations have to be performed, with 0, 1, and 2 indicating the
               X-axis, Y-axis, and Z-axis, respectively.
               E.g., order = (2, 0, 1) specifies that the order in which the rotations need to be carried out is
               Z first, X second, and Y third.

    Returns:
        the rotation matrix (3, 3).
    """

    # Rotation angles.
    alpha_x = alpha[0]
    alpha_y = alpha[1]
    alpha_z = alpha[2]

    # X-axis rotation matrix.
    rot_mtx_x = np.array([
        [1.0,  0.0,               0.0            ],
        [0.0,  np.cos(alpha_x),   np.sin(alpha_x)],
        [0.0,  -np.sin(alpha_x),  np.cos(alpha_x)]
    ])

    # Y-axis rotation matrix.
    rot_mtx_y = np.array([
        [ np.cos(alpha_y),  0.0,  -np.sin(alpha_y)],
        [ 0.0,              1.0,  0.0            ],
        [np.sin(alpha_y),   0.0,  np.cos(alpha_y)]
    ])

    # Z-axis rotation matrix.
    rot_mtx_z = np.array([
        [np.cos(alpha_z),   np.sin(alpha_z),  0.0],
        [-np.sin(alpha_z),  np.cos(alpha_z),  0.0],
        [0.0,               0.0,              1.0]
    ])

    # Organize the rotation matrices in a tuple.
    aux = (rot_mtx_x, rot_mtx_y, rot_mtx_z)

    # Compose the rotations according to the order specified at the input.
    rot_mtx = np.dot(aux[order[2]], np.dot(aux[order[1]], aux[order[0]]))

    return rot_mtx


def graph_nn(theta: np.array, phi: np.array, radius: float, loop: bool, sigma: float) -> Tuple[np.array, np.array]:
    """
    It builds a graph which models a patch on the sphere. The patch is described by an equiangular grid.
    Each entry in the equiangular grid is a node in the graph. Each entry in the equiangular grid is connected to its 8
    adjacent entries. Each edge has a weight that is inversely proportional to the geodesic distance (on the sphere)
    between the two entries.

    Args:
        theta: elevation coordinates (N,).
        phi: azimuth coordinates (N,).
        radius: sphere radius.
        loop: True if the equiangular grid covers the whole sphere and the phi coordinates have to wrap around.
        sigma: weight normalization constant.

    Returns:
        weight: weight matrix (8, N), with W(i, j) the weight from the node j to the node I(i, j) (for I, see below).
                The node j correspond to the j-th entry encountered when scanning the equiangular grid in ROW MAJOR
                ORDER.
        index: index matrix (8, N).
    """

    # Equiangular grid dimensions.
    height = len(theta)
    width = len(phi)

    # Number of nodes.
    node_num = height * width

    # Source nodes.
    node = np.arange(node_num)

    # Equiangular grid in parameter space.
    grid_phi, grid_theta = np.meshgrid(phi, theta)

    # Equiangular grid in (row, col) coordinate (each pair is a node in (row, col) indexing).
    grid_h, grid_v = np.meshgrid(np.arange(width), np.arange(height))

    # Allocate the graph matrices.
    weight = np.zeros((8, node_num), dtype=float)
    index = np.tile(node, (8, 1))   # Initialize the graph with autoloops.

    # Build the neighbor window.
    win_h, win_v = np.meshgrid(np.array([-1, 0, 1]), np.array([-1, 0, 1]))
    win_v = win_v.flatten()
    win_h = win_h.flatten()
    neigh_num = len(win_v)

    # Geodesic weights are symmetric, therefore we consider half of the neighbor window entries.
    win_v = win_v[0:int((neigh_num - 1) / 2.0)]
    win_h = win_h[0:int((neigh_num - 1) / 2.0)]
    neigh_num = len(win_v)

    for i in np.arange(neigh_num):

        # Destination coordinates.
        grid_v_dest = grid_v + win_v[i]
        if loop:
            grid_h_dest = np.mod(grid_h + win_h[i], width)
        else:
            grid_h_dest = grid_h + win_h[i]

        # Detect the valid destination coordinates.
        mask = (grid_v_dest >= 0) & (grid_v_dest < height) & (grid_h_dest >= 0) & (grid_h_dest < width)

        # Compute the edge weights.
        wei = geodesic_dist(
            grid_theta[mask],
            grid_phi[mask],
            grid_theta[grid_v_dest[mask], grid_h_dest[mask]],
            grid_phi[grid_v_dest[mask], grid_h_dest[mask]],
            radius)
        wei = np.exp(- np.power(wei, 2) / np.power(sigma, 2))

        # Valid destination nodes.
        node_dest = np.ravel_multi_index((grid_v_dest[mask], grid_h_dest[mask]), (height, width))

        # Flatten the mask.
        mask = mask.ravel()

        # Store the edge weights and the connections.
        weight[2 * i, mask] = wei
        index[2 * i, mask] = node_dest

        # Store the edge weights for the symmetric connections.
        weight[(2 * i) + 1, node_dest] = wei
        index[(2 * i) + 1, node_dest] = node[mask]

    return weight, index


def graph_tv(theta: np.array, phi: np.array, radius: float, loop: bool) -> Tuple[np.array, np.array]:
    """
    It builds a graph which models a patch on the sphere. The patch is described by an equiangular grid.
    Each entry in the equiangular grid is a node in the graph. Each entry (theta, phi) in the equiangular grid is
    connected to its top neighbor (theta-1, phi) and right neighbor (theta, phi+1) with weigths 1/radius and
    1 / (radius * sin(theta)), respectively. This graph is used to implement TV regularization on the sphere.

    Args:
        theta: elevation coordinates (N,).
        phi: azimuth coordinates (N,).
        radius: sphere radius.
        loop: True if the equiangular grid covers the whole sphere and the phi coordinates have to wrap around.

    Returns:
        weight: weight matrix (2, N), with W(i, j) the weight from the node j to the node index(i, j) (see below).
                The node j corresponds to the j-th entry encountered when scanning the equiangular grid
                in ROW MAJOR ORDER.
        index: index matrix (2, N).
    """

    # Equiangular grid dimensions.
    height = len(theta)
    width = len(phi)

    # Number of nodes.
    node_num = height * width

    # Source nodes.
    node = np.arange(node_num)

    # Equiangular grid in parameter space.
    grid_phi, grid_theta = np.meshgrid(phi, theta)

    # Equiangular grid in (row, col) coordinate (each pair is a node in (row, col) indexing).
    grid_h, grid_v = np.meshgrid(np.arange(width), np.arange(height))

    # Allocate the graph matrices.
    weight = np.zeros((2, node_num), dtype=float)
    index = np.tile(node, (2, 1))  # Initialize the graph with auto-loops.

    # BUILD THE >>> TOP <<< CONNECTIONS:

    # Compute the top node coordinates.
    grid_h_dest = grid_h
    grid_v_dest = grid_v - 1

    # Detect the destination nodes with valid coordinates.
    mask = (grid_v_dest >= 0) & (grid_v_dest < height)

    # Turn the valid destination node coordinates into linear indexes.
    node_dest = np.ravel_multi_index((grid_v_dest[mask], grid_h_dest[mask]), (height, width))

    # Store the valid connections.
    mask = mask.ravel()
    index[0, mask] = node_dest
    weight[0, mask] = 1.0 / radius

    # BUILD THE >>> RIGHT <<< CONNECTIONS:

    # Compute the right node coordinates.
    grid_h_dest = grid_h + 1
    grid_v_dest = grid_v

    # If the full sphere is considered, the last column of the equiangular grid is connected with the first one.
    if loop:
        grid_h_dest = np.mod(grid_h_dest, width)

    # Detect the destination nodes with valid coordinates (the check is necessary only when loop is false).
    mask = (grid_h_dest >= 0) & (grid_h_dest < width)

    # Turn the valid destination node coordinates into linear indexes.
    node_dest = np.ravel_multi_index((grid_v_dest[mask], grid_h_dest[mask]), (height, width))

    # Compute the valid destination node weights.
    node_dest_weight = 1.0 / (radius * np.sin(grid_theta[mask]))

    # Store the valid connections.
    mask = mask.ravel()
    index[1, mask] = node_dest
    weight[1, mask] = node_dest_weight

    return weight, index


def geodesic_dist(theta_a: np.array, phi_a: np.array, theta_b: np.array, phi_b: np.array, radius: float) -> np.array:
    """
    It computes the geodesic distance between the points (theta_a, phi_a) and (theta_b, phi_b) on the sphere.

    Args:
        theta_a: elevation coordinate of the first point (N,).
        phi_a: azimuth coordinate of the first point (N,).
        theta_b: elevation coordinate of the second point (N,).
        phi_b: azimuth coordinate the second point (N,).
        radius: sphere radius.

    Returns:
        The geodesic distance between the pair of points at the input (N,).

    """

    # Euclidean coordinate.
    x_a, y_a, z_a = pol2eu(theta_a, phi_a, radius)
    x_b, y_b, z_b = pol2eu(theta_b, phi_b, radius)

    # Euclidean distance between (x_a, y_a, z_a) and (x_b, y_b, z_b).
    dist = np.sqrt(np.power(x_a - x_b, 2) + np.power(y_a - y_b, 2) + np.power(z_a - z_b, 2))

    # Geodesic distance between (x_a, y_a, z_a) and (x_b, y_b, z_b).
    dist_geo = 2 * radius * np.arcsin(dist / (2 * radius))

    return dist_geo


def sphere_pyramid_down(image: np.array, sigma: float) -> np.array:

    # Input image dimensions.
    height = image.shape[0]
    width = image.shape[1]

    n = math.log2(height)
    if math.ceil(n) != math.floor(n):
        raise ValueError('The input image height must be a power of two.')

    if width != (2 * height):
        raise ValueError('The input image width must be equal to twice the input image height.')

    # Spherical Fourier Transform.
    spectrum = pyshtools.expand.SHExpandDH(image, norm=4, sampling=2)

    # Build the filter.
    bandwidth = spectrum.shape[1]
    axis = np.arange(bandwidth)
    filter_gauss = np.exp(- 0.5 * axis * (axis + 1) * (sigma ** 2))

    # Filter the image coefficients.
    filter_gauss = filter_gauss[None, :, None]
    spectrum_filtered = spectrum * filter_gauss

    # Inverse Spherical Fourier Transform.
    image_filtered = pyshtools.expand.MakeGridDH(spectrum_filtered, norm=4, sampling=2)

    # Down-sample.
    image_filtered = image_filtered[::2, ::2]

    return image_filtered


def sphere_pyramid_up(image: np.array, sigma: float) -> np.array:

    # Input image dimensions.
    height = image.shape[0]
    width = image.shape[1]

    n = math.log2(height)
    if math.ceil(n) != math.floor(n):
        raise ValueError('The input image height must be a power of two.')

    if width != (2 * height):
        raise ValueError('The input image width must be equal to twice the input image height.')

    # Zero filling.
    image_filled = np.zeros((2 * height, 2 * width))
    image_filled[::2, ::2] = image

    # Spherical Fourier Transform.
    spectrum = pyshtools.expand.SHExpandDH(image_filled, norm=4, sampling=2)

    # Build the filter.
    bandwidth = spectrum.shape[1]
    axis = np.arange(bandwidth)
    filter_gauss = np.exp(- 0.5 * axis * (axis + 1) * (sigma ** 2))

    # Filter the image coefficients.
    filter_gauss = filter_gauss[None, :, None]
    spectrum_filtered = spectrum * filter_gauss

    # Inverse Spherical Fourier Transform.
    image_filtered = pyshtools.expand.MakeGridDH(spectrum_filtered, norm=4, sampling=2)

    return image_filtered
