import numpy as np
from interpolation.splines import CubicSplines
from typing import Optional, Tuple, Union, List
import spherelib as splib
from matplotlib import pyplot as plt


class Cam360:
    """
    It implements a 360-degree camera.

    It stores:
    -   the rotation matrix from the world coordinate system to the local one,
    -   the translation vector from local coordinate system to the world one
        (the vector is expressed in the local coordinate system),
    -   the radius adopted in the local parametrization,
    -   the camera texture,
    -   the camera depth map.

    The camera texture is organized in an HxW equiangular grid:

    -   the i-th row corresponds to theta = (i*delta_theta + delta_theta/2), with i = 0, 1, ..., (H-1),
        with delta_theta = pi/H.

    -   the j-th column corresponds to phi = (j*delta_phi + delta_phi/2), with j = 0, 1, ... , (W-1),
        with delta_phi = 2*pi/W.

    In particular, the texture is stored as an (H, W, channels) array with texture[i, j, :] the color at
    (theta, phi) = (i*delta_theta + delta_theta/2, j*delta_phi + delta_phi/2).

    The depth map associated to the camera is stored similarly, but it has always a single channel.
    """

    # Texture value range.
    TEXTURE_MIN = 0.0
    TEXTURE_MAX = 1.0

    # Depth value range.
    DEPTH_MIN = 1e-4
    DEPTH_MAX = 100

    # This class uses Cubic Splines for texture or depth interpolation.
    _SPLINE_ORDER = 3

    def __init__(self,
                 rotation_mtx: np.array, translation_vec: np.array,
                 height: int, width: int, channels: int,
                 texture: Optional[np.array] = None, depth: Optional[np.array] = None):

        # Rotation matrix and translation vector.
        self._rotation_mtx = rotation_mtx
        self._translation_vec = translation_vec

        # Check the input parameters height, width, and channels.
        if (height < 1) or (width < 1) or (channels < 1):
            raise ValueError('Bad input resolution or number of channels.')

        # Set the camera specs (i.e., the equiangular grid dimensions and number of channels).
        self._height = height
        self._width = width
        self._channels = channels
        self._specs = (self._height, self._width, self._channels)

        # Radius of the sphere which models the camera.
        self._radius = 1.0

        # Dimensions of a pixel in the equiangular grid.
        self._delta_theta = np.pi / self._height
        self._delta_phi = (2 * np.pi) / self._width

        # Set the texture, if available.
        self._texture = None
        self._texture_spline = None
        self.texture = texture              # Initialization of self._texture and self._texture_spline.

        # Set the depth map, if available.
        self._depth = None
        self.depth = depth                  # Initialization of self._depth.

    @property
    def rotation_mtx(self) -> np.array:
        return self._rotation_mtx

    @property
    def translation_vec(self) -> np.array:
        return self._translation_vec

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def specs(self) -> Tuple[int, int, int]:
        return self._specs

    @property
    def texture(self) -> Optional[np.array]:
        return self._texture

    @texture.setter
    def texture(self, texture_new: np.array) -> None:

        if texture_new is None:

            self._texture = None
            self._texture_spline = None

        else:

            # Initialize the camera texture (self._texture).

            if texture_new.ndim == 2:

                # Single channel texture.

                # Check the input texture resolution and number of channels.
                if texture_new.shape[0] != self._height or \
                        texture_new.shape[1] != self._width or \
                        self._channels != 1:
                    raise ValueError('The input texture does not match the camera specs.')

                # Clip the texture in the case of values outside the allowed range.
                texture_new_clipped = np.clip(texture_new, Cam360.TEXTURE_MIN, Cam360.TEXTURE_MAX)

                # Set the texture and add a third dimension.
                # Adding a third dimension permits to process single and multi channel images similarly.
                self._texture = np.expand_dims(texture_new_clipped, axis=2)

            elif texture_new.ndim == 3:

                # Multi channel texture.

                # Check the input texture resolution and number of channels.
                if texture_new.shape[0] != self._height or \
                        texture_new.shape[1] != self._width or \
                        texture_new.shape[2] != self._channels:
                    raise ValueError('The input texture does not match the camera specs.')

                # Clip the texture in the case of values outside the allowed range.
                texture_new_clipped = np.clip(texture_new, Cam360.TEXTURE_MIN, Cam360.TEXTURE_MAX)

                # Set the texture.
                self._texture = texture_new_clipped

            else:

                raise ValueError('The input texture must be either a 2D or a 3D numpy array.')

            # Start the initialization of self._texture_spline.
            # self._texture_spline is a (spline-based) continuous version of the camera texture self._texture.
            # It will be useful to evaluate the texture at off-grid coordinates.

            # The texture needs to be padded in order to get rid of the border effects during interpolation.
            padding = Cam360._SPLINE_ORDER + 1

            # Pad the top and bottom parts of the camera texture by repeating the first and last rows, respectively.
            texture_wrapped = np.concatenate(
                (np.tile(self._texture[0, :, :], (padding, 1, 1)),
                 self._texture,
                 np.tile(self._texture[-1, :, :], (padding, 1, 1))), axis=0)

            # Pad the left and right sides of the camera texture in a circular fashion.
            texture_wrapped = np.concatenate(
                (texture_wrapped[:, (-padding):, :],
                 texture_wrapped,
                 texture_wrapped[:, 0:padding, :]), axis=1)

            # Compute the first and last value of the extended theta axis.
            # The axis is referred to as 'extended' as it takes the vertical padding into account.
            extended_theta_axis_first_value = (self._delta_theta / 2) - (self._delta_theta * padding)
            extended_theta_axis_last_value = np.pi - (self._delta_theta / 2) + (self._delta_theta * padding)

            # Compute the first and last value of the extended phi axis.
            # The axis is referred to as 'extended' as it takes the horizontal padding into account.
            extended_phi_axis_first_value = (self._delta_phi / 2) - (self._delta_phi * padding)
            extended_phi_axis_last_value = (2 * np.pi) - (self._delta_phi / 2) + (self._delta_phi * padding)

            # Compute the lengths of the extended theta and phi axes.
            extended_theta_axis_length = texture_wrapped.shape[0]
            extended_phi_axis_length = texture_wrapped.shape[1]

            # Create a spline approximation of the camera texture.
            low = [extended_theta_axis_first_value, extended_phi_axis_first_value]
            up = [extended_theta_axis_last_value, extended_phi_axis_last_value]
            orders = [extended_theta_axis_length, extended_phi_axis_length]
            self._texture_spline = CubicSplines(
                low, up, orders,
                np.reshape(texture_wrapped, (texture_wrapped.shape[0] * texture_wrapped.shape[1], self._channels)))

    @property
    def depth(self) -> Optional[np.array]:
        return self._depth

    @depth.setter
    def depth(self, depth_new: np.array) -> None:

        if depth_new is None:

            self._depth = None

        else:

            if depth_new.ndim != 2:
                raise ValueError('The input depth map must be a 2D numpy array.')

            if depth_new.shape[0] != self._height or (depth_new.shape[1] != self._width):
                raise ValueError('The input depth map does not match the camera specs.')

            # Clip the depth in the case of values outside the allowed range.
            depth_new_clipped = np.clip(depth_new, Cam360.DEPTH_MIN, Cam360.DEPTH_MAX)

            self._depth = depth_new_clipped

    def get_texture_at(self,
                       theta: np.array, phi: np.array, grad=False) -> \
            Optional[Union[np.array, Tuple[np.array, np.array, np.array]]]:
        """
        It computes the texture at the specified (theta, phi) coordinate. It computes the texture gradient if required.
        The gradient is the standard one for planar images.

        Args:
            theta: the target elevation coordinate (N,).
            phi: the target azimuth coordinate (N,).
            grad: True if the gradient is required, False by default.

        Returns:
            texture: texture (self._channels, N) at the input coordinate.
            d_theta: elevation component of the texture gradient (self._channels, N) at the input coordinate.
            d_phi: azimuth component of the texture gradient (self._channels, N) at the input coordinate.
        """

        if self._texture is None:

            return None

        else:

            # Arrange the evaluation coordinates in a single array.
            points = np.column_stack((theta, phi))

            if grad:

                # Compute the texture and the gradient.
                aux = self._texture_spline.interpolate(points, diff=True)

                # Texture.
                texture = aux[0].T

                # Clip the texture in the case of values outside the allowed range.
                texture = np.clip(texture, Cam360.TEXTURE_MIN, Cam360.TEXTURE_MAX)

                #  Gradient.
                d_theta = np.squeeze((aux[1])[:, 0, :]).T
                d_phi = np.squeeze((aux[1])[:, 1, :]).T

                return texture, d_theta, d_phi

            else:

                # Compute the texture.
                texture = self._texture_spline.interpolate(points, diff=False).T

                # Clip the texture in the case of values outside the allowed range.
                texture = np.clip(texture, Cam360.TEXTURE_MIN, Cam360.TEXTURE_MAX)

                return texture

    def get_depth_at(self, theta: np.array, phi: np.array) -> Optional[np.array]:
        """
        It returns the depth value at the coordinate (theta, phi).
        The returned depth value is obtained via interpolation of self._depth.

        Args:
            theta: the elevation coordinate (N,).
            phi: the azimuth coordinate (N,).

        Returns:
            the depth (N,) at the specified coordinates or None if self._depth is not available.
        """

        if self._depth is None:

            return None

        else:

            # Create a (spline-based) continuous version of the camera depth map self._depth.
            # It will be useful to evaluate the texture at off-grid coordinates.

            # Differently from the camera texture, this construction is carried out at each call of this function,
            # as the function is expected to be called rarely.

            # The depth map needs to be padded in order to get rid of the border effects during interpolation.
            padding = Cam360._SPLINE_ORDER + 1

            # Pad the top and bottom parts of the camera depth map by repeating the first and last rows, respectively.
            depth_wrapped = np.concatenate(
                (np.tile(self._depth[0, :], (padding, 1)),
                 self._depth,
                 np.tile(self._depth[-1, :], (padding, 1))), axis=0)

            # Pad the left and right sides of the camera depth map in a circular fashion.
            depth_wrapped = np.concatenate(
                (depth_wrapped[:, (-padding):],
                 depth_wrapped,
                 depth_wrapped[:, 0:padding]), axis=1)

            # Compute the first and last value of the extended theta axis.
            # The axis is referred to as 'extended' as it takes the vertical padding into account.
            extended_theta_axis_first_value = (self._delta_theta / 2) - (self._delta_theta * padding)
            extended_theta_axis_last_value = np.pi - (self._delta_theta / 2) + (self._delta_theta * padding)

            # Compute the first and last value of the extended phi axis.
            # The axis is referred to as 'extended' as it takes the horizontal padding into account.
            extended_phi_axis_first_value = (self._delta_phi / 2) - (self._delta_phi * padding)
            extended_phi_axis_last_value = (2 * np.pi) - (self._delta_phi / 2) + (self._delta_phi * padding)

            # Compute the lengths of the extended theta and phi axes.
            extended_theta_axis_length = depth_wrapped.shape[0]
            extended_phi_axis_length = depth_wrapped.shape[1]

            # Create a spline approximation of the camera texture.
            low = [extended_theta_axis_first_value, extended_phi_axis_first_value]
            up = [extended_theta_axis_last_value, extended_phi_axis_last_value]
            orders = [extended_theta_axis_length, extended_phi_axis_length]
            depth_spline = CubicSplines(low, up, orders, np.expand_dims(depth_wrapped.ravel(), axis=1))

            # Evaluate the depth map spline approximation at the input (theta, phi) coordinates.
            points = np.column_stack((theta, phi))
            depth = depth_spline.interpolate(points, diff=False)
            depth = np.squeeze(depth)

            # Clip the depth in the case of values outside the allowed range.
            depth = np.clip(depth, Cam360.DEPTH_MIN, Cam360.DEPTH_MAX)

            return depth

    def rotate(self,
               rot_mtx: Optional[np.array] = None,
               alpha: Optional[Tuple] = None, order: Optional[Tuple] = None) -> Optional['Cam360']:
        """
        It returns a new object Cam360 obtained by rotating the current camera according to the input rotation matrix.
        Remember that rotating the camera means rotating its coordinate system (the scene remains still).

        Args:
            rot_mtx: rotation matrix (3, 3).
            alpha: 3-tuple with alpha[0], alpha[1], and alpha[3] the rotation angles around the X-axis, Y-axis,
                   and Z-axis, respectively.
            order: 3-tuple specifying the order in which the rotations have to be performed, with 0, 1, and 2 indicating
                   the X-axis, Y-axis, and Z-axis, respectively.
                   E.g., order = (2, 0, 1) specifies that the order in which the rotations need to be carried out is
                   Z first, X second, and Y third.

        Returns:
            the new rotated camera, or None if self._texture is not available.
        """

        if self._texture is None:
            return None

        # Build the rotation matrix, if this is not provided at the input.
        if rot_mtx is None:
            rot_mtx = splib.rotation_mtx(alpha, order)

        # Build an equiangular grid in (theta, phi) coordinates.
        theta, phi = self.get_sampling_axes()
        phi_grid, theta_grid = np.meshgrid(phi, theta)

        # Vectorize the computed equiangular grid.
        theta = theta_grid.ravel()
        phi = phi_grid.ravel()

        # Compute where each pixel in the rotated camera falls in the original camera (self).
        x, y, z = splib.pol2eu(theta, phi, 1)
        aux = rot_mtx.T.dot(np.stack((x, y, z), axis=0))
        x_new = aux[0]
        y_new = aux[1]
        z_new = aux[2]
        theta_new, phi_new, _ = splib.eu2pol(x_new, y_new, z_new)

        # Compute the texture of the rotated camera.
        texture_new = self.get_texture_at(theta_new, phi_new)

        # Reshape the texture of the rotated camera.
        texture_new = np.expand_dims(texture_new, axis=2)
        texture_new = np.transpose(texture_new, (2, 1, 0))
        texture_new = np.reshape(texture_new, (self._height, self._width, self._channels))

        depth_new = None
        if self._depth is not None:

            # Compute the depth map of the rotated camera.
            depth_new = self.get_depth_at(theta_new, phi_new)

            # Reshape the depth map of the rotated camera.
            depth_new = np.reshape(depth_new, (self._height, self._width))

        # Compute the rotation matrix and translation vector of the rotated camera.
        rotation_mtx_new = rot_mtx.dot(self._rotation_mtx)
        translation_vec_new = rot_mtx.dot(self._translation_vec)

        cam_new = Cam360(rotation_mtx_new, translation_vec_new,
                         self._height, self._width, self._channels,
                         texture_new, depth_new)

        return cam_new

    def generate_texture(self, cam: 'Cam360') -> Optional[np.array]:
        """
        It uses the input camera and self._depth_map to generate an estimate of the current camera texture.

        Args:
            cam: the camera whose texture is used to generate the current camera texture.

        Returns:
            the generated texture (N, N, 3), if self._depth_map and the texture of the input camera are both available.
        """

        if self._depth is None:
            raise ValueError('The current camera has no depth map available')

        if cam.texture is None:
            return ValueError('The input camera has no texture available.')

        if self._specs != cam.specs:
            raise ValueError('The current camera and the input one have different dimensions.')

        # Build an equiangular grid in (theta, phi) coordinates.
        theta, phi = self.get_sampling_axes()
        phi_grid, theta_grid = np.meshgrid(phi, theta)

        # Vectorize the computed equiangular grid.
        theta = theta_grid.ravel()
        phi = phi_grid.ravel()

        theta_new, phi_new = splib.project(
            theta, phi,
            self._depth.ravel(),
            self._rotation_mtx, self._translation_vec, self._radius,
            cam.rotation_mtx, cam.translation_vec, cam.radius)

        # Compute the new texture.
        texture_new = cam.get_texture_at(theta_new, phi_new)

        # Reshape the new texture.
        texture_new = np.reshape(texture_new.T, (self._height, self._width, self._channels))

        return texture_new

    def get_sampling_axes(self) -> Tuple[np.array, np.array]:
        """
        It returns the elevation and azimuth axes associated to the samples in self._texture.

        Returns:
            theta: elevation coordinates (N,).
            phi: azimuth coordinates(N,).
        """

        theta = np.linspace(self._delta_theta / 2, np.pi - (self._delta_theta / 2), self._height)
        phi = np.linspace(self._delta_phi / 2, (2 * np.pi) - (self._delta_phi / 2), self._width)

        return theta, phi

    @staticmethod
    def compute_baseline(cam_one: 'Cam360', cam_two: 'Cam360') -> float:
        """
        It computes the euclidean distance between the two camera centers.

        Args:
            cam_one: reference camera.
            cam_two: secondary camera.

        Returns:
            the euclidean distance between the two camera centers.
        """

        baseline = np.linalg.norm(
            cam_one.translation_vec - cam_one.rotation_mtx.dot(cam_two.rotation_mtx.T).dot(cam_two.translation_vec))

        return baseline

    @staticmethod
    def rectify(cam_one: 'Cam360', cam_two: 'Cam360', param: List = []) -> Tuple['Cam360', 'Cam360', List]:
        """
        It rectifies, or de-rectify, the two input cameras using the first one as the reference camera.
        When rectification is performed, the rectified reference camera and the rectified secondary cameras can be
        treated as the left and right cameras of a standard (i.e., perspective) stereo pair, respectively.
        Note that the two textures of the rectified cameras are rectified along the columns. To have the two textures
        rectified along the rows, the two textures can be simply transposed.

        Args:
            cam_one: reference camera.
            cam_two: secondary camera.
            param: None for rectification. If an inverse rectification is required, it must contain:
                   the original rotation matrix of the reference camera,
                   the original rotation matrix of the secondary camera,
                   the original theta coordinate of the epipole,
                   the original phi coordinate of the epipole,
                   in the specified order.

        Returns:
            cam_one_rec: (de-)rectified reference camera.
            cam_two_rec: (de-)rectified secondary camera.
            rect_param: the parameters necessary to revert the rectification.
        """

        # TODO: check whether the two cameras have the same specs.

        # The rectification could be sped up by merging the two rotations of the secondary camera into a single one.

        rect_param = []

        if len(param) == 0:

            # Save the reference and secondary camera rotation matrices.
            rect_param.append(np.copy(cam_one.rotation_mtx))
            rect_param.append(np.copy(cam_two.rotation_mtx))

            # Step 1.
            # Align the secondary camera coordinate system with the reference camera one.
            cam_two_new = cam_two.rotate(rot_mtx=cam_one.rotation_mtx.dot(cam_two.rotation_mtx.T))

            # Compute the epipolar vector, which starts in the origin of the reference camera coordinate system and
            # ends in the origin of the secondary camera coordinate system.
            # The epipolar vector is expressed in the reference camera coordinate system.
            epipole_vec = \
                cam_one.translation_vec - cam_one.rotation_mtx.dot(cam_two.rotation_mtx.T).dot(cam_two.translation_vec)

            # Compute the polar coordinates of the epipolar vector, in the reference camera coordinate system.
            theta, phi, _ = \
                splib.eu2pol(np.array([epipole_vec[0]]), np.array([epipole_vec[1]]), np.array([epipole_vec[2]]))
            theta = theta[0]
            phi = phi[0]

            # The line hosting the epipolar vector intersects the sphere of the reference camera at the coordinates
            # (theta, phi) and (np.pi-theta, np.pi+phi), which are the epipoles of the reference camera.

            # Since the reference camera and the (aligned) secondary camera have their coordinate systems aligned,
            # the (aligned) secondary camera has its epipoles at (theta, phi) and (np.pi-theta, np.pi+phi) as well.

            # Step 2.
            # Rotate the reference camera and the (aligned) secondary camera such that their epipoles at (theta, phi)
            # are moved to their north poles. Equivalently, their Z-axes must be aligned with the epipolar vector and
            # point in its same direction.
            cam_one_new = cam_one.rotate(alpha=(theta, 0.0, np.pi - phi), order=(2, 0, 1))
            cam_two_new = cam_two_new.rotate(alpha=(theta, 0.0, np.pi - phi), order=(2, 0, 1))

            # Save the epipole coordinates.
            rect_param.append(theta)
            rect_param.append(phi)

        else:

            if len(param) != 4:
                raise ValueError('Bad (revert) rectification parameters !!!')

            # TODO: check parameters in param.

            # Extract the parameters to revert the rectification..
            rot_mtx_one = param[0]
            rot_mtx_two = param[1]
            theta = param[2]
            phi = param[3]

            # Revert the rectification.
            cam_one_new = cam_one.rotate(alpha=(-theta, 0.0, -(np.pi - phi)), order=(1, 0, 2))
            cam_two_new = cam_two.rotate(alpha=(-theta, 0.0, -(np.pi - phi)), order=(1, 0, 2))
            cam_two_new = cam_two_new.rotate(rot_mtx_two.dot(rot_mtx_one.T))

        return cam_one_new, cam_two_new, rect_param
