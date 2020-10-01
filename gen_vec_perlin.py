from math import floor 
import numpy as np
from datetime import datetime


class VecPerlinNoise():

    def __init__(self, dimensions=2, period=100):
        self.grid = self.generate_gradients(dimensions, period)
        self.period = period
        self.dimensions = dimensions

    def generate_gradients(self, d, d_length):
        # Create a d-dimensional hypercube as the gradient
        # sample space 
        s_space = [d_length] * d
        # Create a d-dimensional gradient vector at each point
        # in the sample space (each component in [-1, 1])
        gradients = np.random.rand(*s_space, d) * 2 - 1
        # Add single axis to allow for division of gradient
        # vectors by their norms
        norms = np.linalg.norm(gradients, axis=d)[..., np.newaxis]
        return gradients / norms

    # 2D noise function
    def noise_2(self, x, y):
        # Mod x and y coordinates for repetition of noise
        # if x and y length are greater than gradient gridsize
        x = x % self.period
        y = y % self.period
        # Calculate the square of grid points that each point lies within
        x0 = np.floor(x).astype(int)
        x1 = np.where(x0 == self.period - 1, 0, x0 + 1).astype(int)
        y0 = np.floor(y).astype(int)
        y1 = np.where(y0 == self.period - 1, 0, y0 + 1).astype(int)
        # Find the gradients at each of the 4 grid points
        # nearest to the given point
        grad_00 = self.grid[x0, y0]
        grad_01 = self.grid[x0, y1]
        grad_10 = self.grid[x1, y0]
        grad_11 = self.grid[x1, y1]
        # Distances between points and their nearest grid vertices
        x_dist = x - x0
        y_dist = y - y0
        # Dot gradients at corner vertices with distances from points
        # (creates 4 scalar values for every point)
        n_00 = grad_00[..., 0] * x_dist + grad_00[..., 1] * y_dist
        n_01 = grad_01[..., 0] * x_dist + grad_01[..., 1] * (y_dist - 1)
        n_10 = grad_10[..., 0] * (x_dist - 1) + grad_10[..., 1] * y_dist
        n_11 = grad_11[..., 0] * (x_dist - 1) + grad_11[..., 1] * (y_dist - 1)
        # Interpolate between noise values
        w1 = self.quintic(x_dist)
        nx0 = n_00 * (1 - w1) + n_10 * w1
        nx1 = n_01 * (1 - w1) + n_11 * w1
        # Interpolate between dot3 and dot4
        w2 = self.quintic(y_dist)
        return nx0 * (1 - w2) + nx1 * w2

    """ An implementation of classical Perlin noise vectorized using
    numpy arrays in order to improve speed.  The noise function
    can handle data of arbitrary dimensionality, provided that the
    VecPerlinNoise instance associated with it has the same
    number of dimensions. 
    Expects args to contain the coordinates of points for which to
    calculate noise values, separated by component, e.g. x, y, z
    """
    def __noise(self, *args):
        if len(args) != self.dimensions:
            raise ValueError('Wrong dimensionality of points supplied')
        
        point_components = []
        for comp in args:
            point_components.append(comp % self.period)
        point_components = np.array(point_components)
        comp_pairs = []
        total_distances = []
        for comp in point_components:
            comp0 = np.floor(comp).astype(int)
            comp1 = np.where(comp0 == self.period - 1, 0, comp0 + 1).astype(int)
            total_distances.append(comp - comp0)
            comp_pairs.append([comp0, comp1])
        # Number of vertices of hypercube in which point falls = 2^(num_dimensions)
        num_corners = 2 ** self.dimensions
        noise_points = []
        # Calculate gradients at and distances to grid vertices
        for corner in range(num_corners):
            indices = []
            divisor = num_corners / 2
            distances = []
            for dist_0, pair in zip(total_distances, comp_pairs):
                pair_index = floor(corner / divisor) % 2
                indices.append(pair[pair_index])
                divisor /= 2
                dist = dist_0 - 1 if pair_index == 1 else dist_0
                distances.append(dist)
            # Get the gradient at this corner
            gradient = self.grid[tuple(indices)]
            distances = np.stack(distances, axis=-1)
            # Dot product of gradient with distance to grid cell
            mult = np.multiply(gradient, distances)
            dot = np.sum(mult, axis=-1)
            noise_points.append(dot)

        # Interpolate noise values across all dimensions (vertices of cube)
        # starting with the last dimension
        for dimension in range(self.dimensions-1, -1, -1):
            dimension_distances = self.quintic(total_distances[dimension])
            max_dist = np.max(dimension_distances)
            new_vertices = []
            for vertex in range(0, len(noise_points), 2):
                mult1 = np.multiply(noise_points[vertex], 1 - dimension_distances)
                mult2 = np.multiply(noise_points[vertex+1], dimension_distances)
                new_vertices.append(mult1 + mult2)
            noise_points = new_vertices

        return noise_points[0]

    """ Noise wrapper to be called from outside class.  Can be used to
        create octave / harmonic noise (layered noise).  Expects args
        to be a sequence of point coordinates (in float form) that are each
        array-like.
    """
    def noise(self, *args, octaves=1, base_freq=1, persistence=.5):
        if octaves < 1:
            return args
        frequency = base_freq
        base_noise = self.__noise(*args)
        amplitude = 1
        for octave in range(1, octaves):
            freq_coords = [c * frequency for c in args]
            base_noise += self.__noise(*freq_coords) * amplitude
            amplitude *= persistence
            frequency = frequency * 2

        return base_noise


    def quintic(self, t):
        return 6 * (t ** 5) - 15 * (t ** 4) + 10 * (t ** 3)
