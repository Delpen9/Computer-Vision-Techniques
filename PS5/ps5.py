"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import cv2
import numpy as np
import os
import random
import bisect

from ps5_utils import run_kalman_filter, run_particle_filter

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# I/O directories
input_dir = fr'{ROOT_DIR}/input_images'
output_dir = fr'{ROOT_DIR}/output'

# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""
    def __init__(self, init_x, init_y, Q = 0.1 * np.eye(4), R = 0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.F = np.matrix([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.H = np.matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.state = np.matrix([init_x, init_y, 0., 0.]).T
        self.P = np.matrix(np.eye(4)) * 0.1
        self.Q = np.matrix(Q.copy())
        self.R = np.matrix(R.copy())

    def predict(self):
        self.state = self.F * self.state
        self.P = self.F * self.P * self.F.T + self.Q

    def correct(self, meas_x, meas_y):
        measurement = np.matrix([meas_x, meas_y])

        residual_covariance = self.H * self.P * self.H.T + self.R
        kalman_gain = self.P * self.H.T * np.linalg.pinv(residual_covariance)
        self.state = self.state + kalman_gain * (measurement.T - self.H * self.state)

    def process(self, measurement_x, measurement_y):
        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.interval = 0

        self.template = template.copy()
        self.frame = frame.copy()
        self.particles = np.ones((self.num_particles, 2), dtype=np.float64)  # Initialize your particles array. Read the docstring.
        self.weights = np.ones((self.num_particles, 1), dtype=np.float64) / (self.num_particles)  # Initialize your weights array. Read the docstring.

        self.best_template = np.ones((self.template.shape), dtype=np.float64)

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        mean_squared_error = np.mean(np.square(np.subtract(
            frame_cutout,
            template
        )))

        similarity_value = np.e ** (-mean_squared_error / (2 * self.sigma_exp))

        return similarity_value

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        ## Normalize the array
        self.weights /= np.sum(self.weights)
        resampling_wheel = np.cumsum(self.weights)
        particle_data_structure = np.ones((self.num_particles, 2), dtype=np.float64)
        for i in range(self.num_particles):
            sample_value = random.random()
            sample_idx = bisect.bisect_left(resampling_wheel, sample_value)

            particle_pos_x = self.particles[sample_idx, 1] + np.random.normal(0, self.sigma_dyn)
            particle_pos_y = self.particles[sample_idx, 0] + np.random.normal(0, self.sigma_dyn)
            bool_x_coord = (0 < particle_pos_x < self.frame.shape[0])
            bool_y_coord = (0 < particle_pos_y < self.frame.shape[1])
            if not bool_x_coord:
                particle_pos_x = self.particles[sample_idx, 0]
            if not bool_y_coord:
                particle_pos_y = self.particles[sample_idx, 1]

            particle_data_structure[i, :] = [int(particle_pos_y), int(particle_pos_x)]

        self.weights = np.ones((self.num_particles, 1), dtype=np.float64) / (self.num_particles)
        return particle_data_structure

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        self.frame = frame.copy()

        ## Runs only to initially set the particle values
        if self.interval == 0:
            frame_rows = frame.shape[0]
            frame_cols = frame.shape[1]
            for i in range(self.num_particles):
                random_x = random.randrange(frame_rows)
                random_y = random.randrange(frame_cols)
                self.particles[i] = [int(random_x), int(random_y)]
        
        self.interval += 1

        for i in range(self.num_particles):
            if self.template.shape[0] % 2 == 0:
                x_dim_left = int(self.template.shape[0] / 2)
                x_dim_right = int(self.template.shape[0] / 2)
            else:
                x_dim_left = int(self.template.shape[0] / 2)
                x_dim_right = int(self.template.shape[0] / 2) + 1
            
            if self.template.shape[1] % 2 == 0:
                y_dim_left = int(self.template.shape[1] / 2)
                y_dim_right = int(self.template.shape[1] / 2)
            else:
                y_dim_left = int(self.template.shape[1] / 2)
                y_dim_right = int(self.template.shape[1] / 2) + 1

            frame_cutout = self.frame[
                int(self.particles[i, 1] - x_dim_left) : int(self.particles[i, 1] + x_dim_right),
                int(self.particles[i, 0] - y_dim_left) : int(self.particles[i, 0] + y_dim_right)
            ]

            if frame_cutout.shape == self.template.shape:
                similarity_value = self.get_error_metric(self.template.copy(), frame_cutout)
            else:
                similarity_value = 0
                
            self.weights[i] = self.weights[i] * similarity_value

        self.particles = self.resample_particles()
        
    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """
        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
            cv2.circle(frame_in, (int(self.particles[i, 0]), int(self.particles[i, 1])), 1, (0, 0, 255), 1)

        cv2.circle(frame_in, (int(x_weighted_mean), int(y_weighted_mean)), 10, (255, 255, 255), 5)
        point_top_left = (int(x_weighted_mean - self.template.shape[1] / 2), int(y_weighted_mean - self.template.shape[0] / 2))
        point_bottom_right = (int(x_weighted_mean + self.template.shape[1] / 2), int(y_weighted_mean + self.template.shape[0] / 2))
        cv2.rectangle(frame_in, pt1 = point_top_left, pt2 = point_bottom_right, color = (255, 255, 255), thickness = 10)
        # cv2.imshow('frame', frame_in)
        # cv2.waitKey(0)


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def get_best_template(self):
        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        if self.template.shape[0] % 2 == 0:
            x_dim_left = int(self.template.shape[0] / 2)
            x_dim_right = int(self.template.shape[0] / 2)
        else:
            x_dim_left = int(self.template.shape[0] / 2)
            x_dim_right = int(self.template.shape[0] / 2) + 1
        
        if self.template.shape[1] % 2 == 0:
            y_dim_left = int(self.template.shape[1] / 2)
            y_dim_right = int(self.template.shape[1] / 2)
        else:
            y_dim_left = int(self.template.shape[1] / 2)
            y_dim_right = int(self.template.shape[1] / 2) + 1

        self.best_template = self.frame[
                int(y_weighted_mean - x_dim_left) : int(y_weighted_mean + x_dim_right),
                int(x_weighted_mean - y_dim_left) : int(x_weighted_mean + y_dim_right)
        ]

        if self.best_template.shape == self.template.shape:
            self.template = self.alpha * self.best_template + (1 - self.alpha) * self.template

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        self.frame = frame.copy()

        ## Runs only to initially set the particle values
        if self.interval == 0:
            frame_rows = frame.shape[0]
            frame_cols = frame.shape[1]
            for i in range(self.num_particles):
                random_x = random.randrange(frame_rows)
                random_y = random.randrange(frame_cols)
                self.particles[i] = [int(random_x), int(random_y)]
        
        self.interval += 1

        if self.interval > 15:
            self.get_best_template()

        for i in range(self.num_particles):
            if self.template.shape[0] % 2 == 0:
                x_dim_left = int(self.template.shape[0] / 2)
                x_dim_right = int(self.template.shape[0] / 2)
            else:
                x_dim_left = int(self.template.shape[0] / 2)
                x_dim_right = int(self.template.shape[0] / 2) + 1
            
            if self.template.shape[1] % 2 == 0:
                y_dim_left = int(self.template.shape[1] / 2)
                y_dim_right = int(self.template.shape[1] / 2)
            else:
                y_dim_left = int(self.template.shape[1] / 2)
                y_dim_right = int(self.template.shape[1] / 2) + 1

            frame_cutout = self.frame[
                int(self.particles[i, 1] - x_dim_left) : int(self.particles[i, 1] + x_dim_right),
                int(self.particles[i, 0] - y_dim_left) : int(self.particles[i, 0] + y_dim_right)
            ]

            if frame_cutout.shape == self.template.shape:
                similarity_value = self.get_error_metric(self.template.copy(), frame_cutout)
            else:
                similarity_value = 0
                
            self.weights[i] = self.weights[i] * similarity_value

        self.particles = self.resample_particles()


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        raise NotImplementedError


def part_1b(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_2 = {'x': 7.5, 'y': 7.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_2, "matching",
                            save_frames, template_loc, Q, R)
    return out


def part_1c(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_1 = {'x': 2.5, 'y': 2.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_1, "hog",
                            save_frames, template_loc, Q, R)
    return out


def part_2a(obj_class, template_loc, save_frames, input_folder):
    num_particles = 1000  # Define the number of particles
    sigma_mse = 0.09 # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 15 # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_2b(obj_class, template_loc, save_frames, input_folder):
    num_particles = 1000  # Define the number of particles
    sigma_mse = 0.09 # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 15  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_3(obj_class, template_rect, save_frames, input_folder):
    num_particles = 1000  # Define the number of particles
    sigma_mse = 0.09 # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 35  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.3  # Set a value for alpha

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        # input video
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_coords=template_rect)  # Add more if you need to
    return out


def part_4(obj_class, template_rect, save_frames, input_folder):
    num_particles = 1000  # Define the number of particles
    sigma_md = 0.09  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 75  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        template_coords=template_rect)  # Add more if you need to
    return out
