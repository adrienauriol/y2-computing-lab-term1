# -*- coding: utf-8 -*-
"""
This file is provided to students to generate parameters for assignment 1
the 'Boxes' objects for assignment 2 (Boxes class), and the class Rocket for
assignment 3.
"""

from .generator import Generator
import __main__ as main
import logging
import copy
import numpy as np
import numpy.ma as ma  # For masking in RLC_transfer_func
import math
import random
import pickle
import gzip
import os

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

# Setup logging
assignment_logger = logging.getLogger(__name__)
# WARNING to suppress excess logging
assignment_logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(levelname)s: %(message)s')

# file_handler = logging.FileHandler('assignment.log')
# file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# assignment_logger.addHandler(file_handler)
assignment_logger.addHandler(stream_handler)


class Assignment1(Generator):
    """Child class of Generator, inherits allowed_IDs."""

    def __init__(self, studentID):
        """Constructor called in the assignment notebook.
        - studentID is provided by the student.
        - Should be an integer contained within allowed_IDs in Generator.
        - The special value 0 is also allowed in case the student's ID is
          not contained within allowed_IDs, to allow them to get started.
        - Code below ensures that they have to enter a valid ID to make progress on the
          assignment.
        """
        super().__init__(studentID)  # Call Generator __init__ method to get parameters

    def get_parameters(self):
        """Return well parameters specific to studentID
        Also prints an explanatory message to the student.
        Returns:
        a, V0 : (float, float)
            A tuple for assignment to a (well width in m), V0 (well height in eV) by the caller
        """
        # Guard against student trying to progress without parameters
        try:
            a, V0 = self.width, self.potential
        except AttributeError:
            return None, None

        print('The cell width is a = {:.2e} m and '
              'potential is V0 = {:.2f} eV.\n'.format(a, V0))
        print('Do not overwrite these values, and do not use manually set values.\n')
        print('Instead reference the variables "a" and "V0" directly where needed.\n')

        return a, V0

    def check_plot(self):
        """
        NB: Called by check function
        Checks whether student_figure has:

        1) been defined
        2) is a figure object (by ensuring get_axes works)
        3) has axes
        """
        # Check that student_figure is defined
        if not hasattr(main, 'student_figure'):
            assignment_logger.info('student_figure not defined.')
            print('student_figure is not defined.')
        else:
            # Now it is defined, try and extract axes from figure
            try:
                # .get_axes() returns empty list for no axes
                if not main.student_figure.get_axes():
                    assignment_logger.info('student_figure has no axes')
                    print(
                        'student_figure has no axes. Ensure you are plotting onto the student_figure object.')
                else:
                    assignment_logger.info('student_figure has axes')
            # If student_figure is not a figure object, exception thrown
            except AttributeError:
                assignment_logger.info('student_figure not a fig. object')
                print(
                    'student_figure is not a figure object. Ensure you have not reassigned it.')

    def check(self):
        """Check correct names are in global scope and notify student."""

        required_funcs = ['rhs', 'even_equation',
                          'odd_equation',
                          'find_energy']  # List of the requested function names given in the task
        # List of the requested variable names given in the task
        required_vars = ['solution_list']

        # Loop through the required functions and see if they are present.
        # Notify the student either way
        for func_name in required_funcs:
            if not hasattr(main, func_name):
                assignment_logger.info('{} does not exist.'.format(func_name))
                print('{} function is not correctly named.'.format(func_name))
            else:
                assignment_logger.info('{} exists.'.format(func_name))

                # Function name exists, now see if it is callable
                func = getattr(main, func_name)
                if callable(func):
                    assignment_logger.info(
                        '{} function callable.'.format(func_name))
                    print('{} function is correctly named.'.format(func_name))
                else:
                    assignment_logger.info(
                        '{} function not callable.'.format(func_name))
                    print(
                        '{} should be defined as a function, not a variable.'.format(func_name))

        # Similar to above, but for variables
        for var_name in required_vars:
            if not hasattr(main, var_name):
                assignment_logger.info('{} does not exist.'.format(var_name))
                print('{} variable is not correctly named.'.format(var_name))
            else:
                assignment_logger.info('{} exists.'.format(var_name))
                # N.B. main.var_name will throw AttributeError
                var = getattr(main, var_name)
                if callable(var):
                    assignment_logger.info(
                        '{} variable callable.'.format(var_name))
                    print('{} should be a variable, not a function.'.format(var_name))
                else:
                    assignment_logger.info(
                        '{} variable not callable.'.format(var_name))
                    print('{} variable is correctly named.'.format(var_name))

        self.check_plot()


class Boxes(Generator):
    """Child class of Generator, inherits allowed_IDs.

    This class contains functionality for the black boxes used in assignment 2.
    This must all be contained in one class and local variables used to prevent
    students from accessing circuit parameters.
    """

    @classmethod
    def get_boxes(cls, studentID):
        """Factory method (atl. constructor) for the Boxes class

        :param StudentID: ID given by student in assignment2.ipynb (studentID = XXXXXXX)
        :returns Tuple of 5 Boxes objects with different for all tasks if studentID is valid
                 Tuple of 5 None objects if studentID is invalid
        """

        # Initialise three Boxes objects using studentID
        initial_boxes = [cls(studentID), cls(studentID), cls(studentID)]

        # Set transfer_num attribute (see generator.py) to link the box 
        # with the correct transfer function and set task_num (determines 
        # RLC values in self.process)
        for i in range(3):
            # i=0: RL circuit, i=1: RC circuit, i=2: RLC circuit
            initial_boxes[i].transfer_num = initial_boxes[i].transfer_keys[i]
            initial_boxes[i].task_num = 1

        # Empty list to store shuffled Boxes
        return_boxes = []
        # Use keys (def. in generator2) to generate two rand ints - always same for 
        # given ID (consistent seed)
        first_rand = initial_boxes[0].key % 3  # 0, 1 or 2
        # 0, 1. Use second key else two rands are related
        second_rand = initial_boxes[0].key2 % 2

        # Fill return_boxes with a pseudo random ordering of initial_boxes elements
        # .pop(i) removes element indexed by i and returns it
        return_boxes.append(initial_boxes.pop(first_rand))
        return_boxes.append(initial_boxes.pop(second_rand))
        return_boxes.append(initial_boxes.pop())

        # Boxes for other tasks
        task2_box = cls(studentID)
        task2_box.transfer_num = return_boxes[0].transfer_keys[3]  # RL circuit
        task2_box.task_num = 2  # Task 2

        task3_box = cls(studentID)
        # RLC resonant circuit
        task3_box.transfer_num = return_boxes[0].transfer_keys[4]
        task3_box.task_num = 3  # Task 3

        return_boxes.extend([task2_box, task3_box])

        return tuple(return_boxes)

    @classmethod
    def get_example_box(cls):
        """Alternative constructor. Generates box for black_box_example.ipynb"""
        # Call __init__ with sentinel ID 0, set transfer and task number before returning
        example_box = cls(0)
        # example transfer function key
        example_box.transfer_num = example_box.transfer_keys[5]
        # Example task number
        example_box.task_num = 0
        return example_box

    @staticmethod
    def check_plot():
        """
        NB: Called by check function
        Checks whether student_figure has:

        1) been defined
        2) is a figure object (by ensuring get_axes works)
        3) has axes
        """
        # Check that student_figure is defined
        if not hasattr(main, 'student_figure'):
            print('student_figure is not defined.')
        else:
            # Now it is defined, try and extract axes from figure
            try:
                # .get_axes() returns empty list for no axes
                if not main.student_figure.get_axes():
                    print('student_figure has no axes. Ensure you are plotting '
                          'onto the student_figure object.')
            # If student_figure is not a figure object, exception thrown
            except AttributeError:
                print(
                    'student_figure is not a figure object. Ensure you have not reassigned it.')

    @classmethod
    def check(cls):
        """Check correct names are in global scope and notify student."""
        required_funcs = [
            'my_box_process']  # List of the requested function names given in the task
        required_vars = ['RL_circuit', 'RC_circuit', 'RLC_circuit',
                         'corner_frequency', 'inductance', 'resonant_frequency',
                         'bandwidth']  # List of the requested variable names given in the task

        # Loop through the required functions and see if they are present.
        # Notify the student either way
        for func_name in required_funcs:
            if not hasattr(main, func_name):
                print('{} function is not correctly named.'.format(func_name))
            else:
                # Function name exists, now see if it is callable
                func = getattr(main, func_name)
                if callable(func):
                    print('{} function is correctly named.'.format(func_name))
                else:
                    print(
                        '{} should be defined as a function, not a variable.'.format(func_name))

        # Similar to above, but for variables
        for var_name in required_vars:
            if not hasattr(main, var_name):
                print('{} variable is not correctly named.'.format(var_name))
            else:
                # N.B. main.var_name will throw AttributeError
                var = getattr(main, var_name)
                if callable(var):
                    print('{} should be a variable, not a function.'.format(var_name))
                else:
                    print('{} variable is correctly named.'.format(var_name))

        cls.check_plot()

        # NOTE: This block should come last due to return statements (if needed 
        # can wrap in while true and break) Check that RL_circuit etc. have a 
        # value of 1, 2 or 3 AND none of them have been given the same value
        if hasattr(main, 'RL_circuit') and hasattr(main, 'RC_circuit') and hasattr(main, 'RLC_circuit'):
            circuit_list = [main.RL_circuit, main.RC_circuit, main.RLC_circuit]
            valid_values = [num in [1, 2, 3] for num in circuit_list]
            if not all(valid_values):
                print('One or more of RL_circuit, RC_circuit, RLC_circuit are invalid. '
                      'These must be 1, 2 or 3.')
                return
            # Could remove this check if we want to allow a 33% guarantee
            if len(set(circuit_list)) != len(circuit_list):
                print('Two or more of RL_circuit, RC_circuit, RLC_circuit have the same values. '
                      'They must be different.')
                return
            print('RL_circuit, RC_circuit and RLC_circuit are assigned in a valid way.')

    def __init__(self, studentID):
        """Class constructor called by get_boxes and get_example_box

        Boxes objects should only be created via these alternative constructors
        """
        # transfer_num indicates which transfer function should be used:
        # 118 --> RL (task_2)
        # 173 --> RLC (task_3)
        # 145 --> Example
        # Set in get_boxes()
        self.transfer_num = None
        self.task_num = None
        # Call parent __init__ method
        super().__init__(studentID)

    def RL_transfer_func(self, freq_array, **kwargs):
        """Transfer function for RL high pass filter circuit

        Note: Use of kwargs means no problem if RL_transfer_func is also given 
        C as a parameter (as in Task 1):
            - makes it easy to add additional keyword arguments if necessary
            - kwargs may be a dictionary or and unpacked dictionary (i.e. pass my_dict or **my_dict)

        :param freq_array: Numpy Array containing frequencies on which to generate transfer function
        :param kwargs: Dictionary (unpacked - see above note) containing circuit Resistance, 
                       R, and Inductance, L
        :returns: Transfer function evaluated at points of freq_array (Numpy array).
                  None if R or C is not in kwargs
        """
        try:
            R = kwargs['R']
            L = kwargs['L']
        except KeyError as err:
            assignment_logger.warning(
                'RL_transfer_func:', repr(err), 'returning None.')
            return None

        # Array of angular frequencies
        omega = 2 * np.pi * freq_array

        # For RL high pass transfer function see useful_formulae.tex
        return (omega * L * 1j) / (R + omega * L * 1j)

    def RC_transfer_func(self, freq_array, **kwargs):
        """Transfer function for RC low pass filter circuit

        :param freq_array: Numpy Array containing frequencies on which to generate transfer function
        :param kwargs: Dictionary (unpacked) containing circuit Resistance, R, and Capacitance, C
        :returns: Transfer function evaluated at points of freq_array (Numpy array).
                  None if R or C is not in kwargs
        """
        try:
            R = kwargs['R']
            C = kwargs['C']
        except KeyError as err:
            assignment_logger.warning(
                'RC_transfer_func:', repr(err), 'returning None.')
            return None

        # Array of angular frequencies
        omega = 2 * np.pi * freq_array

        # For RC low pass transfer function see useful_formulae.tex
        return 1 / (1 + (omega * R * C) * 1j)

    def RLC_transfer_func(self, freq_array, **kwargs):
        """Transfer function for RLC band pass filter circuit

        The complex impedance of the circuit is firstly calculated
        It is important to avoid a divide by zero error for first value of frequency
        The transfer function tends to zero as frequency tends to zero

        :param freq_array: Numpy Array containing frequencies in Hz at which to 
               generate transfer function
        :param kwargs: Dictionary (unpacked) containing circuit Resistance, R, 
                       Inductance, L, and Capacitance, C
        :returns: Transfer function evaluated at points of freq_array (Numpy array).
                  None if R or C is not in kwargs
        """
        try:
            R = kwargs['R']
            L = kwargs['L']
            C = kwargs['C']
        except KeyError as err:
            assignment_logger.warning(
                'RLC_transfer_func:', repr(err), 'returning None.')
            return None

        # Array of angular frequencies
        omega = 2 * np.pi * freq_array

        # Mask 0 using floating point equality
        omega = ma.masked_values(omega, 0)
        # Note assignment: most numpy methods don't work in place

        # Output across resistor (See useful_formulae.tex)
        masked_return = R / (R + (omega * L - 1 / (omega * C)) * 1j)

        # Return array, filling masked value with 0 (transfer function is zero at freq. == 0)
        return ma.filled(masked_return, 0)

    def example_transfer_func(self, freq_array):
        """Example transfer function for black_box_example.ipynb

        Frequencies between 5 and 15 Hz will be passed, others excluded.

        :param freq_array: Numpy Array containing frequencies on which to generate transfer function
        :return: Transfer function evaluated at points of freq_array (Numpy array).
        """
        # Note: Don't actually need 2nd condition as: 'Portions not covered by any
        # condition have a default value of 0.'. But Explicit is better than implicit.
        # Note: in built abs() will use numpy.absolute() (__abs__() method) - as
        # freq_array is an ndarray

        return np.piecewise(freq_array, \
            [np.logical_and(5 <= abs(freq_array), abs(freq_array) <= 15), \
            np.logical_or(abs(freq_array) < 5, abs(freq_array) > 15)], [1, 0])

    def process(self, time_array, amplitude_array):

        # Check that the inputs are real numpy arrays
        if not isinstance(time_array, np.ndarray):
            assignment_logger.info('time_array not numpy array.')
            print("Error: time_array is not a numpy array.")
            return
        if not isinstance(amplitude_array, np.ndarray):
            assignment_logger.info('amplitude_array not numpy array.')
            print("Error: amplitude_array is not a numpy array.")
            return
        if not np.all(np.isreal(time_array)):
            assignment_logger.info('time_array not wholly real.')
            print("Error: Wait until QFT in fourth year to introduce imaginary time!")
            return

        # Check that time samples are equally spaced
        intervals = np.diff(time_array)
        first_interval_array = np.ones(intervals.size) * intervals[0]
        if not np.allclose(intervals, first_interval_array, rtol=0, atol=1e-12):
            assignment_logger.info('Sampling rate not uniform.')
            print("Error: the sampling rate is not uniform.")
            return

        # Check that the arrays are of the same length
        n_time_samples = time_array.size
        n_amp_samples = amplitude_array.size
        if n_time_samples != n_amp_samples:
            assignment_logger.info('time_array.size != amplitude_array.size')
            print("Error: The number of time and amplitude samples must be equal.")
            return

        # Perform FFT on input data
        transformed_signal = np.fft.fft(amplitude_array)

        # Define quantities RLC here so that they are not accessible to the student
        # N.B. As not object attributes these will have to be generated in exactly 
        # the same way when marking

        # Additional keyword params (stored in dictionary), passed to selected_transfer (unpacked)
        t_func_params = {}
        # Empty by default: For example_transfer_func ('Task 0') which takes no additional args.
        # N.B. Must unpack t_func_params (otherwise attempts to pass dictionary as a positional 
        # arg.)

        # Task 1: All three of R, L, C required
        if self.task_num == 1:
            R = (self.MIN_RESISTANCE +
                 (self.key / self.norm) * (self.MAX_RESISTANCE - self.MIN_RESISTANCE))
            L = (self.MIN_INDUCTANCE +
                 (self.key / self.norm) * (self.MAX_INDUCTANCE - self.MIN_INDUCTANCE))
            C = (self.MIN_CAPACITANCE +
                 (self.key / self.norm) * (self.MAX_CAPACITANCE - self.MIN_CAPACITANCE))
            t_func_params = {'R': R, 'L': L, 'C': C}
        # Task 2: Capacitance not required (Could set to None; and have C as a parameter of RL)
        elif self.task_num == 2:
            R = 200  # Value of Ohms chosen for this task
            L = (self.MIN_INDUCTANCE +
                 (self.key2 / self.norm) * (self.MAX_INDUCTANCE - self.MIN_INDUCTANCE))
            t_func_params = {'R': R, 'L': L}
        # Task 3: All three required again
        elif self.task_num == 3:
            R = (self.MIN_RESISTANCE +
                 (self.key3 / self.norm) * (self.MAX_RESISTANCE - self.MIN_RESISTANCE))
            L = (self.MIN_INDUCTANCE +
                 (self.key3 / self.norm) * (self.MAX_INDUCTANCE - self.MIN_INDUCTANCE))
            C = (self.MIN_CAPACITANCE +
                 (self.key3 / self.norm) * (self.MAX_CAPACITANCE - self.MIN_CAPACITANCE))
            t_func_params = {'R': R, 'L': L, 'C': C}

        # Determine sample points in f_domain using fftfreq (Note: Not angular frequency!)
        timestep = intervals[0]  # Student sample spacing in t domain
        freq_array = np.fft.fftfreq(n_time_samples, d=timestep)

        # Link the keys with the correct transfer function so that transfer_num
        # corresponds to the correct transfer function. See generator.py for explanation
        transfer_functions = \
            {self.transfer_keys[0]: self.RL_transfer_func, \
             self.transfer_keys[1]: self.RC_transfer_func, \
             self.transfer_keys[2]: self.RLC_transfer_func, \
             self.transfer_keys[3]: self.RL_transfer_func, \
             self.transfer_keys[4]: self.RLC_transfer_func, \
             self.transfer_keys[5]: self.example_transfer_func}
        # Apply relevant transfer function to transformed_signal (multiplication in f_domain)
        modified_f_domain_signal = transformed_signal * \
            transfer_functions[self.transfer_num](freq_array, **t_func_params)

        # Check modified_f_domain_signal is a numpy array ('None' if correct 
        # parameters were not present)
        if not isinstance(modified_f_domain_signal, np.ndarray):
            assignment_logger.warning(
                'modified_f_domain_signal not numpy array.')
            return  # Returns None to caller

        # Perform inverse FFT on the modulated input signal
        output_signal = np.fft.ifft(modified_f_domain_signal)

        return output_signal.real

#########################################################################
#########################################################################
#########################################################################

class Rocket(Generator):
    """
    Generates student's rocket and calculates its successive positions 
    during each 'run' (landing attempt)
    """

    def __init__(self, studentID):
        # Call parent __init__ to initialise several rocket parameters. These are:
        # self.__z, self.__y, self.left_thrust, self.right_thrust, self.dt and
        # self.platform_x_pos_list
        # If ID is invalid remaining parameters (e.g. self.flight_data) are not intialised
        super().__init__(studentID)
        self.__z = self._Generator__z
        self.__y = self._Generator__y
        self.__x = self._Generator__x
        self.__w = self._Generator__w
        self.__v = self._Generator__v
        self.__u = self._Generator__u

        # First platform position (at sea level)
        self.platform_pos = np.array([self.platform_x_pos_list[0], 0])

        # Counters for number of successful landsings and total number of flights
        self.flight_counter = 0
        self.successful_landing_counter = 0

        self.reset_required = True

        self.perlin = Perlin()
        random.seed(self.key)

        # get the rocket image
        #fn = os.path.join(os.path.dirname(__file__), 'student_rockets.zip')
        #rocket_images = load_zipped_pickle(fn)
        #self.svg = rocket_images[studentID]

        #print(studentID)

        # Call reset to set rocket position, counter etc
        self.reset()

        # No previous flight hence flight_data is [0, 0, 0] (starting time & x, y p
        # ositions - see self.get_flight_data())
        # self.flight_data = np.array([[self.time, self.rocket_pos[0], self.rocket_pos[1]]])

    #def show(self):
        #return rocket image for display
        #return self.svg

    def get_platform_pos(self):
        """
        Return the platform position. Use this to avoid encouraging
        students to access object properties.

        :return Platform's position in the form np.ndarray([x_pos, y_pos])
        """
        return self.platform_pos

    def get_init_pos(self):
        """
        Return the initial position of the rocket. Use this to avoid encouraging
        students to access object properties.

        :return Rocket's initial position in the form np.ndarray([x_pos, y_pos])
        """
        return self.rocket_pos_init

    def is_in_bounds(self):
        """
        A getter to interface with the .reset_required property. 
        """
        if self.gravity:
            return (self.rocket_pos > self.SCREEN_MIN).all() and (self.rocket_pos < self.SCREEN_MAX).all()
        else:
            return (self.rocket_pos > self.SPACE_MIN).all() and (self.rocket_pos < self.SPACE_MAX).all()

    def reset_flight_counter(self):
        self.flight_counter = 0
        self.successful_landing_counter = 0

    def get_flight_data(self):
        """
        Return information of most recent run (landing attempt) to student, for analysis

        :return flight_data: 2 dimensional ndarray of shape (X, 3) where X is the number 
                             of steps in previous run The 3 elements of each row (axis 1) 
                             are time (s), current x position (m) & current y position (m)
                             These are rounded to POSITION_RESOLUTION decimal place(s) (We 
                             still have access to un-rounded data)

        Note: Further elements could be added to each row e.g. wind speed at that time
        """
        #print(self.flight_data)

        try:
            if self.flight_data.size == 3:
                # No previous flight; notify student
                print("You have not called .advance() on this rocket object yet. "
                      "This function is returning the initial flight data.")
                # Could add e.g. rounding or checking here (no 'out of bound' values etc.)
                # Also, could make it so student can only get flight data in this way AFTER each run
                # has been completed (they
                # would still be able to get position after every step from self.advance())
            self.reset_required = True
            return np.around(self.flight_data, decimals=self.POSITION_RESOLUTION)
        except AttributeError:
            print("Please initialise your rocket before trying to get flight data. "
                  "This function is returning None.")
            return None

    def reset(self, mode="space"):
        """Reset the rocket and move platform ready for next run

        Clears flight data and sets initial time (0), velocity (0, 0) & position (...)
        """

        # # Firstly get index of current platform position and increment it
        # old_platform_index = self.platform_x_pos_list.index(self.platform_pos)
        # new_platform_index = old_platform_index + new_platform_index
        # If index exceeds len(self.platform_x_pos_list), set to 0 (begin cycling t
        # hrough positions again)
        # if new_platform_index > (len(self.platform_x_pos_list - 1)):
        #     new_platform_index = 0

        # More efficient method (Does rely on no funny business with self.flight_counter)
        new_platform_index = self.flight_counter % len(
            self.platform_x_pos_list)  # Number between 0 and (len() - 1)
        # e.g. after first 'flight' self.flight_counter -> 1 and reset is called, 
        # so new_platform_index -> 1
        self.platform_pos = np.array(
            [self.platform_x_pos_list[new_platform_index], 0])
        self.mode = mode

        self.rocket_pos = np.array(
            [(self.SCREEN_MAX[0] - self.SCREEN_MIN[0]) / 2, self.SCREEN_MAX[1]], dtype=float)
        if mode == "drop" or mode == "landing":
            self.gravity = self.GRAVITATIONAL_ACC
            terminal_vel = -0.6 * self.__z * \
                self.GRAVITATIONAL_ACC / self.__y[1]
            self.rocket_velocity = np.array([0.0, terminal_vel])
        elif mode == "space":
            self.gravity = 0
            self.rocket_pos = np.array([0.0, 0.0])
            self.rocket_velocity = np.array([0.0, 0.0])
        else:
            print(" .reset() can be used with mode 'space', 'drop' or 'landing'")
            return
        self.rocket_pos_init = self.rocket_pos
        self.time = 0.0
        self.flight_data = np.array(
            [[self.time, self.rocket_pos[0], self.rocket_pos[1]]])

        self.offset = random.randint(0, 1000)
        self.reset_required = False
        self.crashed = False
        self.fuel = self.max_fuel

    def landed_successfully(self):
        """
        Determines whether the rocket landed safely on the platform (assumes 
        Rocket has reached sea level)

        TODO (possibly): Return False if rocket lands too 'heavily' - this would 
        require an additional upwards thruster

        :return bool; True if rocket landed on platform, False otherwise
        """
        # Assumed to be between 0 and self.SCREEN_MAX
        platform_centre = self.platform_pos[0]
        # x position of left & right edges of platform (must be 'on screen' i.e. 
        # between 0 and self.SCREEN_MAX)
        left_edge = platform_centre - self.PLATFORM_WIDTH / 2
        right_edge = platform_centre + self.PLATFORM_WIDTH / 2

        # Check whether current rocket x pos. is between left and right edges of platform
        if left_edge < self.rocket_pos[0] < right_edge:
            if self.mode == "drop":
                return True
            elif self.mode == "landing":
                if all(np.abs(self.rocket_velocity) < self.LANDING_TOLERANCE):
                    return True
                elif self.rocket_pos[1] < self.SCREEN_MIN[1]:
                    self.crashed = True
        # else:  # A redundant else - I do wonder whether it will be removed?
        # Yes it will
        return False

    def advance(self, student_left_thrust_command,
                student_right_thrust_command, student_landing_thrust_command=0):
        """
        Advance position of rocket by self.TIMESTEP (1 / 60 s) using the Euler Method

        Rocket is subject to gravitational & drag forces in the vertical direction, and 
        thruster & drag forces in the horizontal direction (Newtons)- see useful_formulae.tex 
        for equation of motion. These forces are assumed constant over each interval. The 
        step size is self.COMPUTATION_TIMESTEP.

        Flight ends when sea level is reached (y == 0) or rocket goes out of bounds 
        (e.g. x < 0 || x > 200 || y > 200) Firstly check that a reset of the rocket is not 
        required

        :param student_left_thrust_command: Left thrusting force desired by student (to generate 
        rightward i.e. positive force)
            - Actual thrusting force is capped at self.max_left_thrust (defined in generator.py)
        :param student_right_thrust_command: Right thrusting force desired by student
        (to generate leftward i.e. negative force)
            - Actual thrusting force is capped at self.max_right_thrust (defined in generator.py)


        :return: [x,y] position relative to top left hand corner. In metres and rounded
        to self.POSITION_RESOLUTION d.p.
        """

        #print(self.rocket_pos)
        if self.reset_required:
            return np.around(self.rocket_pos, decimals=self.POSITION_RESOLUTION)
            #return None

        # Actual thrusting force is max. of force dictated by student & the max. thrust 
        # the each engine can produce
        # If student gives negative thrust, then apply zero force
        # N.B. self.max_left_thrust and self.max_right_thrust are currently visible to 
        # student (dir() exploit)
        #left_thrust = max(0, min(student_left_thrust_command, self.max_left_thrust))
        #right_thrust = max(0, min(student_right_thrust_command, self.max_right_thrust))
        left_thrust = np.array(
            student_left_thrust_command - self.__w).clip(min=0.0, max=self.__x)
        right_thrust = np.array(
            student_right_thrust_command - self.__v).clip(min=0.0, max=self.__x)
        up_thrust = np.array(student_landing_thrust_command).clip(
            min=0.0, max=self.__u)
        #print("Thrust {} {}".format(left_thrust, right_thrust))

        # Position is to be returned after a time self.TIMESTEP
        t_final = self.time + self.TIMESTEP
        wind_speed = 100 * \
            self.perlin.Sum(self.offset + t_final, 0.2, 2, 1.05, 1.3)

        while self.time < t_final:
            # Vertical acceleration (Gravitational & air resistance)
            # Horizontal acceleration (Net thruster & air resistance)
            if self.gravity:
                wind_acc = self.__y[0] * \
                    (self.rocket_velocity[0] - wind_speed) / self.__z
                a_vert = - self.gravity - \
                    (self.__y[1] * self.rocket_velocity[1]) / self.__z
                a_horiz = (left_thrust - right_thrust) / self.__z - wind_acc
                #print("Fuel left: {}".format(self.fuel))
                if self.fuel > 0:
                    a_vert += up_thrust / self.__z
                    self.fuel -= self.dt * up_thrust
            else:
                a_vert = 0.0
                a_horiz = (left_thrust - right_thrust) / self.__z

            #print("a_horiz {}".format(a_horiz))

            # Save starting rocket position; used to calculate avg. velocity
            # Copy to avoid this changing when self.rocket_pos is modified
            rocket_pos_before = copy.copy(self.rocket_pos)
            # calculate new velocity and position
            self.rocket_velocity += np.array([a_horiz, a_vert]) * self.dt
            self.rocket_pos += self.rocket_velocity * self.dt
            # Increment time (currently self.dt = self.TIMESTEP / 60)
            self.time += self.dt
            # Flight ends if rocket reaches sea level or goes out of bounds. In either case must
            # update flight counter and reset flag and break from this computation loop
        finished = 0
        if not self.is_in_bounds():
            self.flight_counter += 1
            #print("Flight ended, call .reset()")
            self.reset_required = True

            if self.gravity:
                if self.landed_successfully():
                    self.successful_landing_counter += 1
                    finished = 2
                    print('The rocket lands on the platform.\n{} / {} flights have '
                          'resulted in a successful landing.\n'
                          'Call .reset("drop") before attempting '
                          'a new flight.\n'.format(self.successful_landing_counter,
                                                   self.flight_counter))
                else:
                    finished = 1
                    if self.crashed:
                        if np.abs(self.rocket_velocity[1]) > self.LANDING_TOLERANCE[1]:
                            print("The rocket crashed into the platform!")
                        else:
                            print(
                                "The rocket landed on the platform but toppled over!")
                    else:
                        print("The rocket did not hit the platform.")
                    print("{} / {} flights have resulted in a successful landing.\n" \
                          "Call .reset('drop') before attempting a new " \
                          "flight.\n".format(self.successful_landing_counter,
                                             self.flight_counter))
                self.reset_required = True
            else:
                finished = 1
                self.reset_required = True
                print(
                    "Flight ended (rocket outside of bounds).\n"
                    "Call .reset() before attempting a new flight.\n")

        # Move the platform
        if self.gravity:
            dpos = np.random.normal(0.0, 0.2, 1)[0]
            xy = np.array([dpos, 0])
            zz = np.add(self.platform_pos, xy)
            self.platform_pos = zz

        # Update flight_data with current time and (un-rounded) position
        # np.append does not append in place. Shape of array added (1, 3) and axis specification are crucial
        self.flight_data = np.append(
            self.flight_data, [[self.time, self.rocket_pos[0], self.rocket_pos[1]]], axis=0)
        #print(self.flight_data)
        # Return new position, rounded
        rounded = np.around(self.rocket_pos, decimals=self.POSITION_RESOLUTION)
        return [rounded[0], rounded[1], finished]


#import numpy as np

class Perlin(object):
    def __init__(self):

        self.hash = [151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233,  7, 225,
                     140, 36, 103, 30, 69, 142,  8, 99, 37, 240, 21, 10, 23, 190,  6, 148,
                     247, 120, 234, 75,  0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
                     57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
                     74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
                     60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
                     65, 25, 63, 161,  1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
                     200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186,  3, 64,
                     52, 217, 226, 250, 124, 123,  5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
                     207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
                     119, 248, 152,  2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172,  9,
                     129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
                     218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
                     81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
                     184, 84, 204, 176, 115, 121, 50, 45, 127,  4, 150, 254, 138, 236, 205, 93,
                     222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
                     151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233,  7, 225,
                     140, 36, 103, 30, 69, 142,  8, 99, 37, 240, 21, 10, 23, 190,  6, 148,
                     247, 120, 234, 75,  0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
                     57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
                     74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
                     60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
                     65, 25, 63, 161,  1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
                     200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186,  3, 64,
                     52, 217, 226, 250, 124, 123,  5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
                     207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
                     119, 248, 152,  2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172,  9,
                     129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
                     218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
                     81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
                     184, 84, 204, 176, 115, 121, 50, 45, 127,  4, 150, 254, 138, 236, 205, 93,
                     222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180]
        self.hashMask = 255
        self.gradients1D = [1.0, -1.0]
        self.gradientsMask1D = 1

    def lerp(self, A, B, factor):
        """interpolates A and B by factor"""
        return (1.0 - factor) * A + factor * B

    def Smooth(self, t):
        # t: float
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def Perlin1D(self, point, frequency):
        # point = float, frequcncy = float
        # int: i0, i1
        # float t0, t1, g0, g1, v0, v1, t
        my_int = np.vectorize(int)
        my_float = np.vectorize(float)
        point *= frequency
        i0 = my_int(np.floor(point))
        t0 = my_float(point - i0)
        t1 = my_float(t0 - 1.0)
        i0 &= self.hashMask
        i1 = i0 + 1

        g0 = self.gradients1D[self.hash[i0] & self.gradientsMask1D]
        g1 = self.gradients1D[self.hash[i1] & self.gradientsMask1D]

        v0 = g0 * t0
        v1 = g1 * t1

        t = self.Smooth(t0)
        return self.lerp(v0, v1, t) * 2.0

    # float: point, frquency, lacunricy, persistamce; int: octaves
    def Sum(self, point, frequency, octaves, lacunarity, persistence):
        sum = self.Perlin1D(point, frequency)
        amplitude = 1.0
        range = 1.0
        for o in np.arange(octaves):
            frequency *= lacunarity
            amplitude *= persistence
            range += amplitude
            sum += self.Perlin1D(point, frequency) * amplitude
        return sum / range
