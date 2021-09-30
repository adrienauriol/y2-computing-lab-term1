# -*- coding: utf-8 -*-
import os
import sys
import warnings
import numpy as np
import scipy.constants as constants
#import csv
import hashlib, binascii
import pickle
import gzip


class Generator:
    """Creates pseudorandom numbers used to generate parameters from student IDs. Container for assignment constants.

    Parent to Assignment1, Solution1, Boxes and Solution2 classes.

    Class Variables
    ---------------
    allowed_IDs : tuple of ints
                All IDs of the yr2 students in the module.
                Used to generate the parameters for a given student.

    Assignment 1 Constants:

    MIN_WIDTH : float
            Minimum permitted width of well which may occur in problem.
            Units : Metres
    MAX_WIDTH : float
            Maximum permitted width of well which may occur in problem.
            Units : Metres
    MIN_POTENTIAL : float
            Minimum permitted potential which may occur in problem.
            Units : eV
    MAX_POTENTIAL : float
            Minimum permitted potential which may occur in problem.
            Units : eV

    Assignment 2 Constants:

    MIN_RESISTANCE : float
            Minimum value of the resistance in Ohms.

    MAX_RESISTANCE : float
            Maximum value of the resistance in Ohms.

    MIN_CAPACITANCE : float
            Minimum value of the capacitance in Farads

    MAX_CAPACITANCE : float
            Maximum value of the capacitance in Farads

    MIN_INDUCTANCE : float
            Minimum value of the inductance in Henrys

    MAX_INDUCTANCE : float
            Maximum value of the inductance in Henrys


    Assignment 3 Constants:
    -----------------------

    MIN_MASS : int
            Minimum mass of the rocket in kilograms.

    MAX_MASS : int
            Maximum mass of the rocket in kilograms.

    MIN_DRAG_X : float
            Minimum value of drag coefficient (Air resistance propto vel. squared)

    MAX_DRAG_X : float
            Maximum value of drag coefficient (Air resistance propto vel. squared)
    DRAG_Y

    Each thurster will have an offset and maximim so that the real force
    is computed from the user input as: input.clip(min=offset, max=maximum)

    MIN_THRUST : float
    MAX_THRUST : float
            Range for maximum
    MIN_OFFSET : float
    MAX_OFFSET : float
            Range for minimum

    TIMESTEP : float
            Time (seconds) between returns (successive rocket position) of Rocket.advance()

    COMPUTATION_TIMESTEP : float
            Step size for Forward Euler Method in Rocket.advance(). This is smaller than ROCKET_TIMESTEP for more
            accurate computation of rocket positions


    GRAVITATIONAL_ACC : float
            Gravitational acceleration on Earth in ms^-2 (assumed constant 9.81 ms^-2)

    PLATFORM_WIDTH : int
            Width of platform in metres (Wider platform makes for an easier landing)

    POSSIBLE_PLATFORM_POSITIONS : tuple of ints
            List of (9) possible platform positions in metres in ascending order.
            End elements should satisfy:
                POSSIBLE_PLATFORM_POSITIONS[0] > (0 + PLATFORM_WIDTH / 2)
                POSSIBLE_PLATFORM_POSITIONS[-1] < (SCREEN_MAX_X - PLATFORM_WIDTH / 2)
            So that the entirely of the platform appears on the screen.
            Each student will receive a shuffled list version of this tuple - see below.

    POSITION_RESOLUTION : int
            'Resolution' of position sensors of rocket, in number of decimal places (e.g. 1 -> round to nearest 0.1m)
            This can be negative (-1 -> nearest 10m etc.)

    For 'drop' mode:
    SCREEN_MAX: [float,float]
            x,y position (in metres) defining theupper right corner of the screen

    SCREEN_MIN: [float,float]
            x,y position (in metres) defining the lower left corner the screen
    For 'space' mode:
    SPACE_MAX: [float,float]
    SPACE_MIN: [float,float]

    """

    # ~~~~~~~~~~~~~ Allowed IDs ~~~~~~~~~~~~~ #
    # TODO remove below after test
    # reading file exported from official excel spreadhsset (on ID per line)
    #fn = os.path.join(os.path.dirname(__file__), 'Student_IDs.csv')
    #with open(fn) as f:
    #    lines = f.read().splitlines()
    #f.close()
    # skipping empty lines and converting to int
    #allowed_IDs =[int(e.strip()) for e in lines if e]

    # Assignment 1
    MIN_WIDTH = 1.5e-9  # Width in metres
    MAX_WIDTH = 2e-9  # Maximum will actually be 1.995
    MIN_POTENTIAL = 2.0  # Potential in eV
    MAX_POTENTIAL = 3.0

    # Assignment 2 Task 1 Cut-off/resonant frequencies (Hz)
    # RL: Min = 265 Max = 637
    # RC: Min = 248 Max = 637
    # RLC: Min = 324 Max = 503
    MIN_RESISTANCE = 50
    MAX_RESISTANCE = 80

    MIN_CAPACITANCE = 5e-06
    MAX_CAPACITANCE = 8e-06

    MIN_INDUCTANCE = 2e-02
    MAX_INDUCTANCE = 3e-02

    # Assignment 2 Task 2 (RL high pass)
    # Use the same idea and bounds as task1 but use key2 in process
    # R = 200 set in .process in
    # Cut-off frequency (Hz) (rough)
    # Min = 1060 Max = 1590

    # Assignment 3 Constants
    R_A = 152.5
    R_B = 10
    R_C = 11
    R_D = (R_C - R_B) * 4 + R_B # = 14
    R_E = R_D - R_B # = 4
    
    # MIN_MASS = 1650
    R_F = (R_A - R_B / R_E) * R_C
    # MAX_MASS = 2170
    R_G = (R_A + R_B / R_E) * R_D
    # DRAG_X = 110.0
    R_H = R_B * R_C
    # DRAG_Y = 50.0
    R_I = (R_C - R_B + R_E) * R_B
    # MIN_THRUST = 7200.0
    R_J = R_B * (R_A * R_E + R_B * R_C)
    # MAX_THRUST = 11100.0
    R_K = R_B**2 * (R_B * R_C + R_C - R_B)
    # MIN_OFFSET = 283.0
    R_L = R_D * (R_B + R_C) - R_C
    # MAX_OFFSET = 820.0
    R_M = R_B * R_D * (R_E + 2 * (R_C - R_B)) - 2 * R_B

    TWR = 5 # Thrust-to-Weight ratio

    LANDING_TOLERANCE = np.array([1, 5])

    MAX_THRUST_TIME = 7

    TIMESTEP = 1 / 60.0

    COMPUTATION_TIMESTEP = TIMESTEP / 60.0

    GRAVITATIONAL_ACC = 9.81

    PLATFORM_WIDTH = 20.0

    POSSIBLE_PLATFORM_POSITIONS = (40, 50, 80, 90, 100, 110, 120, 150, 160)

    POSITION_RESOLUTION = 10

    SCREEN_MIN = np.array([0.0,0.0])
    SCREEN_MAX = np.array([200.0, 2000])
    SPACE_MIN  = np.array([-20000.0,-20000.0])
    SPACE_MAX  = np.array([ 20000.0, 20000.0])

    def in_interval(self, min_value, max_value, pos):
        return min_value + pos * (max_value - min_value)

    # tools for hashing and bit-based indices. From rocket_procedure.py
    def hash32(self, ID):
        """Returns a unique int to int mapping with a pseudorandom
        distribution, see http://burtleburtle.net/bob/hash/integer.html
        """
        warnings.filterwarnings('ignore')
        a = np.uint32(ID)
        a -= (a << np.uint32(6))
        a ^= (a >> np.uint32(17))
        a -= (a << np.uint32(9))
        a ^= (a << np.uint32(4))
        a -= (a << np.uint32(3))
        a ^= (a << np.uint32(10))
        a ^= (a >> np.uint32(15))
        warnings.filterwarnings('default')
        return a

    def triple_hash(self, ID):
        """Calls hash32 three times for more pseudo randomness"""
        return self.hash32(self.hash32(self.hash32(ID)))

    def limited_hash(self, key, startbit, endbit):
        """Takes an input number (key), returns the number resulting from
        taking the bits from startbit -> endbit, numbered like an array.

        eg.
        limited_hash(93, 2, 5) -> 7
        """
        newkey = key >> startbit
        newkey = newkey & (np.power(2, endbit - startbit + 1) - 1)
        return newkey

    @classmethod
    def check_ID(cls, studentID):
        """Class method to inform the student whether their student ID was valid.

        Called in Assignment1.__init__() and Boxes.get_boxes()

        :param StudentID: ID given by student in assignment2.ipynb (studentID = XXXXXXX)
        :returns    True if the studentID is valid
                    False if the studentID is not valid
        """
        studentIDNumber = int(str(studentID))
        # If a student has problems entering their ID i.e. it's not in allowed_IDs and should be,
        # then demonstrator enters 0 as a 'magic input' to get them going and records their
        # actual ID to add to allowed_IDs.
        try:
            if studentIDNumber == 0:
                print('*** Using a temporary ID. Switch to your own ID as soon as possible. ***\n')
                return True
            #elif studentIDNumber in cls.allowed_IDs:
            elif checkID(studentIDNumber, 'student_IDs.gz'):
                print('The student ID is valid.\n')
                return True
            else:
                raise ValueError
        except ValueError:
            print('*** Student ID not found. Double-check the ID or notify a demonstrator. ***\n')
            #sys.exit(1)
            return False

        return True

    def __init__(self, studentID):
        self.ID = studentID

        if (self.check_ID(self.ID)):
            self.validID = True
        else:
            self.validID = False

        # Number of bits taken from hash
        n = 10
        # Add to self.ID before generating keys in case it is 0.
        # Multiplication of self.ID would give 0 for all keys
        self.key = self.limited_hash(self.triple_hash(self.ID + 1), 0, n - 1)
        self.key2 = self.limited_hash(self.triple_hash(self.ID + 2), 0, n - 1)
        self.key3 = self.limited_hash(self.triple_hash(self.ID + 3), 0, n - 1)
        self.key4 = self.limited_hash(self.triple_hash(self.ID + 4), 0, n - 1)
        self.key5 = self.limited_hash(self.triple_hash(self.ID + 5), 0, n - 1)
        self.key6 = self.limited_hash(self.triple_hash(self.ID + 6), 0, n - 1)
        self.norm = 2 ** n  # Max is really 2^n - 1
        self.factor = self.key/self.norm

        # ~~~~~~~~~~ Assignment 1 Parameters ~~~~~~~~~~ #
        # Well width in m
        self.width = self.in_interval(self.MIN_WIDTH, self.MAX_WIDTH, self.factor)

        # Well depth in eV
        self.potential = self.in_interval(self.MIN_POTENTIAL, self.MAX_POTENTIAL, self.factor)

        # Working in SI units
        self.lambda_0 = (constants.m_e * (self.width**2) * self.potential *
                         constants.e) / (2 * constants.hbar**2)

        # ~~~~~~~~~~ Assignment 2 Specific Code ~~~~~~~~~~ #
        # Boxes object has the property `transfer_num` which links it to a given
        # transfer function and therefore defines its inner component.
        # This is visible to the inquisitive student through dir().
        # To prevent students realising the link between this and the contained component, add
        # a layer of scrambling which leads to a given transfer function corresponding to a
        # different transfer_num for different students.

        # self.transfer_keys is such that self.transfer_keys[i] should be the value that
        # self.transfer_num should take if the box wants to have the ith inner component.
        # Where i=0: Task1 RL, i=1: Task1 RC, i=2: Task1 RLC,
        # i=3: Task2 RL, i=4: Task3 RLC, i=5: Example notebook transfer func.
        # Note that Task1 and Task2 boxes need different transfer_num values to prevent
        # student using the Task2 RL box to know the number of the Task1 RL box.
        self.transfer_keys = []

        # i=0,1,2 entries of self.transfer_keys are the last digits of the generated keys above
        # (corrected to be unique if necessary)
        last_digits = [int(str(self.key)[-1]), int(str(self.key2)[-1]), int(str(self.key3)[-1])]
        for digit in last_digits:
            while True:
                if digit in self.transfer_keys:
                    digit += 1
                else:
                    break
            self.transfer_keys.append(digit)

        # Now add some arbitrary numbers to correspond to the Task2, Task3 and example boxes
        self.transfer_keys.extend([118, 173, 145])

        # ~~~~~~~~~~ Assignment 3 Parameters ~~~~~~~~~~ #
        #print("Key: {}, Norm {}, factor {}".format(self.key, self.norm, self.key/self.norm))
        self.__z = self.in_interval(self.R_F, self.R_G, self.factor) # self.mass = in_interval(self.MIN_MASS, self.MAX_MASS, self.factor)
        self.__y = np.zeros(2)  # self.drag = np.zeros(2)
        self.__y[1] = self.R_I  # self.drag[1] = self.DRAG_Y
        self.__y[0] = self.R_H  # self.drag[0] = self.DRAG_X
        self.__x = self.in_interval(self.R_J, self.R_K, self.factor) # self.max_thrust  = in_interval(self.MIN_THRUST, self.MAX_THRUST, self.factor)
        self.__w = self.in_interval(self.R_L, self.R_M, self.key3 / self.norm)   # self.offset_left  = in_interval(self.MIN_OFFSET, self.MAX_OFFSET, self.key3/self.norm)
        self.__v = self.in_interval(self.R_L, self.R_M, self.key4 / self.norm)   # self.offset_right = in_interval(self.MIN_OFFSET, self.MAX_OFFSET, self.key4/self.norm)
        self.__u = self.TWR * self.GRAVITATIONAL_ACC * self.__z # max v-thrust
        self.max_fuel = self.__u * self.MAX_THRUST_TIME # total amount of fuel
        self.dt = self.TIMESTEP

        #if True:
        #if False:
            #print("--- Debug info, will not be shown to the students ---")
            #print("mass: {} kg".format(self.__z))
            #print("drag: x {} y {} kg/s".format(self.__y[0], self.__y[1]))
            #print("max_thrust: {} N".format(self.__x))
            #print("offset_left:  {} N".format(self.__w))
            #print("offset_right: {} N".format(self.__v))
            #print("max_landing_thrust: {} N".format(self.__u))
            #print("max_fuel: {} m/s".format(self.max_fuel / self.__z))


        # Generate 9 pseudo random numbers using keys to shuffle platform positions
        self.platform_x_pos_list = []  # To hold shuffled platform x positions
        platform_pos_list = list(self.POSSIBLE_PLATFORM_POSITIONS)  # Temporary list
        reducing_modulo = len(platform_pos_list)  # 9; used as a divisor in below loop
        while reducing_modulo > 0:
            # Use hashing functions to generate a pseudo rand int between 0 and 2^n - 1 (n = 10)
            pseudo_rand = self.limited_hash(self.triple_hash(self.ID + reducing_modulo), 0, n - 1)
            # Use modulo op. to get random int between 0 & Max; Max is current largest index of possible_platform_list
            rand_index = pseudo_rand % reducing_modulo  # Result: 0 - (reducing_modulo - 1) ***
            # Pop element indexed by rand_index off platform_pos_list and append to self.platform_x_pos_list
            self.platform_x_pos_list.append(platform_pos_list.pop(rand_index))
            # Note: At this point in final iteration: reducing_modulo == 1, rand_index == 0
            # Reduce reducing_modulo for next iteration (length of platform_pos_list has decreased by 1)
            reducing_modulo -= 1

def hashID(ID, mysalt):
    myhash = binascii.hexlify(hashlib.pbkdf2_hmac('sha256', bytes(ID), bytes(mysalt,'utf-8'), 100000))   
    return str(myhash,'utf-8')

def checkID(ID, filename):
    fn = os.path.join(os.path.dirname(__file__), filename)
    hashes = read_zipped_pickle(fn)
    for line in hashes:
        myhash, mysalt = line.split(':')
        current_student_hash = hashID(ID, mysalt)
        #print(str(ID)+" current "+ current_student_hash +" : "+mysalt)
        #print(str(ID)+" in-file "+ myhash               +" : "+mysalt)
        #print(current_student_hash, myhash)
        if current_student_hash == myhash:
            return True
    return False

def read_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
