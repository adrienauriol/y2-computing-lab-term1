# -*- coding: utf-8 -*-
"""
This file is responsible for the automatic marking of Assignments 1, 2 and 3.
Each Solution class works with a .py file (converted_file) that was converted and parsed
by a Convert class (convert.py).
This .py is imported, loading relevant student-defined objects and variables for marking.
"""

from .generator import Generator
import numpy as np
import numpy.ma as ma  # For masking in RLC_transfer_func
from scipy import constants
from scipy import optimize
import logging
import math
import os
import sys
import inspect  # Get details of student functions (eg number of args)
import importlib.util  # To import the converted student file using an absolute path
import datetime
import textwrap

class WrappedFixedIndentingLog(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', width=70, indent=4):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.wrapper = textwrap.TextWrapper(width=width, initial_indent='', \
        subsequent_indent=' '*indent, replace_whitespace=True)

    def format(self, record):
        return self.wrapper.fill(super().format(record))


class Solution(Generator):
    """
    Parent class of Solution1, 2 and 3 which initialises the relevant logging,
    mode and properties
    """

    def __init__(self, ID=None, converted_file=None):
        """
        :param converted_file: Convert1, Convert2 or Convert3
        object corresponding to the converted student file
        :param ID: Integer

        This class and its inherited classes have two modes of behaviour:

           If the notebook is marked from within itself, then ID should be passed.
            - self.mode is set to 'unit-test'
            - __main__ is imported and assigned to self.attempt
            - A stream handler is added to self.mark_logger

           If this is marked through master.py, then converted_file should be passed.
            - self.mode is set to 'batch'
            - student_file.py (converted & parsed) is imported and assigned to self.attempt
            - A file handler is added to self.mark_logger.
            - Additional .txt files for marks and feedback are created.
        """
        if converted_file:
            self.mode = 'batch'
            # NOTE: converted_file.ID guaranteed to be int as Solution1() is called
            # only if self.validID is true after conversion.
            studentID = converted_file.ID
        else:
            self.mode = 'unit_test'
            if ID == None:
                raise ValueError('Student ID must be set!')
            print("Marking student ID {}".format(ID))
            studentID = ID

        # NOTE: In batch mode, convert has already called generator in creating the converted_file
        # but it is more syntactically pleasant to call it again here rather than doing a deep copy.
        super().__init__(studentID)

        if self.mode == 'batch':
            self.dirname = os.path.dirname(converted_file.abs_path)
            self.basename = os.path.basename(converted_file.abs_path)
            self.student_name = os.path.basename(self.dirname)  # In the form danksdominic

            #print("marker  init 1 --------")
            # Name logger after the student for clarity
            self.mark_logger = logging.getLogger(self.student_name)
            #formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
            #                              datefmt='%Y-%m-%d %H:%M:%S')

            formatter = logging.Formatter('%(message)s')
            # {student_name}_marking.log is logging file to be used by mark_logger;
            # set format using formatter
            self.file_handler = logging.FileHandler(os.path.join(
                self.dirname, '{}.log').format(self.student_name))
            self.file_handler.setFormatter(formatter)
            self.mark_logger.addHandler(self.file_handler)

            # Try and import the student's effort.
            # If it doesn't run, set the flag error_in_import to True
            # This is then picked up in master.py
            self.error_in_import = False
            import traceback
            #print("marker  init 2 --------")
            try:
                spec = importlib.util.spec_from_file_location(
                    os.path.splitext(self.basename)[0], converted_file.abs_path)
                student_file = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(student_file)
                self.attempt = student_file
            except Exception:
                traceback.print_exc()
                # Fired if their code doesn't run
                # This flag is caught in master.py, leads to this file not being marked
                self.error_in_import = True

                # Convert.py redirects the stdout to devnull in the student's file to
                # supress print statements. If an exception is thrown in their code
                # then stdout is not redirected. This code redirects stdout back to
                # normal, preventing ResourceError.
                sys.stdout = sys.__stdout__
                # Exit the constructor as nothing else needs to be done as the file isn't marked
                return
            #print("marker  init 3 --------")

        if self.mode == 'unit_test':
            # To mark from within notebook we import __main__ (the file importing from marker)
            import __main__ as main
            self.attempt = main

            self.mark_logger = logging.getLogger(__name__)
            #formatter = logging.Formatter(WrappedFixedIndentingLog('%(message)s',
            #                              datefmt='%Y-%m-%d %H:%M:%S')
            formatter = WrappedFixedIndentingLog('%(message)s',
                                                 datefmt='%Y-%m-%d %H:%M:%S', indent=1)
            stream_handler = logging.StreamHandler()
            #stream_handler.setFormatter(formatter)
            self.mark_logger.addHandler(stream_handler)  # Logging info in notebook

        self.mark_logger.setLevel(logging.DEBUG)  # DEBUG shows feedback, INFO just marks

    # Placeholders defined fully in child classes
    def task1(self):
        pass

    def task2(self):
        pass

    def task3(self):
        pass

    def task4(self):
        pass

    def task5(self):
        pass

    def mark(self, master_dir=None, new_call=False, mark_as_null=False):
        """
        Call each task function and write scores to assignment#_student_marks.txt

        Also writes feedback to assignment#_feedback.txt and a log file in student's
        directory in the unzipped submissions folder

        :param master_dir: String, Absolute path to the directory containing master.py

        :param new_call: Boolean, True for the first call of mark from master.py.
        This acts to write file headers and separate the output of successive runs
        of master.py by using datestamps and newlines.

        :return: None
        """

        # List of functions responsible for marking each task
        # Only Solution1 defines a fifth task
        if self.assignment_num == 1:
            task_marking_functions = [self.task1, self.task2, self.task3, self.task4, self.task5]
        elif self.assignment_num == 2:
            task_marking_functions = [self.task1, self.task2, self.task3, self.task4]
        elif self.assignment_num == 3:
            task_marking_functions = [self.task1, self.task2, self.task3, self.task4]
        else:
            print("Assignment must be 1, 2 or 3")
            sys.exit(1) 

        # comment_and_mark_container is a list of tuples.
        # Each tuple is (comment, mark, MAX_MARK) for the corresponding task
        # comment_and_mark_container[0] references task 1 and so on...
        comment_and_mark_container = []
        assignment_mark = 0  # Student's mark for the entire assignment
        max_assignment_mark = 0  # Maximum possible mark for the entire assignment

        if self.mode == 'unit_test':
            for i, mark_task in enumerate(task_marking_functions):
                comment_and_mark_container.append(mark_task())

                # Store comment and mark data
                comment = comment_and_mark_container[i][0]
                task_mark = comment_and_mark_container[i][1]
                max_task_mark = comment_and_mark_container[i][2]

                assignment_mark += task_mark
                max_assignment_mark += max_task_mark

                self.mark_logger.info("Task {}:".format(i + 1))
                # Feedback as would be given to the student
                self.mark_logger.debug(comment)
                self.mark_logger.info("Score: {}/{} \n".format(task_mark, max_task_mark))
            self.mark_logger.info("Total assignment {} score: {}/{}"
                                  .format(self.assignment_num, \
                                  assignment_mark, max_assignment_mark))
            # Remove stream_handler to stop duplicate logging in the notebook
            handler = self.mark_logger.handlers[0]
            handler.close()
            self.mark_logger.removeHandler(handler)

        elif self.mode == 'batch':
            marks_filepath = os.path.join(master_dir, "output/Head_of_class/_marks.txt")
            feedback_filepath = os.path.join(master_dir, "output/Head_of_class/_feedback.txt")

            try:
                # We want to write the total assignment mark to the feedback file
                # above their comments, however the value is not known at that stage.
                # Therefore, we need to use .seek followed by .write, which is not valid
                # in append mode. Therefore replicate the same functionality as append mode
                # using 'r+'' and 'w' whilst allowing the use of .seek and .write combo.
                if os.path.isfile(feedback_filepath):
                    feedback_open_mode = 'r+'
                else:
                    feedback_open_mode = 'w'

                marks_file_exists = os.path.isfile(marks_filepath)
                with open(marks_filepath, 'a') as mark_file, open(feedback_filepath, \
                          feedback_open_mode) as feedback_file:
                    feedback_file.seek(0, 2)  # Move the file pointer to the end of the file
                    if new_call:
                        if marks_file_exists:
                            mark_file.write("\n\n")
                            feedback_file.write("\n\n")
                        mark_file.write(
                            "TIME: {:%Y-%m-%d %H:%M:%S}\n".format(datetime.datetime.now()))
                        feedback_file.write(
                            "TIME: {:%Y-%m-%d %H:%M:%S}\n".format(datetime.datetime.now()))
                        # Write appropriate headers to mark_file
                        if self.assignment_num == 1:
                            headers = ("{0:33}{1:7}{2:7}{3:7}{4:7}{5:7}{6}\n".format(
                                "Student", "Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Total"))
                            headers = headers+("{0:35}{1:7}{2:7}{3:7}{4:7}{5:7}{6}\n".format(
                                " ", "2", "7", "4", "4", "3", "20"))
                            headers = headers+("----------------------------------"
                                               "---------------------------------------\n")
                        elif self.assignment_num == 2:
                            #headers = ("{0:35}{1}{2:7}{1}{3:7}{1}{4:7}{1}{5:7}{6}\n".format(
                            #    "Student", "Task", " 1 (/6)", " 2 (/3)", " 3 (/8)", " 4 (/5)",
                            # "Total (/20)"))
                            headers = ("{0:33}{1:7}{2:7}{3:7}{4:7}{5}\n".format(
                                "Student", "Task 1", "Task 2", "Task 3", "Task 4", "Total"))
                            headers = headers+("{0:35}{1:7}{2:7}{3:7}{4:7}{5}\n".format(
                                " ", "4", "3", "8", "5", "20"))
                            headers = headers+("----------------------------------"
                                               "---------------------------------------\n")
                        elif self.assignment_num == 3:
                            #headers = ("{0:35}{1}{2:7}{1}{3:7}{1}{4:7}{1}{5:7}{6}\n".format(
                            #    "Student", "Task", " 1 (/6)", " 2 (/3)", " 3 (/8)", " 4 (/5)",
                            # "Total (/20)"))
                            headers = ("{0:33}{1:7}{2:7}{3:7}{4:7}{5}\n".format(
                                "Student", "Task 1", "Task 2", "Task 3", "Task 4", "Total"))
                            headers = headers+("{0:35}{1:7}{2:7}{3:7}{4:7}{5}\n".format(
                                " ", "8", "2", "6", "4", "20"))
                            headers = headers+("----------------------------------"
                                               "---------------------------------------\n")
                        mark_file.write(headers)

                    # First column is always student ID
                    # If the ID is special i.e. '0', print the name
                    IDcolwidth = 35
                    if self.ID == 0:
                        mark_file.write("{:{width}}".format(self.student_name, width=IDcolwidth))
                    else:
                        mark_file.write("{:{width}}".format(self.student_name+" "+str(self.ID), \
                                        width=IDcolwidth))

                    feedback_file.write("=================================\n");
                    feedback_file.write("Name: {}\n".format(self.student_name))
                    if self.ID == 0:
                        feedback_file.write("ID: '0'\n")
                    else:
                        feedback_file.write("ID: {}\n".format(self.ID))

                    header_width = 7  # Number of characters in eg. Task 1 (/2) minus 1

                    # Store the desired point at which to write the total mark
                    # and insert some whitespace so that other data is not overridden
                    total_mark_pos = feedback_file.tell()
                    feedback_file.write("{:20}".format(""))

                    # Write marks and feedback to relevant files
                    for i, mark_task in enumerate(task_marking_functions):
                        if mark_as_null:
                            comment = "(could not be marked)"
                            task_mark = 0
                            max_task_mark = 0
                        else:
                            comment_and_mark_container.append(mark_task())
                            # Store comment and mark data
                            comment = comment_and_mark_container[i][0]
                            task_mark = comment_and_mark_container[i][1]
                            max_task_mark = comment_and_mark_container[i][2]

                        assignment_mark += task_mark
                        max_assignment_mark += max_task_mark

                        mark_file.write('{:<{width}}'.format(task_mark, width=header_width))

                        feedback_file.write("\nTask {}:".format(i + 1))
                        # Feedback as would be given to the student
                        feedback_file.write(comment)
                        feedback_file.write("\nScore: {}/{} \n".format(task_mark, max_task_mark))

                        self.mark_logger.info("Task {}:".format(i + 1))
                        # Feedback as would be given to the student
                        self.mark_logger.debug(comment)
                        self.mark_logger.info("Score: {}/{} \n".format(task_mark, max_task_mark))
                    # Total mark and newline
                    mark_file.write("{:<{width}}\n".format(assignment_mark, width=header_width))

                    # Separate successive students
                    feedback_file.write("\n\n")

                    # Now that the total mark of the student is known, write this to feedback
                    feedback_file.seek(total_mark_pos)
                    feedback_file.write("Total Mark: {}\n".format(assignment_mark))
                    feedback_file.seek(0, 2)  # Reset pointer to the end of the file

                    self.assignment_mark = assignment_mark  # To be accessed in master.py
                    self.mark_logger.info("Total assignment {} score: {}/{} \n"
                                          .format(self.assignment_num, assignment_mark, \
                                          max_assignment_mark))

            except IOError as e2:
                # Capture exceptions and record them with traceback
                print("Operation failed: {}s".format(e2.strerror))


class Solution1(Solution):
    """
    Class responsible for marking the tasks of assignment 1
    """

    # Define functions used in the solution here to save repeated cluttering of functions
    def sol_rhs(self, x_array):
        return np.sqrt(self.lambda_0 - x_array ** 2) / x_array

    def sol_even(self, x_array):
        return np.tan(x_array) - np.sqrt(self.lambda_0 - x_array ** 2) / x_array

    def sol_odd(self, x_array):
        return 1 / np.tan(x_array) + np.sqrt(self.lambda_0 - x_array ** 2) / x_array

    def task1(self):
        """
        Mark the first task (out of 2): Defining rhs function

        Marking points
        --------------
        1: rhs is named and callable
        1: rhs accepts NumPy arrays and evaluates correctly on (0, lambda_0)

        :return: tuple; (Marking comment: string, Mark: int, MAX_MARK)
        """

        MAX_MARK = 2
        running_comment = ""

        failure_comment = (" A not-working rhs function will result in marks lost "
                           "in follow-on tasks.")
        if not hasattr(self.attempt, 'rhs'):
            return (" We checked whether your notebook's namespace had an entry 'rhs' "
                    "and found that it did not." +failure_comment), 0, MAX_MARK
        elif not callable(self.attempt.rhs):
            return (" We found that your notebook's namespace had an entry named 'rhs'."
                    " We then checked whether the object 'rhs' was callable "
                    "and found that it was not."+failure_comment), 0, MAX_MARK
        running_comment += (" We found that 'rhs' was present in your notebook's "
                            "namespace and was callable.")

        # Number of arguments the student function expects
        number_of_arguments = len(inspect.getfullargspec(self.attempt.rhs)[0])
        # If the function is defined and callable but has incorrect number of arguments, give 1
        if number_of_arguments != 1:
            return (" We found that a 'rhs' function was present in the notebook's "
                    "namespace and was callable, but it expects {} arguments when "
                    "it should expect 1.".format(number_of_arguments)+failure_comment), 1, MAX_MARK

        # Sample points to check student's rhs is correct. Avoid singularity at 0 (could mask)
        cut_off = math.sqrt(self.lambda_0)  # Max value of x for a valid square root

        # endpoint=False as need open interval to avoid sqrt error.
        # With endpoint=True, unpredictable behaviour in sqrt near cut_off
        x_array = np.linspace(0, cut_off, endpoint=False)
        # Masked to remove division by 0 at x = 0
        x_array = np.ma.masked_values(x_array, 0, atol=1e-2)

        # Checking return type and argument type is np compatible
        try:
            return_val = self.attempt.rhs(x_array)
            if not isinstance(return_val, np.ndarray):
                return (running_comment + " We passed a numpy array to your 'rhs' "
                        "function and the return type was not an np.ndarray or a "
                        "subclass thereof."+failure_comment), 1, MAX_MARK
        except Exception:
            # Likely AttributeError (TypeError if incorrect No. of args. but this
            # will have been dealt with)
            return (running_comment + " We passed a test numpy array to your 'rhs' function "
                    "and an exception was thrown."+failure_comment), 1, MAX_MARK
        # NOTE: rtol is measured with respect to the second argument
        # NOTE: atol is essential to avoid floating point error issues for small rhs(x)
        if not np.allclose(return_val, self.sol_rhs(x_array), rtol=0.01, atol=1e-08):
            return (running_comment +
                    " Passing a numpy array to the function did not raise an exception and the "
                    "return type was a numpy array as expected. However, the values contained "
                    "within the returned array were outside the region of tolerance around "
                    "the correct values."+failure_comment), 1, MAX_MARK
        else:
            return (running_comment +
                    " Passing a numpy array to the function did not raise an exception and the "
                    "return type was a numpy array as expected. Furthermore, the elements "
                    "of the returned array were numerically correct to the required tolerance."
                   ), 2, MAX_MARK
        
    def task2(self):
        """Mark the second task (out of 7): Plotting

        Marking points
        --------------
        1: Have figure object with axes defined
        1-2: Have one of x AND ylabels, legend (3 entries), title (1). Have all (2)
        1: 3 plots (lines) on one axis
        1: x values (correct range and suitable sample density)
        1-2: One plot numerically correct (1), all plots (2)

        :return: tuple; (Marking comment: string, Mark: int, MAX_MARK: 7)
        """

        MAX_MARK = 7

        if not hasattr(self.attempt, 'student_figure'):
            return (" We checked whether your notebook's namespace had an entry 'student_figure' "
                    "and found that it did not."), 0, MAX_MARK

        running_comment = " We found that your notebook's namespace had an entry 'student_figure'."
        # Now it is defined, try and extract axes from figure
        try:
            # .get_axes() returns empty list for no axes
            #if not self.attempt.student_figure.get_axes():
            #import matplotlib
            #print("------------ Axes:")
            #print(matplotlib.__version__)
            #print(self.attempt.student_figure.axes[0].axis)
            #print(self.attempt.student_figure.axes[0].get_xlabel())
            if not (self.attempt.student_figure.axes):
                return (running_comment + " However, 11 the figure object had no Axes "
                        "objects associated with it."), 0, MAX_MARK

            # 1 mark for simply having a figure object with axes
            running_mark = 1

            # Get first axes of figure (only the first axes is marked)
            #student_axes = self.attempt.student_figure.get_axes()[0]
            student_axes = self.attempt.student_figure.axes[0]

        # If student_figure is not a figure object, exception thrown
        except Exception:
            # Likely AttributeError
            return (running_comment + " However, when we tried to access the Axes "
                                      "objects from 'student_figure' "
                                      "an exception was thrown."), 0, MAX_MARK
        else:
            running_comment += (" The first Axes object of 'student_figure' "
                                "was accessed successfully.")

        # Count number of x/y labels, a title and legend which exist
        number_of_key_properties = 0

        # Find whether the legend is present and has 3 labels:
        l = student_axes.get_legend()
        if hasattr(l, 'texts') and len(l.texts) == 3:
            running_comment += " Your legend had 3 labelled entries as requested."
            number_of_key_properties += 1
        else:
            running_comment += " Your legend did not have 3 labelled entries."

        if student_axes.get_xlabel() and student_axes.get_ylabel():
            running_comment += " You had x and y axis labels as requested."
            number_of_key_properties += 1
        elif student_axes.get_ylabel():
            running_comment += (" You had a y-axis label but lacked an x-axis label. "
                                "Ensure this is included in future.")
        elif student_axes.get_xlabel():
            running_comment += (" You had an x-axis label but lacked a y-axis label. "
                                "Ensure this is included in future.")
        else:
            running_comment += (" You lacked both x and y axis labels. Please "
                                "include these in future.")

        if student_axes.get_title():
            running_comment += " You had a title as requested."
            number_of_key_properties += 1
        else:
            running_comment += " You lacked a title, please include this in future."

        # 2 marks for all properties existing
        if number_of_key_properties == 3:
            running_mark += 2
        # 1 mark for at least 1
        elif number_of_key_properties > 0:
            running_mark += 1
        # Else no additional marks

        lines = student_axes.get_lines()

        # If there are 3 plots on the axes, give one mark
        if len(lines) == 3:
            running_comment += " Your Axes object had 3 line objects as expected."
            running_mark += 1
        else:
            running_comment += " Your Axes object had {} line objects when 3 were expected.".format(
                len(lines))

        # Define and list correct functions for plots
        def mcot(x_array):
            return -1 / np.tan(x_array)

        fns_to_plot = [np.tan, mcot, self.sol_rhs]

        """ Loop through lines of student_axes. If student's x values are
        sensible, compare the value of the correct functions evaluated at
        these values with the student's y_values. If the student's x values
        are not suitable, prune them before making this comparison.
        MAX_ITER to prevent blocking if student plots ridiculous number of
        lines (not currently penalised). """

        MAX_ITER = 10
        number_plotted = 0  # Number of correct plots
        x_data_valid = 0  # Number of plots with valid x_data
        if len(lines) > 3:
            running_comment += (" There appear to be more than 3 traces in the plot (if you "
                                "just plotted three, this can be caused by other errors, such as "
                                "for example an `division by zero'). "
                                "All traces will be checked.")
        for i, line in enumerate(lines):
            running_comment += " Trace number {} : ".format(i + 1)
            if i >= MAX_ITER:
                # Stop after MAX_ITER iterations
                break
            # Get THEIR x and y data
            x_data = line.get_xdata()
            y_data = line.get_ydata()

            #print (x_data)
            #print ("@@@@ ",x_data.size)

            # Now we check whether their x_data is valid...
            cut_off = math.sqrt(self.lambda_0)  # Max value of the desired range of x_values

            # Check whether the data lies within the correct range and
            # whether the interval is used fully enough, i.e. not
            # just a few points in the middle of the interval.
            # If it is good, then set as valid
            x_data_min = np.amin(x_data)
            x_data_max = np.amax(x_data)
            if x_data_max < cut_off and x_data_min > 0:
                #running_comment += " All of your plotted points were within the open
                #interval as requested. "
                if x_data.size >= 100 and x_data_max > 0.9 * cut_off and x_data_min < 0.1 * cut_off:
                    running_comment += (" You also had at least 100 points in the open interval "
                                        "and sampled a range of x values covering 0.1 * "
                                        "sqrt(lambda_0) to 0.9 * sqrt(lambda_0).")
                    x_data_valid += 1
                elif x_data.size >= 100:
                    running_comment += (" You also had at least 100 points within the open "
                                        "interval. However, you did not explore enough of the "
                                        "interval (we expected the range of x values "
                                        "between 0.1 * sqrt(lambda_0) and 0.9 * sqrt(lambda_0)"
                                        " to be fully explored).")
                elif x_data_max > 0.9 * cut_off and x_data_min < 0.1 * cut_off:
                    running_comment += (" You explored a good range of points "
                                        "(at least from 0.1 * sqrt(lambda_0) to 0.9 * "
                                        "sqrt(lambda_0)), however you did not sample at "
                                        "least 100 points in the open interval.")
                else:
                    running_comment += (" You did not plot at least 100 points in the open "
                                        "interval and did not explore enough of the interval "
                                        "(we expected the range of x values "
                                        "between 0.1 * sqrt(lambda_0) and 0.9 * sqrt(lambda_0) "
                                        "to be fully explored).")
            else:
                running_comment += (" At least one plotted point has an x value outside of the "
                                    "open interval (0, sqrt(lambda_0)) and so you lost a "
                                    "mark here.")
                # Their x_data is unsafe and would cause issues in the next block
                # due to division by zero or imaginary square root errors.
                # Can't use our own x values as wouldn't be valid to compare against y_data
                # Therefore modify their data to be safe by removing values outside of interval.

                # Do this by looping through their x_data, storing the indices of the problematic
                # values in a list. Then delete the elements with these indices from their data.
                indices_to_remove = []
                for j, elem in enumerate(x_data):
                    if elem >= cut_off or elem <= 0:
                        indices_to_remove.append(j)
                indices_to_remove.reverse()
                for index in indices_to_remove:
                    x_data = np.delete(x_data, index)
                    y_data = np.delete(y_data, index)

            # Now their data is safe, but the size could be very small or 0
            # If there is too little, give 0 for this line as we cannot reliably check the
            # function acting on x_data gives the correct functional behaviour.
            if x_data.size < 20:
                running_comment += (" There were less than 20 data points inside "
                                    "the open interval - too low a number to assess "
                                    "the validity of your plotted function.")
                continue

            # At this point, their data is safe and there is enough of it to mark
            # NOTE: Not removed tan and cot singularities apart from cot(0) but this is
            # not a real problem as eg. tan(np.pi/2) is just v. large, not an error.

            # Now check if their y_data matches with the result of the correct function
            # acting on their (modified) x_data.
            # If it does, then add one to number plotted so that the number of correctly
            # plotted functions can be counted below.
            plot_valid = False
            # only check is still functions to be found
            if (len(fns_to_plot) > 0):
                for i, fn in enumerate(fns_to_plot):
                    #print("i {}, x0 {} y0 {} f(x) {}".format(i, x_data[0], y_data[0], fn(x_data[0])))
                    if np.allclose(y_data, fn(x_data), rtol=0.01, atol=1e-08):
                        number_plotted += 1
                        plot_valid = True
                        # Remove fn to ensure that multiple marks can't be gained
                        # from plotting the same function repeatedly.
                        fns_to_plot.remove(fn)
                        running_comment += (" This line matches an expected function within relevant "
                                        "tolerances - well done.")
                        break
                if not plot_valid:
                    running_comment += (" This line does not match with an expected function "
                                        "within relevant tolerances.")

        # 1 mark for all x data valid
        if x_data_valid == 3:
            running_mark += 1

        # 2 marks for all plots correct
        if number_plotted == 3:
            running_mark += 2
        # 1 mark for at least 1
        elif number_plotted > 0:
            running_mark += 1
        # Else no additional marks
        running_comment += (" Overall, {}/3 of your x data sets were fully valid "
                            "with enough points in the relevant range and {} "
                            "of your plotted functions were checked to be correct."
                            .format(x_data_valid, number_plotted))

        # Return final comment, mark and MAX_MARK
        return running_comment, running_mark, MAX_MARK

    def task3(self):
        """Mark the third task (out of 4): even_ and odd_equation

        Marking points
        --------------
        1: even_equation named and callable with correct number of arguments (1)
        1: odd_equation named and callable with correct number of arguments (1)
        1: even_equation numerically correct
        1: odd_equation numerically correct


        :return: tuple; (Marking comment: string, Mark: int, MAX_MARK: 4)
        """

        MAX_MARK = 4
        running_mark = 0
        running_comment = ""

        cut_off = math.sqrt(self.lambda_0)  # Endpoint of the x_interval of interest

        # Generate arrays of sample points at which to compare their
        # functions to the model function.
        x_array = np.linspace(0, cut_off, endpoint=False)  # No endpoint to avoid imaginary sqrt

        # List containing the singular points of cot and tan on the interval
        n_max = int(cut_off // (np.pi / 2))
        tan_singularities = [n * np.pi / 2 for n in range(n_max + 1) if n % 2 == 1]
        cot_singularities = [n * np.pi / 2 for n in range(n_max + 1) if n % 2 == 0]

        # x_for_tan is an np array to pass to even_equation. It is x_array
        # with the tan singularities masked.
        x_for_tan = x_array
        for y in tan_singularities:
            x_for_tan = np.ma.masked_values(x_for_tan, y, atol=1e-2)
        # x_for_cot is an np array to pass to odd_equation. It is x_array
        # with the cot singularities masked.
        x_for_cot = x_array
        for y in cot_singularities:
            x_for_cot = np.ma.masked_values(x_for_cot, y, atol=1e-2)

        # Marking of even equation
        # Firstly check even_equation is named and callable
        if not hasattr(self.attempt, 'even_equation'):
            running_comment += (" We checked whether your notebook's namespace had "
                                "an entry 'even_equation' and found that it did not.")
        elif not callable(self.attempt.even_equation):
            running_comment += (" We checked whether your notebook's namespace had an "
                                "entry 'even_equation. We then checked whether the "
                                "object 'even_equation' was callable and found that "
                                "is was not.")
        else:
            running_comment += (" We found 'even_equation' to be present in the "
                                "notebook's namespace and was callable.")
            # Function must take a single argument
            even_arguments = len(inspect.getfullargspec(self.attempt.even_equation)[0])
            if even_arguments != 1:
                running_comment += (" However it expects {} arguments when it should "
                                    "expect 1 and so cannot "
                                    "be marked.".format(even_arguments))
            else:
                # Correct number of arguments (1 mark), now test the functionality
                running_mark += 1
                try:
                    # If an exception is thrown, put down to not accepting np
                    # arrays (likely AttributeError)
                    even_return_value = self.attempt.even_equation(x_for_tan)
                    if not isinstance(even_return_value, np.ndarray):
                        running_comment += " However it does not return a np.ndarray."
                    elif np.allclose(self.attempt.even_equation(x_for_tan),
                                     self.sol_even(x_for_tan), rtol=0.01,
                                     atol=1e-08):
                        # Numerically correct (1 mark)
                        running_mark += 1
                        running_comment += (" Passing a numpy array to 'even_equation'e "
                                            "did not raise an exception and "
                                            "the return type was a numpy array as expected. "
                                            "Furthermore, the elements of the returned array "
                                            "were numerically correct to the required "
                                            "tolerance.")
                    else:
                        running_comment += (" Passing a numpy array to 'even_equation' did not "
                                            "raise an exception and "
                                            "the return type was a numpy array as expected. "
                                            "However, the values contained within the returned "
                                            "array were outside the region of tolerance "
                                            "around the correct values.")
                except Exception:
                    running_comment += (" However we passed a test numpy array to this function "
                                        "and an exception was thrown.")
        # Marking of odd equation
        # Firstly check odd_equation is named and callable
        if not hasattr(self.attempt, 'odd_equation'):
            running_comment += (" We checked whether your notebook's namespace had an entry "
                                "'odd_equation' and found that it did not.")
        elif not callable(self.attempt.odd_equation):
            running_comment += (" We checked whether your notebook's namespace had an entry "
                                "'odd_equation.' We then checked whether the object "
                                "'odd_equation' was callable and found "
                                "that is was not.")
        else:
            running_comment += (" We found 'odd_equation' to be present in the notebook's "
                                "namespace and was callable.")
            # Function must take a single argument
            odd_arguments = len(inspect.getfullargspec(self.attempt.odd_equation)[0])
            if odd_arguments != 1:
                running_comment += (" However it expects {} arguments when it should expect 1 "
                                    "and so could not be marked.".format(odd_arguments))
            else:
                # Correct number of arguments (1 mark), now test the functionality
                running_mark += 1
                try:
                    # If an exception is thrown, put down to not accepting np arrays
                    # (likely AttributeError)
                    odd_return_value = self.attempt.odd_equation(x_for_cot)
                    if not isinstance(odd_return_value, np.ndarray):
                        running_comment += (" However it does not return a np.ndarray or a "
                                            "subclass thereof.")
                    elif np.allclose(self.attempt.odd_equation(x_for_cot), 
                                     self.sol_odd(x_for_cot), rtol=0.01,
                                     atol=1e-08):
                        # Numerically correct (1 mark)
                        running_mark += 1
                        running_comment += (" Passing a numpy array to 'odd_equation' did not "
                                            "raise an exception and "
                                            "the return type was a numpy array as expected. "
                                            "Furthermore, the "
                                            "elements of the returned array were numerically "
                                            "correct to the required tolerance.")
                    else:
                        running_comment += (" Passing a numpy array to 'odd_equation' did not "
                                            "raise an exception and "
                                            "the return type was a numpy array as expected. "
                                            "However, the values contained within the returned "
                                            "array were outside the region of tolerance "
                                            "around the correct values.")
                except Exception:
                    running_comment += (" However we passed a test numpy array to this function "
                                        "and an exception was thrown.")

        return running_comment, running_mark, MAX_MARK

    def task4(self):
        """Mark the fourth task (out of 4): bisection method

        Marking points
        --------------
        1: student_list exists and is a sorted list
        1-3: 1 mark for each root numerically correct


        :return: tuple; (Marking comment: string, Mark: int, MAX_MARK: 4)
        """

        MAX_MARK = 4

        running_mark = 0
        running_comment = ""

        # Check solution_list exists as a list
        if not hasattr(self.attempt, "solution_list"):
            return (" We checked whether your notebook's namespace had an entry 'solution_list' "
                    "and found that it did not."), 0, MAX_MARK
        if not isinstance(self.attempt.solution_list, list):
            return (" Your notebook's namespace had an entry 'solution_list', "
                    "however it was not a list object.",
                    0, MAX_MARK)

        student_list = self.attempt.solution_list
        marking_list = []

        running_comment += " Your notebook's namespace had a list object 'solution_list',"
        if len(student_list) != 3:
            return ((running_comment + ". However it had {} elements when it should only have "
                                       "3.".format(len(student_list))), 0, MAX_MARK)

        # Generate solutions and store in marking_list
        delta = 1e-2  # To avoid singularities
        marking_list.append(optimize.bisect(self.sol_even, 0 + delta, np.pi / 2 - delta))
        marking_list.append(optimize.bisect(self.sol_odd, np.pi / 2 + delta, np.pi - delta))
        marking_list.append(optimize.bisect(self.sol_even, np.pi + delta, 3 * np.pi / 2 - delta))

        # Make copy of student's list that is definitely sorted
        copy_list = sorted(student_list)
        # Now check if student_list is sorted; 1 mark if it is
        if student_list == copy_list:
            running_mark += 1
            running_comment += " that was correctly sorted (ascending order)."
        else:
            running_comment += " which was not correctly sorted (ascending order)."

        # Loop through student_list & see if the entry matches with an entry in marking_list.
        # If it does, mark as a correct entry and remove the corresponding entry from marking_list
        # so that multiple marks cannot be gained from the same correct root in the student's list.
        MAX_ITER = 10
        number_correct = 0
        #print("student_list: {}".format(student_list))
        #print("marking_list: {}".format(marking_list))

        for i, student_root in enumerate(student_list):
            #print("------------------------"+student_root)
            if i >= MAX_ITER:
                break  # Stop if MAX_ITER reached

            if not isinstance(student_root, float):
                ### TODO be explicit about float and if so deduct marks
                running_comment += " Your list of solutions did not contain float values."
                #break
            for sol_root in marking_list:
                #print("student_root: {}".format(student_root))
                #print("sol_root: {}".format(sol_root))
                if math.isclose(float(student_root), sol_root, rel_tol=1e-02, abs_tol=1e-08):
                    number_correct += 1
                    marking_list.remove(sol_root)
                    # Move to next entry in student list; no need to check the rest of marking_list
                    break

        # One mark for each numerically correct root
        running_mark += number_correct
        running_comment += " {} roots were found to be numerically correct.".format(
            number_correct)

        # Return to mark
        return running_comment, running_mark, MAX_MARK

    def task5(self):
        """Mark the fifth task (out of 3): find_energy and formatting

        Marking points
        --------------
        2: Return a list of 3 strings formatted to 3dp
        1: Return numerically correct values


        :return: tuple; (Marking comment: string, Mark: int, MAX_MARK: 3)
        """

        # Maximum achievable mark for this task
        MAX_MARK = 3
        # Current mark of student and comment on this mark
        running_mark = 0
        running_comment = ""

        # Give zero if the function is undefined or not callable
        if not hasattr(self.attempt, "find_energy"):
            return (" We checked whether your notebook's namespace had an entry 'find_energy' "
                    "and found that it did not."), 0, MAX_MARK
        elif not callable(self.attempt.find_energy):
            return (" We found that your notebook's namespace has an entry named 'find_energy'."
                    "We then checked whether the object `find_energy` was callable "
                    "and found that it was not."), 0, MAX_MARK

        # Check find_energy takes correct number of args.
        number_of_arguments = len(inspect.getfullargspec(self.attempt.find_energy)[0])

        running_comment += (" We found that a 'find_energy' function was present in the "
                            "notebook's namespace and was callable.")
        if number_of_arguments != 1:
            return (running_comment + " However it expects {} arguments when it should "
                                      "expect 1.".format(number_of_arguments)), 0, MAX_MARK

        # Defined and callable with correct number args. Now pass
        # the correct list to the function and check output.

        # First generate the correct x values
        delta = 1e-2  # To avoid singularities
        x_values = []
        x_values.append(optimize.bisect(self.sol_even, 0 + delta, np.pi / 2 - delta))
        x_values.append(optimize.bisect(self.sol_odd, np.pi / 2 + delta, np.pi - delta))
        x_values.append(optimize.bisect(self.sol_even, np.pi + delta, 3 * np.pi / 2 - delta))

        # List to hold correct numerical values of energy
        correct_energies = []
        a = self.width
        for x_val in x_values:
            E = 2 * (constants.hbar ** 2) * (x_val ** 2) / (constants.e * constants.m_e * (a ** 2))
            correct_energies.append(E)

        # Pass the correct x_values to the function to test its functionality,
        # No exception would be raised if a reasonable function is written by student,
        # therefore give 0 marks if one is raised
        
        running_comment += " We passed it a list of the 3 correct roots"
        try:
            student_return_value = self.attempt.find_energy(x_values)
        except Exception:
            # Probably AttributeError (TypeError if incorrect No. of args.
            # but this will have been dealt with)
            return running_comment + " and an exception was thrown.", 0, MAX_MARK
        # If the return value is not a list, give zero
        if not isinstance(student_return_value, list):
            return (running_comment + " and it did not return a list object or " +
                    "a subclass thereof."), 0, MAX_MARK
        # If it changes the length of the input list, give zero
        elif len(student_return_value) != 3:
            return (running_comment + " and it returned a list object of " +
                    "length {} (3 expected)."
                    .format(len(student_return_value))), 0, MAX_MARK

        # Now consider the values in the list.

        # Number out of 3 elements in the return value that are correctly formatted strings to 3dp
        elements_correctly_formatted = 0

        # Number of out 3 elements in the return value that are numerically correct
        elements_numerically_correct = 0

        # Loop through the returned list and check for number in a string and 3dp
        for i, student_energy_string in enumerate(student_return_value):
            try:
                student_energy_num = float(student_energy_string)
            except:
                return (running_comment + " and it returned a list whose elements "
                        "were not convertible to float."), 0, MAX_MARK
            else:
                # Now we know that the output can be converted to a number
                # See if it is a formatted string
                if isinstance(student_energy_string, str):
                    # The output was a number within a string
                    # Now test whether it had 3dp by searching the string for
                    # the decimal point after removing exponential notation.
                    student_energy_mantissa = student_energy_string.split('e')[0]
                    student_energy_mantissa = student_energy_mantissa.split('E')[0]
                    sig_dig = count_sig_figs(student_energy_mantissa)
                    #print(sig_dig)
                    # allow for 4, 3 or 2 (in case the last digit is 0)
                    # allowing 4 as we asked for decimal places TODO change to significant digits!
                    if sig_dig == 4 or sig_dig == 3 or sig_dig == 2:
                        elements_correctly_formatted += 1

                # Now check the numerical value of the elements. Set tols to check 3dp
                # Generous here but that is better than issues with floating point inaccuracy
                if math.isclose(student_energy_num, correct_energies[i],
                                rel_tol=1e-20, abs_tol=0.001):
                    elements_numerically_correct += 1

        if elements_correctly_formatted == 3:
            running_mark += 2
            running_comment += " and it returned a correctly formatted list, "
        else:
            running_comment += " and it returned an incorrectly formatted list, "

        if elements_numerically_correct == 3:
            running_mark += 1
            running_comment += " of numerically correct values."
        else:
            running_comment += " of numerically incorrect values."

        return running_comment, running_mark, MAX_MARK

    def mark(self, master_dir=None, new_call=False, mark_as_null=False):
        self.assignment_num = 1
        super().mark(master_dir, new_call, mark_as_null)

def count_sig_figs(digits):
    '''Return the number of significant figures of the input digit string'''
    integral, _, fractional = digits.partition(".")
    if fractional:
        return len((integral + fractional).lstrip('0'))
    else:
        return len(integral.strip('0'))

class Solution2(Solution):
    """Class responsible for marking the tasks of assignment 2"""

    def RLC_transfer_func(self, freq_array, **kwargs):
        """Transfer function for RLC band pass filter circuit

        The complex impedance of the circuit is firstly calculated
        It is important to avoid a divide by zero error for first value of frequency
        The transfer function tends to zero as frequency tends to zero

        :param freq_array: Numpy Array containing frequencies in Hz at which to g
               enerate transfer function
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
            assignment_logger.warning('RLC_transfer_func:', repr(err), 'returning None.')
            return None

        # Array of angular frequencies
        omega = 2 * np.pi * freq_array

        # Mask 0 using floating point equality
        omega = ma.masked_values(omega, 0)
        # Note assignment: most numpy methods don't work in place

        # Output across resistor (See useful_formulae.tex)
        masked_return = R / (R + (omega * L - 1 / (omega * C)) * 1j)

        # Return array, filling masked value with 0 (transfer function is zero at freq. of 0)
        return ma.filled(masked_return, 0)

    @staticmethod
    def RL_LP_transfer_func(freq_array, R, L):
        """Transfer function for RL LOW pass filter circuit

        :param freq_array: Numpy Array containing frequencies on which to generate transfer function
        :param R: Resistance in Ohms
        :param L: Inductance in Henrys
        :returns: Transfer function evaluated at points of freq_array (Numpy array).
        """
        # Array of angular frequencies
        omega = 2 * np.pi * freq_array

        # For RL low pass transfer function see useful_formulae.tex
        return R / (R + omega * L * 1j)

    @classmethod
    def my_box_process(cls, t_array, input_array, R, L):
        """Solution function for task 4

        :param t_array: Numpy Array containing the times at which the input signal is sampled
        :param input_array: Numpy Array containing the amplitude of the signal at the sample times
        :param R: Resistance in Ohms
        :param L: Inductance in Henrys
        :returns: Output amplitude signal of 'black box' (Numpy array).
        """
        # Functionality similar to Boxes.process in assignment.py
        transformed_signal = np.fft.fft(input_array)
        intervals = np.diff(t_array)
        # Determine sample points in f_domain using fftfreq (Note: Not angular frequency)
        timestep = intervals[0]
        #print("mark N {}, total {}, timestep {:.8}".format(t_array.size,
        #t_array[-1]-t_array[0],timestep))
        freq_array = np.fft.fftfreq(t_array.size, d=timestep)
        # Apply relevant transfer function to transformed_signal (multiplication in f_domain)
        modified_f_domain_signal = transformed_signal * cls.RL_LP_transfer_func(freq_array, R, L)
        # Perform inverse FFT on the modulated input signal
        output_signal = np.fft.ifft(modified_f_domain_signal)
        return output_signal.real

    def task1(self):
        """Mark the first task (out of 10): Black box identification

        Marking points
        --------------
        1: One box correctly identified
        2: Two boxes correctly identified
        4: All boxes correctly identified

        :return: tuple; (Marking comment: string, Mark: int, MAX_MARK: 4)
        """
        MAX_MARK = 4
        running_mark = 0
        running_comment = ""
        # Recreate ordering generated by assignment2.Boxes.get_boxes
        required_circuits = ["RL_circuit", "RC_circuit", "RLC_circuit"]

        first_rand = self.key % 3  # 0, 1 or 2
        second_rand = self.key2 % 2  # 0, 1. Use second key else two rands are related

        correct_circuits = []
        correct_circuits.append(required_circuits.pop(first_rand))
        correct_circuits.append(required_circuits.pop(second_rand))
        correct_circuits.append(required_circuits.pop())
        # Number of boxes correctly identified
        number_correct = 0
        running_comment += (" We checked that '{}', '{}' and '{}' were variables "
                            "in your notebook: "
                            "").format(correct_circuits[0],
                                       correct_circuits[1],
                                       correct_circuits[2])
        # Loop to check correctly named
        for index, circuit_name in enumerate(correct_circuits):
            if index > 0:
                running_comment += ", "
            if not hasattr(self.attempt, circuit_name):
                running_comment += "{} was not correctly named".format(circuit_name)
                continue
            else:
                running_comment += "{} was found".format(circuit_name)
            # N.B. main.circuit_name will throw AttributeError
            student_box_number = getattr(self.attempt, circuit_name)
            if callable(circuit_name):
                running_comment += ("{} was correctly named but was found to be "
                                    "callable (i.e. a function not a variable)"
                                   ).format(circuit_name)
                continue
        running_comment += ". "
        for index, circuit_name in enumerate(correct_circuits):
            if index > 0:
                running_comment += ", you "
            else:
                running_comment += "You "
            student_box_number = getattr(self.attempt, circuit_name)
            if student_box_number == index + 1:
                number_correct += 1
                running_comment += "correctly identified box {} with {}".format(
                    index + 1, circuit_name)
            else:
                running_comment += "thought that {} was in box {} but it was actually in box {}"\
                    .format(circuit_name, student_box_number, index + 1)
        running_comment += ". "

        # Apply marking scheme laid out in the plan document
        if number_correct == 1:
            running_mark += 1
        elif number_correct == 2:
            running_mark += 2
        elif number_correct == 3:
            running_mark += 4

        #running_comment += "Number correct: {}".format(number_correct)
        return running_comment, running_mark, MAX_MARK

    def task2(self):
        """Mark the second task (out of 3)

        Marking points
        --------------
        1: corner_frequency accurate to 50 Hz
        2: corner_frequency accurate to 10 Hz
        1: inductance correct (using their corner_frequency)

        N.B. corner_frequency in Hz (Not an angular frequency)

        :return: tuple; (Marking comment: string, Mark: int, MAX_MARK: 3)
        """
        MAX_MARK = 3
        running_mark = 0
        running_comment = ""
        corner_var_name = "corner_frequency"
        inductance_var_name = "inductance"

        # Student receives no marks if corner frequency is not a variable
        if not hasattr(self.attempt, corner_var_name):
            return " We did not find the variable '{}' in your notebook."\
                .format(corner_var_name), 0, MAX_MARK
        student_corner_freq = getattr(self.attempt, corner_var_name)
        running_comment += " '{}' was found".format(
            corner_var_name)
        if callable(student_corner_freq):
            return (running_comment + ", but it was found to be callable (i.e. "
                    "a function not a variable)."), 0, MAX_MARK
        if student_corner_freq == 0:
            return (running_comment + ", but it was found to be zero."
                    "".format(corner_var_name)), 0, MAX_MARK

        # Calculate the corner frequency and compare it to the student's value,
        # awarding marks where necessary
        R = 200  # Value of Ohms chosen for this task
        L = (self.MIN_INDUCTANCE +
             (self.key2 / self.norm) * (self.MAX_INDUCTANCE - self.MIN_INDUCTANCE))
        true_corner_freq = R / (2 * np.pi * L)  # Actual corner frequency in Hz
        #print("true corner {}, found {}".format(true_corner_freq,student_corner_freq ))
        if math.isclose(student_corner_freq, true_corner_freq, rel_tol=1e-20, abs_tol=10):
            running_mark += 2
            running_comment += (" and its value was within 10Hz of the true value "
                                "({:.1f}).").format(true_corner_freq)
        elif math.isclose(student_corner_freq, true_corner_freq, rel_tol=1e-20, abs_tol=50):
            running_mark += 1
            running_comment += (" but its value was only correct within a tolerance "
                                "of 50Hz (not 10Hz) compared to the "
                                "true value ({:.1f}).").format(true_corner_freq)
        else:
            running_comment += (" but its value was not within 50Hz of the true "
                                "value ({:.1f}).").format(true_corner_freq)

        # difference = abs(true_corner_freq - student_corner_freq)  # Difference
        # between true and student value
        # running_comment += "(diff = {:.3g}Hz) ".format(difference)  # For debugging

        # No further marks if inductance not named or is a function
        if not hasattr(self.attempt, inductance_var_name):
            running_comment += " The variable '{}' was not found.".format(inductance_var_name)
            return running_comment, running_mark, MAX_MARK
        student_task2_inductance = getattr(self.attempt, inductance_var_name)
        running_comment += " '{}' was found".format(
            inductance_var_name)
        if callable(inductance_var_name):
            running_comment += (" but it was callable (i.e. a function not a variable)")
            return running_comment, running_mark, MAX_MARK
        # Calculate 'correct' inductance using student_corner_freq value
        their_true_inductance = R / (2 * np.pi * student_corner_freq)
        #print("f {:.8e}, L {:.8e}".format(student_corner_freq,  their_true_inductance ))
        # 1 additional mark if close to within ~> floating point error.
        print(student_task2_inductance- their_true_inductance)
        if math.isclose(student_task2_inductance, their_true_inductance,
                        rel_tol=1e-2, abs_tol=1e-4):
            running_comment += " and its value was correct (error carried forward considered)."
            running_mark += 1
        else:
            running_comment += (" but its value ({:.3e}) was not the correct one "
                                "({:.3e}) (tolerance=1e-2, error carried forward "
                                "considered)."
                                "".format(student_task2_inductance, their_true_inductance))

        return running_comment, running_mark, MAX_MARK

    def task3(self):
        """Mark the third task (out of 8): Plotting Band Pass Resonance

        Marking points
        --------------
        1: Figure object with axis, title & axis labels
        1: Logarithmic x_scale (use of semilogx)
        1: Plot includes correct range: 0-2000 Hz
        1: Correct y_data (transfer function)
        1-2: Correct resonance frequency: within 50 Hz -> 1, within 10 Hz -> 2
        1-2: Correct bandwidth: within 50 Hz -> 1, within 10 Hz -> 2

        :return: tuple; (Marking comment: string, Mark: int, MAX_MARK: 8)
        """
        MAX_MARK = 8
        running_mark = 0
        running_comment = ""

        # Parameters of the RLC circuit in this task
        R = (self.MIN_RESISTANCE +
             (self.key3 / self.norm) * (self.MAX_RESISTANCE - self.MIN_RESISTANCE))
        L = (self.MIN_INDUCTANCE +
             (self.key3 / self.norm) * (self.MAX_INDUCTANCE - self.MIN_INDUCTANCE))
        C = (self.MIN_CAPACITANCE +
             (self.key3 / self.norm) * (self.MAX_CAPACITANCE - self.MIN_CAPACITANCE))
        # param for self.RLC_transfer_func
        t_func_params = {'R': R, 'L': L, 'C': C}

        # No marks if student_figure isn't defined or has no axes
        if not hasattr(self.attempt, 'student_figure'):
            return (" A figure with name 'student_figure' was not found."), 0, MAX_MARK

        running_comment = " A figure called 'student_figure' was found"
        # Now it is defined, try and extract axes from figure
        figure_works = False
        try:
            # .get_axes() returns empty list for no axes
            #if not self.attempt.student_figure.get_axes():
            if not self.attempt.student_figure.axes:
                running_comment += " but the figure object had no Axes objects associated with it."
            else:
                # Get first axes of figure (only the first axes is marked)
                #student_axes = self.attempt.student_figure.get_axes()[0]
                student_axes = self.attempt.student_figure.axes[0]
                figure_works = True

        # If student_figure is not a figure object, exception thrown
        except Exception:
            # Likely AttributeError
            running_comment += (" but when we tried to access the Axes objects "
                                "from 'student_figure' an exception was thrown.")
        if figure_works:
            running_comment += ". "
            # See if axes has x/y labels, a title and legend, give a mark if they are all there
            # Also check each case individually to give feedback
            key_properties_present = [bool(student_axes.get_ylabel()),
                                      bool(student_axes.get_xlabel()),
                                      bool(student_axes.get_title())]
            num_key_properties = 0
            if all(key_properties_present):  # Only case resulting in a mark
                num_key_properties = 3
                running_mark += 1
                running_comment += " The figure has an x axis label, y axis label and a title "
            elif not any(key_properties_present):
                running_comment += " The figure has no x axis label, y axis label or title. "
            else:
                running_comment += " This figure has:"
                if key_properties_present[1]:
                    num_key_properties += 1
                    running_comment += " x axis label"
                if key_properties_present[0]:
                    if num_key_properties > 0:
                        running_comment += ","
                    num_key_properties += 1
                    running_comment += " y axis label"
                if key_properties_present[2]:
                    if num_key_properties > 0:
                        running_comment += ","
                    num_key_properties += 1
                    running_comment += " title"
                running_comment += ". It does not have:"
                set_comma = False
                if not key_properties_present[1]:
                    running_comment += " x axis label"
                    set_comma = True
                if not key_properties_present[0]:
                    if set_comma:
                        running_comment += ","
                    running_comment += " y axis label"
                    set_comma = True
                if not key_properties_present[2]:
                    if set_comma:
                        running_comment += ","
                    running_comment += " title"
            running_comment += ". "

            # Test whether ticks are logarithmically spaced (log 10 scale used)
            # Assumes student has not meddled with the axis ticks
            student_ticks = student_axes.get_xticks()
            #print(student_ticks)
            if (student_ticks <= 0).any():
                #print("a")
                running_comment += (" The x axis was not logarithmic spaced (a log10 "
                                    "scale was required).")
            else:
                log_ticks = np.log10(student_ticks)
                tdiff = np.diff(log_ticks)
                if np.allclose(tdiff, np.ones(tdiff.size)*tdiff[0], rtol=0, atol=1e-12):
                    running_mark += 1
                    running_comment += " The x axis is logarithmic spaced, as required (log10)."
                else:
                    running_comment += (" The x axis was not logarithmic spaced (a log10 "
                                        "scale was required).")
            # .get_lines() returns empty list (falsy) if no lines (nothing plotted)
            if student_axes.get_lines():
                # Get first line (plotted function) from student's fig (additional lines ignored)
                first_student_line = student_axes.get_lines()[0]
                # No penalty for unsuitable sample density (will likely loose marks
                # on resonance frequency)
                # Get THEIR x and y data
                x_data = first_student_line.get_xdata()
                y_data = first_student_line.get_ydata()

                # Check whether the data covers at least 0.1 to 0.9 of the cutoff at 2000Hz
                # (1 mark)
                # Lenient to allow for endpoint=False and frequency spacing type errors
                # Issues here will be penalised later on when their resonance frequency
                # is inaccurate
                x_data_min = np.amin(x_data)
                x_data_max = np.amax(x_data)
                cutoff = 2000
                running_comment += " "
                if x_data_min <= 0.1 * cutoff and x_data_max >= 0.9 * cutoff:
                    running_mark += 1
                    running_comment += " The range of frequencies you explored were good " \
                                   "(at least from {} to {}).".format(0.1 * cutoff, 0.9 * cutoff)
                else:
                    running_comment += (" However, you did not explore enough of the interval "
                                        "[{}, {}] (we expected a "
                                        "range of x values between {} and {})."
                                        "".format(0, cutoff, 0.1 * cutoff, 0.9 * cutoff))

                running_comment += " "
                # taking only positive frequencies just in case the students plotted the whole FFT
                idxs = x_data > 0
                x_data = x_data[idxs]
                y_data = y_data[idxs]
                #print(x_data)
                #print(y_data)
                #print(np.max(np.abs(y_data-np.abs(self.RLC_transfer_func(np.abs(x_data),**t_func_params)))))
                #print(np.absolute(self.RLC_transfer_func(x_data, **t_func_params)))
                # Check plot y_data is correct. Use student x_values & compare to
                # the magnitude of true RLC_transfer_func
                if np.allclose(y_data,
                               np.absolute(self.RLC_transfer_func(np.abs(x_data),
                                                                  **t_func_params)),
                               rtol=0.02, atol=1e-2):
                    # Only 1 mark as only testing use of .process (student doesn't have
                    # to write a func)
                    running_mark += 1
                    running_comment += (" Your plotted values matched the output of the "
                                        "true transfer function "
                                        "to within an acceptable tolerance.")
                else:
                    running_comment += (" Your plotted values did not match the output of "
                                        "the true transfer "
                                        "function to within an acceptable tolerance.")
            else:
                running_comment += (" However, no lines were found to be plotted on "
                                    "this Axes object.")


        #running_comment += " "
        # Correct value of resonance frequency for comparison
        true_resonance_freq = 1 / (2 * np.pi * math.sqrt(L * C))
        # Calculate lower and upper cut-off frequencies using parameter values
        f_lower = 1 / (2 * np.pi) * (math.sqrt(R ** 2 / (2 * L) ** 2 + 1 / (L * C)) - R / (2 * L))
        f_upper = 1 / (2 * np.pi) * (math.sqrt(R ** 2 / (2 * L) ** 2 + 1 / (L * C)) + R / (2 * L))
        # Hence calculate correct value of bandwidth
        true_bandwidth = f_upper - f_lower
        # Tuple of tuples of the form (variable_name, correct_value)
        task3_params = (("resonant_frequency", true_resonance_freq),
                        ("bandwidth", true_bandwidth))
        # Unpack each tuple into param_name and true_value variables
        for param_name, true_value in task3_params:
            # Check whether variable exists & isn't a function. Otherwise no further marks
            if hasattr(self.attempt, param_name):
                running_comment += " '{}' was found".format(param_name)
                student_var = getattr(self.attempt, param_name)
                if not callable(student_var):
                    # A value within 10 Hz gets 2 marks; within 50 Hz (but not 10 Hz) 1 mark
                    if math.isclose(student_var, true_value, rel_tol=1e-20, abs_tol=10):
                        running_mark += 2
                        running_comment += (" and its value was within 10Hz of the "
                                            "true value ({:.1f}).".format(true_value))
                    # If require diff. tolerances for resonance/bandwidth could add
                    # a 'tol' element to each tuple
                    elif math.isclose(student_var, true_value, rel_tol=1e-20, abs_tol=50):
                        running_mark += 1
                        running_comment += (" but its value was only correct within a "
                                            "tolerance of 50Hz (not 10Hz) compared to "
                                            "the true value ({:.1f}).".format(true_value))
                    else:
                        running_comment += (" but its value was not within 50Hz of the "
                                            "true value ({:.1f}).".format(true_value))
                    # # Calculate difference between student answer and correct answer (debugging)
                    # difference = abs(true_value - student_var)
                    # running_comment += "(diff = {:.3g}Hz) ".format(difference)
                else:
                    running_comment += " but it was found to be callable (i.e. a function)."
            else:
                running_comment += " '{}' was not found.".format(param_name)

        # Formatting - remove superfluous newline character
        # running_comment = running_comment[:-2]

        if self.mode == 'batch':
            # In master.py, solution objects are instantiated which import the student
            # notebooks. Each time this happens, lines are added to the student_figure
            # object and are not removed when a given notebook ends (quirk of matplotlib).
            # Without dealing with this, the student_figure object would have lines
            # from previous students, resulting in fragile marking. To deal with this, clear
            # the student_figure object in this module.

            # NOTE: plt.clf(), plt.close(), plt.gcf().clear() all work after importing
            # matplotlib.pyplot as plt
            self.attempt.student_figure.clf()

        # NOTE: clf is not used in unit_test mode as it clears the figure so that
        # if the mark function is called repeatedly in the same notebook Without
        # rerunning the plot cell, 0 marks will be given for the plot.

        # Return marking tuple to main
        return running_comment, running_mark, MAX_MARK

    def task4(self):
        """Mark the fourth task (out of 5): Defining my_box_process function

        This function should mimic the behaviour of an RL low pass filter circuit

        Marking points
        --------------
        1 mark for  function formally correctly defined
        1 mark for 1-2/3 tests gave the expected output
        3 marks for 3/3 tests gave the expected output

        :return: tuple; (Marking comment: string, Mark: int, MAX_MARK: 5)
        """
        MAX_MARK = 5
        running_mark = 0
        running_comment = ""
        box_func_name = "my_box_process"

        # No marks if function does not exist or not callable
        if not hasattr(self.attempt, box_func_name):
            running_comment += " A function '{}' was not found.".format(box_func_name)
            return running_comment, running_mark, MAX_MARK

        running_comment += " '{}' was found".format(
            box_func_name)
        student_box_process = getattr(self.attempt, box_func_name)
        if not callable(student_box_process):
            running_comment += " but it was not callable."
            return running_comment, running_mark, MAX_MARK

        n_arguments = len(inspect.getfullargspec(student_box_process)[0])
        if n_arguments != 4:
            running_comment += (", it was callable, but expected {} arguments "
                                "when it should have expected 4."
                                "".format(n_arguments))
            return running_comment, running_mark, MAX_MARK

        running_comment += " and it was callable, expecting the correct number of arguments (4)."
        running_mark += 1

        # student_box_process function with correct arg. number; proceed with marking

        # Constant signal to provide a basic test of student function against self.my_box_process
        # Values below give cutoff freq of ~50Hz
        # f_max = 100^-
        R1 = 30
        L1 = 0.1
        N1 = 1700
        t_max1 = 0.5
        time_array1 = np.linspace(0, t_max1, num=N1, endpoint=False)
        amplitude_array1 = 5 * np.ones(time_array1.size)
        params1 = (time_array1, amplitude_array1, R1, L1)

        # More complex test using sinc which is a top-hat in freq domain
        # Cutoff freq ~ 1320Hz
        # After FFT, sinc(2*pi*f_max*t) will be non-zero only for f < f_max
        f_max2 = 2000  # Hz
        R2 = 500
        L2 = 0.06
        N2 = 2100
        # N = 2BL where FFT sample intervals are [0, L) (t-dom) and [-B, B) (freq dom)
        t_max2 = N2 / (2 * f_max2)
        time_array2 = np.linspace(0, t_max2, num=N2, endpoint=False)
        amplitude_array2 = np.sinc(2 * np.pi * f_max2 * time_array2)
        params2 = (time_array2, amplitude_array2, R2, L2)

        # Another complex test which uses the Fourier series idea
        # Cutoff freq ~ 1600Hz
        f_max3 = 2000  # Hz
        R3 = 100
        L3 = 0.01
        N3 = 3100
        t_max3 = N3 / (2 * f_max3)
        time_array3 = np.linspace(0, t_max3, num=N3, endpoint=False)
        amplitude_array3 = 0
        for n in range(2000):
            amplitude_array3 += np.sin(2 * np.pi * n * time_array3)
        params3 = (time_array3, amplitude_array3, R3, L3)

        # tuple of parameters for each test function (unpacked as *args)
        test_params = (params1, params2, params3)

        # For each test signal & parameters, see if student's process func
        # returns values close to our model
        # function, my_box_process. Add 5, 4, or 2 marks accordingly.
        tests_successful = 0
        for i, args in enumerate(test_params):
            try:
                if np.allclose(self.my_box_process(*args),
                               student_box_process(*args), rtol=0, atol=1e-8):
                    tests_successful += 1
                    # running_comment += "test {} succeeded. ".format(i + 1)
                    continue
                # running_comment += "test {} returned incorrect values. ".format(i + 1)
            except Exception:
                pass
                # running_comment += "test {} threw an exception. ".format(i + 1)
        if tests_successful == 3:
            running_mark += 4
        elif tests_successful == 2:
            running_mark += 1
        elif tests_successful == 1:
            running_mark += 1

        running_comment += (" Your Box produced correct output values for {}/3 of the "
                            "trial signals we attempted to"
                            " pass to it.".format(tests_successful))

        return running_comment, running_mark, MAX_MARK

    def mark(self, master_dir=None, new_call=False, mark_as_null=False):
        self.assignment_num = 2
        super().mark(master_dir, new_call, mark_as_null)

# =======================================================================================================

class Solution3(Solution):
    """
    Class responsible for marking the tasks of assignment 3
    """

    def set_rocket_params(self):
        self.m = self.in_interval(self.R_F, self.R_G, self.factor) # mass
        self.tmax = self.in_interval(self.R_J, self.R_K, self.factor) # tmax
        self.o_left = self.in_interval(self.R_L, self.R_M, self.key3 / self.norm) # o_left
        self.o_right = self.in_interval(self.R_L, self.R_M, self.key4 / self.norm) # o_right

    def set_student_params(self, attempt):
        if hasattr(attempt, 'm'):
            self.student_m = attempt.m
        else:
            self.student_m = self.m
        if hasattr(attempt, 'tmax'):
            self.student_tmax = attempt.tmax
        else:
            self.student_tmax = self.tmax
        if hasattr(attempt, 'o_left'):
            self.student_o_left = attempt.o_left
        else:
            self.student_o_left = self.o_left
        if hasattr(attempt, 'o_right'):
            self.student_o_right = attempt.o_right
        else:
            self.student_o_right = self.o_right

    def acc2thrust_left(self, acceleration):
        thrust = acceleration * self.student_m + self.student_o_left
        return thrust
    def acc2thrust_right(self, acceleration):
        thrust = acceleration * self.student_m + self.student_o_right
        return thrust

    def position_feedback(self, pos, target):
        gain = 0.1
        accL = gain * max(target - pos, 0)
        accR = gain * max(pos - target, 0)
        left_thrust = self.acc2thrust_left(accL)
        right_thrust = self.acc2thrust_right(accR)
        return left_thrust, right_thrust

    def mark(self, master_dir=None, new_call=False, mark_as_null=False):
        self.assignment_num = 3
        super().mark(master_dir, new_call, mark_as_null)

    def task1(self):
        """
        Mark the first task: System Identification, Marks 8

        Marking points
        --------------
        `obtain the numerical values for oo (left and right), tmaxtmax and mm 
        with the following accuracy:'
        m, rocket mass, +- 1.0 kg               -> 2 marks
        o_left, o_right, thrust offset, +- 1 N  -> 4 marks
        tmax, maximum thrust, +- 10.0 N, 1 mark -> 2 marks

        o_left = lefto
        o_right = righto
        m = mass
        tmax = tMax

        :return: tuple; (Marking comment: string, Mark: int, MAX_MARK)
        """

        MAX_MARK = 8
        running_comment = ""
        running_mark = 0
        self.set_rocket_params()
        vars = ['m', 'o_left', 'o_right', 'tmax']
        values = [self.m, self.o_left, self.o_right, self.tmax]
        # check for 2 times the required accuracy to be a bit generous
        tol_abs = [2.0, 2.0, 2.0, 20]
        commnent, num_correct = check_variables(self.attempt, vars, values=values, tol_abs=tol_abs, Abs=True)
        running_comment += commnent
        running_mark += num_correct * 2

        #print ("Task 1 done")
        return running_comment, running_mark, MAX_MARK

    def task2(self):
        """
        Mark the second task: There and Stop, Marks 2

        Marking points
        --------------
        First write two utility functions that make use of the known thruster
        calibration to compute the required thrust for a given acceleration that
        we want to apply to the rocket. Each function should accept a Numpy array
        and return a Numpy array (both of float values).:
        def acc2thrust_left(acceleration):
        def acc2thrust_right(acceleration):
        -> 1 mark
        `Now make use of your utility functions to move the rocket exactly 100 
        meters to the right such that it stops there.'
        student_track1 = my_rocket.get_flight_data() 
        -> 1 mark
        
        :return: tuple; (Marking comment: string, Mark: int, MAX_MARK)
        """

        MAX_MARK = 2
        running_comment = ""
        running_mark = 0
        self.set_rocket_params()
        self.set_student_params(self.attempt)
        xdata =[]
        ydata =[]        
        a = np.array([])
        b = np.array([])
        xdata.append(np.array([0.0, 0.1, 0.3]))
        ydata.append(self.acc2thrust_left(xdata[0]))
        comment, correct_a = check_function(self.attempt, 'acc2thrust_left', rtol=0.01, xdata=xdata, ydata=ydata, Abs=True)
        running_comment += comment
        if correct_a == 2:
            try:
                a = self.attempt.acc2thrust_left(np.array([0.0, 1.0]))
            except ValueError:
                running_comment += " 'acc2thrust_left' function caused an exception when called."        
        ydata =[]        
        ydata.append(self.acc2thrust_right(xdata[0]))
        comment, correct_b = check_function(self.attempt, 'acc2thrust_right', rtol=0.01, xdata=xdata, ydata=ydata, Abs=True)
        if correct_b != 2:
            # check whether student has used negative acceleration instead:
            xdata = []
            xdata.append(np.array([0.0, -0.1, -0.3]))
            comment, correct_b = check_function(self.attempt, 'acc2thrust_right', rtol=0.01, xdata=xdata, ydata=ydata, Abs=True)
        running_comment += comment
        if correct_b == 2:
            try:
                a = self.attempt.acc2thrust_right(np.array([0.0, 1.0]))
            except:
                running_comment += " 'acc2thrust_right' function caused an exception when called."
        #aa = self.attempt.acc2thrust_right(xdata[0])
        #print(f"aa {aa}")        
        #print(f"ydata {ydata}")
        #print(f"student_o_right {self.student_o_right}")
        #print(f"o_right {self.o_right}")
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            running_comment += " At least one of the 'acc2thrust' functions did not return numpy array."
        elif correct_a == 2 and correct_b == 2:
            running_mark += 1

        track, comment = check_track(self.attempt, "student_track1")
        running_comment += comment
        if track is not None:
            comment, t, x, v, a = track_data(track)
            running_comment += comment
            #print("---------------1 "+str(len(t)))
            #print("---------------1 "+str(len(v)))
            if a is not None and len(x) > 3:
                x0 = x[0]
                x1 = x[-1]
                v1 = v[-1]
                if np.isclose(x1-x0, 100.0, atol=1):
                    running_comment += " The data shows that you successfully moved 100 meters to the right"
                    if max(x) - x0 > 101.0:
                        running_comment += " but your track shows an overshoot of more than 1 meter."
                    else:
                        running_comment += " without overshooting more than 1 meter,"
                        if abs(v1)<0.1:
                            running_comment += " and stopping at the end point (v={:.2f}m/s).".format(np.abs(v1))
                            running_mark += 1
                        else:   
                            running_comment += " but your end volicity was too high ({:.2f}m/s).".format(np.abs(v1))
                else:
                    running_comment += " The final position was not +100 meters to the right from the start (x1-x0={:.2f}).".format(x1-x0)
        #print ("Task 2 done")
        return running_comment, running_mark, MAX_MARK

    def task3(self):
        """
        Mark the third task: Feedback control, Marks 6

        Marking points
        --------------
        `Write a function that accepts 2 input arguments: the x-position of the rocket 
        and the target position, and returns two values: left_thrust and right_thrust'
        def position_feedback(pos, target):
            return left_thrust, right_thrust
        -> 1 marks

        Test your function by trying to move the rocket to a target position which is 
        100 m to the right of your starting position. Adjust the gain in your function 
        such that you can clearly see the rocket oscillation around the target position 
        during an approximately 60 s long flight. 
        student_track2 = my_rocket.get_flight_data() 
        -> 2 marks

        `Write a new feedback function to compute the rocket thrust. The function should 
        accept three input arguments, position, velocity and target position. Internally 
        it should use two gain values, a position gain pos_gain and a velocity gain 
        v_gain, and return the thrust values as before. '
        def damped_feedback(pos, v, target):
            return left_thrust, right_thrust
        -> 1 marks

        Adjust the two gains such that the rocket reaches its target quickly and does
        not overshoot the target by more than 1 m.'
        student_track3 = my_rocket.get_flight_data() 
        -> 2 marks
        
        :return: tuple; (Marking comment: string, Mark: int, MAX_MARK)
        """
        MAX_MARK = 6
        running_comment = ""
        running_mark = 0
        self.set_rocket_params()
        self.set_student_params(self.attempt)
        comment, correct = check_function(self.attempt, 'position_feedback')
        running_comment += comment
        if correct == 1:
            try:
                out = self.attempt.position_feedback(0.0, 0.0)
            except:
                running_comment += " The case position=0.0, target=0.0 is handled incorrectly."
            try:
                out = self.attempt.position_feedback(0.0, 0.0)
                if len(out) == 2:
                    xdata = [0.001, 0.002, 0.003] 
                    ydata = []
                    for x in xdata:
                        thrusts = np.array(self.attempt.position_feedback(0.0, x))
                        ydata.append((thrusts[0]-self.student_o_left)/x)
                        #print ("--1---------"+ str(thrusts))
                        #print ("--2---------"+ str(self.student_o_left))
                        #print ("--3---------"+ str(ydata[-1]))
                    if np.allclose(np.array(ydata),ydata[0], rtol=5e-02):
                        running_comment += " The computed thrust is proportional to the distance to target."
                        running_mark += 1
                    else:
                        running_comment += " The computed thrust is not proportional to the distance to target."
                else:
                    running_comment += " Your function did not return the correct number of values."   
            except:
                running_comment += " Your function caused an exception when called."

        track, comment = check_track(self.attempt, "student_track2", length=3000)
        running_comment += comment
        if track is not None:
            comment, t, x, v, a = track_data(track)
            running_comment += comment
            x0 = x[0]
            x1 = x-100.0
            v1 = v[-1]
            if a is not None:
                # count zero crossings 
                oscillations = ((x1[:-1] * x1[1:]) < 0).sum() + (x1 == 0).sum()
                #oscillations = (np.diff(np.sign(x1)) != 0).sum() - (x1 == 0).sum()
                running_comment += " Your rocket oscillated {} times around the target position.".format(oscillations)
                if oscillations >=2:
                    running_mark += 2

        comment, correct = check_function(self.attempt, 'damped_feedback')
        running_comment += comment
        if correct:
            try:
                out = self.attempt.damped_feedback(0.0, 0.0, 0.0)
            except:
                running_comment += " The case position=0.0, velocity=0.0, target=0.0 is handled incorrectly."
            try:
                out = self.attempt.damped_feedback(0.0, 0.0, 0.0)
                if len(out) == 2:
                    running_comment += " The function returned the right number of values."
                    running_mark += 1
            except:
                running_comment += " Your function caused an exception when called."

        track, comment = check_track(self.attempt, "student_track3", length=300)
        running_comment += comment
        if track is not None:
            comment, t, x, v, a = track_data(track)
            running_comment += comment
            x0 = x[0]
            x1 = x[-1]
            v1 = v[-1]
            if a is not None:
                if np.isclose(x1, 100.0, atol=1):
                    running_comment += " The data shows that you successfully moved 100 meters to the right"
                    if max(x) - x0 > 101.0:
                        running_comment += " but your track shows an overshoot of more than 1 meter ({:.2f}).".format(max(x)-x0-100.0)
                    else:
                        running_comment += " without overshooting more than 1 meter,"
                        running_mark += 1
                        if abs(v1)<0.1:
                            running_comment += " and stopping at the end point (v={:.2f}m/s).".format(np.abs(v1))
                            running_mark += 1
                        else:   
                            running_comment += " but your end volicity was too high ({:.2f}m/s).".format(np.abs(v1))
                else:
                    running_comment += " The final position was not +100 meters to the right from the start (x1-x0={:.2f}).".format(x1-x0)

        #print ("Task 3 done")
        return running_comment, running_mark, MAX_MARK

    def task4(self):
        """
        Mark the fourth task: Landing the rocket, Marks 4

        Marking points
        --------------
        `Write a loop at that attempts 40 landings, the number of successful landings
        will be recorded automatically. Note that you might not be able to succeed in
        100% of your your attempts. You can achieve full marks by landing 70% of your
        rockets successfully.'

        my_rocket.successful_landing_counter >=  5 -> 1 mark
        my_rocket.successful_landing_counter >= 10 -> 2 marks
        my_rocket.successful_landing_counter >= 20 -> 3 marks
        my_rocket.successful_landing_counter >= 30 -> 4 marks

        :return: tuple; (Marking comment: string, Mark: int, MAX_MARK)
        """

        MAX_MARK = 4
        running_comment = ""
        running_mark = 0

        self.set_rocket_params()
        self.set_student_params(self.attempt)

        # Check if the tracks list exists and has exactly 40 entries
        if hasattr(self.attempt, 'tracks'):
            student_tracks = self.attempt.tracks
            if type(student_tracks) in [list,tuple]:
                number_of_tracks = len(student_tracks)
                if number_of_tracks > 40:
                    comment = " The list `tracks` contains more rocket tracks than expected (" + str(number_of_tracks) + " instead of 40)."
                    running_comment += comment
                elif number_of_tracks < 40:
                    comment = " The list `tracks` contains fewer rocket tracks than expected (" + str(number_of_tracks) + " instead of 40)."
                    running_comment += comment
                else: # exactly 40 tracks
                    running_comment += " We found the list `tracks` with 40 rocket tracks as expected."
                    faults = 0
                    for track in student_tracks:
                        t = track[:, 0]
                        #if len(t) < 300:
                        #    track data too short
                        #    fault += 1
                        #    continue
                        x = track[:, 1]
                        v = np.diff(x)/np.diff(t)
                        a = np.diff(v)/np.diff(t[0:-1])
                        if a.size > 2 and max(np.abs(a) > 20):
                            #print(" - ------------  max acc " + str(max(np.abs(a))))
                            # anomal acceleration
                            faults += 1
                            continue
                    if faults>0:
                        running_comment += " Some of the tracks showed unphysical acceleration."
                    else:
                        landings = self.attempt.my_rocket.successful_landing_counter
                        running_comment += " You successfully landed {} out of {} attempts.".format(landings, number_of_tracks)
                        if landings >= 5:
                            running_mark += 1
                        if landings >= 10:
                            running_mark += 1
                        if landings >= 20:
                            running_mark += 1
                        if landings >= 30:
                            running_mark += 1
            else:
                running_comment += " We found a variable 'tracks' but it is neither a list nor a tuple."
        else:
            running_comment += " We could not find a list 'tracks'."

        return running_comment, running_mark, MAX_MARK

def check_track(attempt, track_name, length=None):
    comment = ""
    track = None
    if hasattr(attempt, track_name):
        comment += " We found '{}'".format(track_name)
        track = getattr(attempt, track_name)
        if isinstance(track, np.ndarray):
            comment += " of the right data type"
            if length is not None:
                if length<track.size:
                    comment += " and sufficient size"
                else:
                    comment += " which did not contain enough data"
                    track = None
        else:
            comment += " but is was not a Numpy array"
            track = None    
        comment += "."
    return track, comment

def track_data(track):
    comment = ""
    t = track[:, 0]
    x = track[:, 1]
    v = np.diff(x)/np.diff(t)
    a = np.diff(v)/np.diff(t[0:-1])
    if a.size >2:
        if max(np.abs(a) > 8):
            comment += " The track data shows unphysical accelearation (a_max={:.2f}).".format(np.max(np.abs(a)))
            a = None
    return comment, t, x, v, a

def check_function(attempt, func, *args, rtol=1e-12, xdata=[], ydata=[], Abs=False, **kwargs):
    comment = " Your notebook"
    correct = 0
    if hasattr(attempt, func):
        comment += " contains `{}'".format(func)
        student_func = getattr(attempt, func)
        if callable(student_func):
            correct += 1
            if len(xdata) and len(ydata):
                try:
                    for idx, x in enumerate(xdata):
                        y_student = np.array(student_func(x, *args))
                        if Abs:
                            if np.allclose(np.abs(ydata[idx]), np.abs(y_student), rtol=rtol, equal_nan=False):
                                correct += 1
                        else:
                            if np.allclose(ydata[idx], y_student, rtol=rtol, equal_nan=False):
                                correct += 1
                        #else:
                        #    comment += "Wrong result: correct {}, student: {}".format(ydata[idx], y_student)
                    comment += " which returned {} out of {} correct numerical values for test input data".format(correct-1, len(xdata))
                except:
                    comment += " but the function caused an exception when called"
        else:
            comment += " but it is not callable (a function)"
    else:
        comment += " does not contain `{}'".format(func)
    comment += "."
    return comment, correct


def check_variables(attempt, vars, values=[], tol_rel=[], tol_abs=[], Abs=False):
    number_correct = 0
    comment = " We checked if "
    for idx, var in enumerate(vars):
        comment += "{}".format(var)
        if idx == len(vars) - 2:
            comment += " and "
        elif idx < len(vars) - 2:
            comment += ", "
    comment += " where present in your notebook: "

    # Setting default absolutetolerance to 1e-12 when not set
    if len(tol_rel) == 0 and len(tol_abs) == 0:
        tol_abs = 1.0e-12 * np.ones(len(vars))
        tol_rel = np.ones(len(vars))
    elif len(tol_rel) == 0:
        tol_rel = np.zeros(len(vars))
    else:
        tol_abs = np.zeros(len(vars))

    for idx, var in enumerate(vars):
        if idx > 0:
            comment += ", "
        if not hasattr(attempt, var):
            comment += "'{}' was not found".format(var)
            continue
        else:
            comment += "'{}' was found".format(var)
        if callable(var):
            comment += ("'{}' was found but is "
                        "callable (i.e. a function not a variable).").format(var)
            continue
        if len(values) == len(vars):
            student_value = getattr(attempt, var)
            if len(tol_rel) == len(vars):
                a = values[idx]
                b = student_value
                if Abs:
                    # checking only absolutes
                    b=np.abs(b)
                    a=np.abs(a)
                if np.isclose(a, b, rtol=tol_rel[idx], atol=tol_abs[idx], equal_nan=False):
                    number_correct += 1
                    comment += " and has the correct value"
                else:
                    comment += " but its value deviates from the expected one ({:f})".format(a)
                    comment += " by {:2e}".format(abs(a-b))            
    comment += "."
    return comment, number_correct
