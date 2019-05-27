from abc import ABC, abstractmethod
import numpy as np


class Environment(ABC):
    def __init__(self, programs_library, prog_to_func, prog_to_precondition, prog_to_postcondition):
        """

        Args:
            programs_library (dict): Maps a program name to a level and a bool indicating whether recursive
            prog_to_func (dict): Maps 0 level programs to their implementation function
            prog_to_precondition (dict): Maps a program name to the function that states whether its preconditions are fulfilled
            prog_to_postcondition (dict): Maps a program name to the function that states whether its postconditions are fulfilled
        """
        super().__init__()

        self.programs_library = programs_library
        self.prog_to_func = prog_to_func
        self.prog_to_precondition = prog_to_precondition
        self.prog_to_postcondition = prog_to_postcondition

        self.programs = list(self.programs_library.keys())
        self.primary_actions = [prog for prog in self.programs_library if self.programs_library[prog]['level'] <= 0]
        self.mask = dict((p, self._get_available_actions(p)) for p in self.programs_library if self.programs_library[p]["level"] > 0)
        # correct mask for recursive programs
        for program_name, program_mask in self.mask.items():
            if self.programs_library[program_name]['recursive']:
                program_mask[self.programs_library[program_name]['index']] = 1

        self.prog_to_idx = dict((prog, elems["index"]) for prog, elems in self.programs_library.items())
        self.idx_to_prog = dict((idx, prog) for (prog, idx) in self.prog_to_idx.items())

        self.maximum_level = max([x['level'] for prog, x in self.programs_library.items()])

        self.current_task_index = None
        self.tasks_dict = {}
        self.tasks_list = []

        self.has_been_reset = False

    def get_maximum_level(self):
        """
        Returns the maximum program level.

        Returns:
            maximum level
        """
        return self.maximum_level

    def _get_available_actions(self, program):
        """
        Args:
          program (str): program name

        Returns:
            mask

        """
        level_prog = self.programs_library[program]["level"]
        assert level_prog > 0
        mask = np.zeros(len(self.programs))
        for prog, elems in self.programs_library.items():
            if elems["level"] < level_prog:
                mask[elems["index"]] = 1
        return mask

    def get_program_from_index(self, program_index):
        """Returns the program name from its index.

        Args:
          program_index: index of desired program

        Returns:
          the program name corresponding to program index

        """
        return self.idx_to_prog[program_index]

    def get_num_non_primary_programs(self):
        """Returns the number of programs with level > 0.

        Returns:
            the number of available programs of level > 0 (the number of non primary programs)

        """
        return len(self.programs) - len(self.primary_actions)

    def get_num_programs(self):
        """Returns the number of available programs.

        Returns:
            the number of available programs (all levels)

        """
        return len(self.programs)

    def get_program_level_from_index(self, program_index):
        """
        Args:
            program_index: program index

        Returns:
            the level of the program
        """
        program = self.get_program_from_index(program_index)
        return self.programs_library[program]['level']

    def get_reward(self):
        """Returns a reward for the current task at hand.

        Returns:
            1 if the task at hand has been solved, 0 otherwise.

        """
        task_init_state = self.tasks_dict[len(self.tasks_list)]
        state = self.get_state()
        current_task = self.get_program_from_index(self.current_task_index)
        current_task_postcondition = self.prog_to_postcondition[current_task]
        return int(current_task_postcondition(task_init_state, state))

    def start_task(self, task_index):
        """Function used to begin a task. The task at hand defines the reward signal and stop boolean
        returned by the function step. This function resets the environment as well.

        Args:
          task_index: the index corresponding to the program(task) to start

        Returns:
          the environment observation
        """
        task_name = self.get_program_from_index(task_index)
        assert self.prog_to_precondition[task_name], 'cant start task {} ' \
                                                     'because its precondition is not verified'.format(task_name)
        self.current_task_index = task_index
        self.tasks_list.append(task_index)

        if len(self.tasks_dict.keys()) == 0:
            # reset env
            self.reset_env()

        # store init state
        init_state = self.get_state()
        self.tasks_dict[len(self.tasks_list)] = init_state

        return self.get_observation()

    def end_task(self):
        """
        Ends the last tasks that has been started.
        """
        del self.tasks_dict[len(self.tasks_list)]
        self.tasks_list.pop()
        if self.tasks_list:
            self.current_task_index = self.tasks_list[-1]
        else:
            self.current_task_index = None
            self.has_been_reset = False

    def end_all_tasks(self):
        self.tasks_dict = {}
        self.tasks_list = []
        self.has_been_reset = False

    def act(self, primary_action):
        """Apply a primary action that modifies the environment.

        Args:
          primary_action: action to apply

        Returns:
          the environment observation after the action has been applied

        """
        assert self.has_been_reset, 'Need to reset the environment before acting'
        assert primary_action in self.primary_actions, 'action {} is not defined'.format(primary_action)
        self.prog_to_func[primary_action]()
        return self.get_observation()

    def render(self):
        """Print a graphical representation of the current environment state"""
        assert self.has_been_reset, 'Need to reset the environment before rendering'
        s = self.get_state()
        str = self.get_state_str(s)
        print(str)

    def get_mask_over_actions(self, program_index):
        """Returns the mask of possible programs to call given the current program.

        Args:
          program_index: index of program for which is wanted the mask of possible programs to call

        Returns:
          mask of possible programs to call
        """
        program = self.get_program_from_index(program_index)
        assert program in self.mask, "Error program {} provided is level 0".format(program)
        mask = self.mask[program].copy()
        # remove actions when pre-condition not satisfied
        for program, program_dict in self.programs_library.items():
            if not self.prog_to_precondition[program]():
                mask[program_dict['index']] = 0
        return mask

    @abstractmethod
    def compare_state(self, state1, state2):
        """Compares two states to determine whether they are the same state.
        Args:
            state1 (tuple): Describes the environment
            state2 (tuple): Describes the environment
        returns:
            bool: The return value. True if state1 and state2 are the same, False otherwise.
        """
        pass

    @abstractmethod
    def reset_env(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_observation(self):
        pass

    @abstractmethod
    def get_observation_dim(self):
        pass

    @abstractmethod
    def reset_to_state(self, state):
        """
        Args:
            state (tuple): Describes the environment state
        """
        pass

    @abstractmethod
    def get_state_str(self, state):
        """

        Args:
            state (tuple): Describes the environment state
       Returns:
            String describes the environment in a more human-friendly way
        """
        pass