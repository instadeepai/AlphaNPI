import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from environments.environment import Environment


class RecursiveListEnvEncoder(nn.Module):
    '''
    Implement an encoder (f_enc) specific to the List environment. It encodes observations e_t into
    vectors s_t of size D = encoding_dim.
    '''

    def __init__(self, observation_dim, encoding_dim):
        super(RecursiveListEnvEncoder, self).__init__()
        self.l1 = nn.Linear(observation_dim, 100)
        self.l2 = nn.Linear(100, encoding_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = torch.tanh(self.l2(x))
        return x


class RecursiveListEnv(Environment):
    def __init__(self, length=10, encoding_dim=32):

        assert length > 1, "length must be a positive integer superior to 1"
        self.length = length
        self.start_pos = 0
        self.end_pos = length-1
        self.scratchpad_ints = np.zeros((length,))
        self.p1_pos = 0
        self.p2_pos = 0
        self.encoding_dim = encoding_dim
        self.has_been_reset = False

        self.programs_library = {'PTR_1_LEFT': {'level': 0, 'recursive': False},
                                 'STOP': {'level': -1, 'recursive': False},
                                 'PTR_2_LEFT': {'level': 0, 'recursive': False},
                                 'PTR_1_RIGHT': {'level': 0, 'recursive': False},
                                 'PTR_2_RIGHT': {'level': 0, 'recursive': False},
                                 'SWAP': {'level': 0, 'recursive': False},
                                 'RSHIFT': {'level': 1, 'recursive': False},
                                 'LSHIFT': {'level': 1, 'recursive': False},
                                 'COMPSWAP': {'level': 1, 'recursive': False},
                                 'RESET': {'level': 2, 'recursive': True},
                                 'BUBBLE': {'level': 2, 'recursive': True},
                                 'BUBBLESORT': {'level': 3, 'recursive': True}}

        for i, key in enumerate(sorted(list(self.programs_library.keys()))):
            self.programs_library[key]['index'] = i

        self.prog_to_func = {'STOP': self._stop,
                             'PTR_1_LEFT': self._ptr_1_left,
                             'PTR_2_LEFT': self._ptr_2_left,
                             'PTR_1_RIGHT': self._ptr_1_right,
                             'PTR_2_RIGHT': self._ptr_2_right,
                             'SWAP': self._swap}

        self.prog_to_precondition = {'STOP': self._stop_precondition,
                                     'RSHIFT': self._rshift_precondition,
                                     'LSHIFT': self._lshift_precondition,
                                     'COMPSWAP': self._compswap_precondition,
                                     'RESET': self._reset_precondition,
                                     'BUBBLE': self._bubble_precondition,
                                     'BUBBLESORT': self._bubblesort_precondition,
                                     'PTR_1_LEFT': self._ptr_1_left_precondition,
                                     'PTR_2_LEFT': self._ptr_2_left_precondition,
                                     'PTR_1_RIGHT': self._ptr_1_right_precondition,
                                     'PTR_2_RIGHT': self._ptr_2_right_precondition,
                                     'SWAP': self._swap_precondition}

        self.prog_to_postcondition = {'RSHIFT': self._rshift_postcondition,
                                     'LSHIFT': self._lshift_postcondition,
                                     'COMPSWAP': self._compswap_postcondition,
                                     'RESET': self._reset_postcondition,
                                     'BUBBLE': self._bubble_postcondition,
                                     'BUBBLESORT': self._bubblesort_postcondition}

        super(RecursiveListEnv, self).__init__(self.programs_library, self.prog_to_func,
                                               self.prog_to_precondition, self.prog_to_postcondition)

    def _decr_length_right(self):
        assert self._decr_length_right_precondition(), 'precondition not verified'
        if self.end_pos > self.start_pos:
            self.end_pos -= 1

    def _decr_length_right_precondition(self):
        return self.end_pos > self.start_pos and (self.p1_pos < self.end_pos and self.p2_pos < self.end_pos)

    def _decr_length_left(self):
        assert self._decr_length_left_precondition(), 'precondition not verified'
        if self.start_pos < self.end_pos:
            self.start_pos += 1

    def _decr_length_left_precondition(self):
        return self.start_pos < self.end_pos and (self.p1_pos > self.start_pos and self.p2_pos > self.start_pos)

    def _incr_length_left(self):
        assert self._incr_length_left_precondition(), 'precondition not verified'
        if self.start_pos > 0:
            self.start_pos -= 1

    def _incr_length_left_precondition(self):
        return self.start_pos > 0

    def _incr_length_right(self):
        assert self._incr_length_right_precondition(), 'precondition not verified'
        if self.end_pos < self.length-1:
            self.end_pos += 1

    def _incr_length_right_precondition(self):
        return self.end_pos < self.length-1

    def _ptr_1_left(self):
        assert self._ptr_1_left_precondition(), 'precondition not verified'
        if self.p1_pos > self.start_pos:
            self.p1_pos -= 1

    def _ptr_1_left_precondition(self):
        return self.p1_pos > self.start_pos

    def _stop(self):
        assert self._stop_precondition(), 'precondition not verified'
        pass

    def _stop_precondition(self):
        return True

    def _ptr_2_left(self):
        assert self._ptr_2_left_precondition(), 'precondition not verified'
        if self.p2_pos > self.start_pos:
            self.p2_pos -= 1

    def _ptr_2_left_precondition(self):
        return self.p2_pos > self.start_pos

    def _ptr_1_right(self):
        assert self._ptr_1_right_precondition(), 'precondition not verified'
        if self.p1_pos < self.end_pos:
            self.p1_pos += 1

    def _ptr_1_right_precondition(self):
        return self.p1_pos < self.end_pos

    def _ptr_2_right(self):
        assert self._ptr_2_right_precondition(), 'precondition not verified'
        if self.p2_pos < self.end_pos:
            self.p2_pos += 1

    def _ptr_2_right_precondition(self):
        return self.p2_pos < self.end_pos

    def _swap(self):
        assert self._swap_precondition(), 'precondition not verified'
        self.scratchpad_ints[[self.p1_pos, self.p2_pos]] = self.scratchpad_ints[[self.p2_pos, self.p1_pos]]

    def _swap_precondition(self):
        return self.p1_pos != self.p2_pos

    def _compswap_precondition(self):
        list_length = self.end_pos - self.start_pos + 1
        bool = list_length > 1
        bool &= self.p1_pos < self.end_pos
        bool &= self.p2_pos == self.p1_pos or self.p2_pos == (self.p1_pos + 1)
        return bool

    def _compswap_postcondition(self, init_state, state):
        new_scratchpad_ints, new_p1_pos, new_p2_pos, new_start_pos, new_end_pos = init_state
        new_scratchpad_ints = np.copy(new_scratchpad_ints)
        if new_p1_pos == new_p2_pos and new_p2_pos < new_end_pos:
            new_p2_pos += 1
        idx_left = min(new_p1_pos, new_p2_pos)
        idx_right = max(new_p1_pos, new_p2_pos)
        if new_scratchpad_ints[idx_left] > new_scratchpad_ints[idx_right]:
            new_scratchpad_ints[[idx_left, idx_right]] = new_scratchpad_ints[[idx_right, idx_left]]
        new_state = (new_scratchpad_ints, new_p1_pos, new_p2_pos, new_start_pos, new_end_pos)
        return self.compare_state(state, new_state)

    def _lshift_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_start_pos, init_end_pos = init_state
        scratchpad_ints, p1_pos, p2_pos, start_pos, end_pos = state
        bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
        bool &= init_start_pos == start_pos
        bool &= init_end_pos == end_pos
        if init_p1_pos > init_start_pos:
            bool &= p1_pos == (init_p1_pos-1)
        else:
            bool &= p1_pos == init_p1_pos

        if init_p2_pos > init_start_pos:
            bool &= p2_pos == (init_p2_pos-1)
        else:
            bool &= p2_pos == init_p2_pos

        return bool

    def _rshift_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_start_pos, init_end_pos = init_state
        scratchpad_ints, p1_pos, p2_pos, start_pos, end_pos = state
        bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
        bool &= init_start_pos == start_pos
        bool &= init_end_pos == end_pos
        if init_p1_pos < init_end_pos:
            bool &= p1_pos == (init_p1_pos+1)
        else:
            bool &= p1_pos == init_p1_pos
        if init_p2_pos < init_end_pos:
            bool &= p2_pos == (init_p2_pos+1)
        else:
            bool &= p2_pos == init_p2_pos
        return bool

    def _reset_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_start_pos, init_end_pos = init_state
        scratchpad_ints, p1_pos, p2_pos, start_pos, end_pos = state
        bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
        bool &= init_start_pos == start_pos
        bool &= init_end_pos == end_pos
        bool &= (p1_pos == start_pos and p2_pos == start_pos)
        return bool

    def _bubblesort_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_start_pos, init_end_pos = init_state
        scratchpad_ints, p1_pos, p2_pos, start_pos, end_pos = state
        bool = init_start_pos == start_pos
        bool &= init_end_pos == end_pos
        # check if list is sorted
        bool &= np.all(scratchpad_ints[:end_pos] <= scratchpad_ints[(start_pos+1):(end_pos+1)])
        return bool

    def _bubble_postcondition(self, init_state, state):
        new_scratchpad_ints, new_p1_pos, new_p2_pos, new_start_pos, new_end_pos = init_state
        new_scratchpad_ints = np.copy(new_scratchpad_ints)
        for idx in range(new_start_pos, new_end_pos):
            if new_scratchpad_ints[idx+1] < new_scratchpad_ints[idx]:
                new_scratchpad_ints[[idx, idx+1]] = new_scratchpad_ints[[idx+1, idx]]
        new_p1_pos = new_end_pos
        new_p2_pos = new_end_pos
        new_state = (new_scratchpad_ints, new_p1_pos, new_p2_pos, new_start_pos, new_end_pos)
        return self.compare_state(state, new_state)

    def _lshift_precondition(self):
        return self.p1_pos > self.start_pos or self.p2_pos > self.start_pos

    def _rshift_precondition(self):
        return self.p1_pos < self.end_pos or self.p2_pos < self.end_pos

    def _bubble_precondition(self):
        bubble_index = self.programs_library['BUBBLE']['index']
        if self.current_task_index != bubble_index:
            bool = self.p1_pos == self.start_pos
            bool &= ((self.p2_pos == self.start_pos) or (self.p2_pos == (self.start_pos+1)))
        else:
            bool = self.p1_pos == (self.start_pos + 1)
            bool &= ((self.p2_pos == self.start_pos) or (self.p2_pos == (self.start_pos + 1)))
            bool &= self._decr_length_left_precondition()
        return bool

    def _reset_precondition(self):
        bool = (self.p1_pos > self.start_pos or self.p2_pos > self.start_pos)
        reset_index = self.programs_library['RESET']['index']
        if self.current_task_index == reset_index:
            bool &= self._decr_length_right_precondition()
        return bool

    def _bubblesort_precondition(self):
        bool = self.p1_pos == self.start_pos
        bool &= self.p2_pos == self.start_pos
        bubblesort_index = self.programs_library['BUBBLESORT']['index']
        if self.current_task_index == bubblesort_index:
            bool &= self._decr_length_right_precondition()
        return bool

    def _one_hot_encode(self, digit, basis=10):
        """One hot encode a digit with basis.

        Args:
          digit: a digit (integer between 0 and 9)
          basis:  (Default value = 10)

        Returns:
          a numpy array representing the 10-hot-encoding of the digit

        """
        encoding = np.zeros(basis)
        encoding[digit] = 1
        return encoding

    def _one_hot_decode(self, one_encoding):
        """Returns digit associated to a one hot encoding.

        Args:
          a: numpy array representing the 10-hot-encoding of a digit.
          one_encoding:

        Returns:
          the digit encoded in one_encoding

        """
        return np.argmax(one_encoding)

    def reset_env(self):
        """Reset the environment. The list are values are draw randomly. The pointers are initialized at position 0
        (at left position of the list).

        """
        assert self.length > 1, "list length must be greater than 1"
        self.start_pos = 0
        self.end_pos = self.length-1
        self.scratchpad_ints = np.random.randint(10, size=self.length)
        current_task_name = self.get_program_from_index(self.current_task_index)
        if current_task_name == 'BUBBLE' or current_task_name == 'BUBBLESORT':
            init_pointers_pos1 = 0
            init_pointers_pos2 = 0
        elif current_task_name == 'RESET':
            while True:
                init_pointers_pos1 = int(np.random.randint(0, self.length))
                init_pointers_pos2 = int(np.random.randint(0, self.length))
                if not (init_pointers_pos1 == 0 and init_pointers_pos2 == 0):
                    break
        elif current_task_name == 'LSHIFT':
            while True:
                init_pointers_pos1 = int(np.random.randint(0, self.length))
                init_pointers_pos2 = int(np.random.randint(0, self.length))
                if not (init_pointers_pos1 == 0 and init_pointers_pos2 == 0):
                    break
        elif current_task_name == 'RSHIFT':
            while True:
                init_pointers_pos1 = int(np.random.randint(0, self.length))
                init_pointers_pos2 = int(np.random.randint(0, self.length))
                if not (init_pointers_pos1 == self.length - 1 and init_pointers_pos2 == self.length - 1):
                    break
        elif current_task_name == 'COMPSWAP':
            init_pointers_pos1 = int(np.random.randint(0, self.length - 1))
            init_pointers_pos2 = int(np.random.choice([init_pointers_pos1, init_pointers_pos1 + 1]))
        else:
            raise NotImplementedError('Unable to reset env for this program...')

        self.p1_pos = init_pointers_pos1
        self.p2_pos = init_pointers_pos2
        self.has_been_reset = True

    def get_observation(self):
        """Returns an observation of the current state.

        Returns:
            an observation of the current state
        """
        assert self.has_been_reset, 'Need to reset the environment before getting observations'

        p1_val = self.scratchpad_ints[self.p1_pos]
        p2_val = self.scratchpad_ints[self.p2_pos]
        is_sorted = int(self._is_sorted())
        pointers_same_pos = int(self.p1_pos == self.p2_pos)
        pt_1_left = int(self.p1_pos == self.start_pos)
        pt_2_left = int(self.p2_pos == self.start_pos)
        pt_1_right = int(self.p1_pos == self.end_pos)
        pt_2_right = int(self.p2_pos == self.end_pos)
        p1p2 = np.eye(10)[[p1_val, p2_val]].reshape(-1)  # one hot encoding of values at pointers pos
        bools = np.array([
            pt_1_left,
            pt_1_right,
            pt_2_left,
            pt_2_right,
            pointers_same_pos,
            is_sorted
        ])
        return np.concatenate((p1p2, bools), axis=0)

    def get_observation_dim(self):
        """

        Returns:
            the size of the observation tensor
        """
        return 2 * 10 + 6

    def get_state(self):
        """Returns the current state.

        Returns:
            the environment state

        """
        assert self.has_been_reset, 'Need to reset the environment before getting states'
        return np.copy(self.scratchpad_ints), self.p1_pos, self.p2_pos, self.start_pos, self.end_pos

    def reset_to_state(self, state):
        """

        Args:
          state: a given state of the environment
        reset the environment is the given state

        """
        self.scratchpad_ints = state[0].copy()
        self.p1_pos = state[1]
        self.p2_pos = state[2]
        self.start_pos = state[3]
        self.end_pos = state[4]

    def get_state_str(self, state):
        """Print a graphical representation of the environment state"""
        scratchpad = state[0].copy()
        p1_pos = state[1]
        p2_pos = state[2]
        start_pos = state[3]
        end_pos = state[4]
        scratchpad = scratchpad[start_pos:end_pos+1]
        str = 'list: {}, p1 : {}, p2 : {}, start_pos: {}, end_pos: {}'.format(scratchpad, p1_pos,
                                                                              p2_pos, start_pos, end_pos)
        return str

    def compare_state(self, state1, state2):
        """
        Compares two states.

        Args:
            state1: a state
            state2: a state

        Returns:
            True if both states are equals, False otherwise.

        """
        bool = np.array_equal(state1[0], state2[0])
        bool &= (state1[1] == state2[1])
        bool &= (state1[2] == state2[2])
        bool &= (state1[3] == state2[3])
        bool &= (state1[4] == state2[4])
        return bool

    def _is_sorted(self):
        """Assert is the list is sorted or not.

        Args:

        Returns:
            True if the list is sorted, False otherwise

        """
        arr = self.scratchpad_ints
        return np.all(arr[:self.length-1] <= arr[1:self.length])

    def start_task(self, task_index):
        if self.tasks_list.count(task_index) > 0:
            task = self.get_program_from_index(task_index)
            if task == 'RESET' or task == 'BUBBLESORT':
                self._decr_length_right()
            if task == 'BUBBLE':
                self._decr_length_left()
        return super(RecursiveListEnv, self).start_task(task_index)

    def end_task(self):
        current_task = self.get_program_from_index(self.current_task_index)
        if current_task == 'RESET' or current_task == 'BUBBLESORT':
            if self.tasks_list.count(self.current_task_index) > 1:
                self._incr_length_right()
        if current_task == 'BUBBLE':
            if self.tasks_list.count(self.current_task_index) > 1:
                self._incr_length_left()
        super(RecursiveListEnv, self).end_task()
