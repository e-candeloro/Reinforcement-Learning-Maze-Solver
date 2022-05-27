import numpy as np
import pandas as pd
import random
import os
import time

"""
NOTE: The following code was adapted and reformatted from a previous exercise done for the course of Machine Learning and Deep Learning (2020-2021) at the University of Modena and Reggio Emilia.
The original project and code can be found at this link:
https://drive.google.com/drive/folders/1btN4CHqwsDtXdGXHTj7CMlHoKsS7ob2Z?usp=sharing

All credit for the snippets used goes to the original author(s)
"""

# path and name of the labyrinth csv file, if imported
LABYRINT_FILE = "labyrinth.csv"


class TDAgent:
    """
    Class of the Reinforcement Learning Agent using the TD (Temporal Difference) learning method.

    --------
    Methods:

    - get_action_eps_greedy: perform the action using a greedy policy with prob. 1 - epsilon and a random action with probability epsilon
    - get_action_greedy: perform the action using the learned greedy policy
    - get_action_symb: returns the action symbol (for printing purposes)
    - update_Q_function: performs the TD step to update the policy on the fly inside an episode

    """
    # DEFINE THE ACTION SYMBOLS
    ACTIONS = {"UP": 0, "LEFT": 1, "DOWN": 2, "RIGHT": 3}

    def __init__(self, alpha: int, gamma: int, epsilon: float, lab_matrix_shape: tuple):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = len(self.ACTIONS)
        self.rows, self.columns = lab_matrix_shape
        self.Q = np.random.rand(self.rows, self.columns, self.num_actions)

    def get_action_eps_greedy(self, state: tuple):
        """
        Performs an action in a greedy way with prob. 1 - epsilon or in a random way with prob. epsilon.
        This allows to explore and exploit the solution space and avoid a 100% greedy agent behaviour when learning (updating) the policy.

        Parameters
        ----------
        state: tuple
            state (position) of the RL agent in the form of the tuple (y_row, x_col)

        Returns
        -------
        next_action: int
            one of the possibile action between up, left, down, right -> [0,1,2,3]

        """

        y_cor, x_cor = state
        eps = random.random()

        if eps < self.epsilon:
            return random.randint(0, self.num_actions-1)
        else:
            return np.argmax(self.Q[y_cor, x_cor])

    def get_action_greedy(self, state: tuple):
        """
        Performs an action in a fully greedy way.
        This methods is used only after the policy is learned after the training, to exploit the best move possibile for the agent in a given state.

        Parameters
        ----------
        state: tuple
            state (position) of the RL agent in the form of the tuple (y_row, x_col)

        Returns
        -------
        next_action: int
            one of the possibile action between up, left, down, right -> [0,1,2,3]

        """
        y_cor, x_cor = state

        return np.argmax(self.Q[y_cor, x_cor])

    def get_action_symb(self, state):
        """
        Returns the action symbol for printing purposes (to help show the learned policy in the environment).

        Parameters
        ----------
        state: tuple
            state (position) of the RL agent in the form of the tuple (y_row, x_col)

        Returns
        -------
        action_symb: char
            one of the possibile action symbols between up, left, down, right -> [^,<,v,>]

        """

        action = self.get_action_greedy(state)

        if action == self.ACTIONS["UP"]:
            return "^"
        if action == self.ACTIONS["LEFT"]:
            return "<"
        if action == self.ACTIONS["DOWN"]:
            return "v"
        if action == self.ACTIONS["RIGHT"]:
            return ">"

    def update_Q_function(self, prev_state, new_state, reward, action):
        """
        Updates the Q function (policy) using the TD0 or Q-learning method.
        Given the current state (prev_state), the action performed, the new_state and the reward from the action, update on the fly the value for that combination of current state and action.

        The following formula is used:
        along with the parameters alpha (weight given to the adjustement step) and gamma (discounting factor) that gives more or less weight to the max reward obtainable in the next step.

        Parameters
        ----------
        prev_state: tuple
            previous state (position) of the RL agent in the form of the tuple (y_row, x_col)

        new_state: tuple
            new state (position) of the RL agent in the form of the tuple (y_row, x_col)

        reward: int
            reward of the agent computed when executing the action

        action: int
            action perfomed when transitioning from the prev_state to the new_state
        """

        self.Q[prev_state[0], prev_state[1], action] =\
            self.Q[prev_state[0], prev_state[1], action] +\
            self.alpha * (reward + self.gamma * self.Q[new_state[0], new_state[1]].max() -
                          self.Q[prev_state[0], prev_state[1], action])


class Environment:
    """
    Environment class that models the labirynth environment.

    --------
    Methods:

    - init_default_labyrinth: initializes the default labyrinth pattern using the class attributes
    - init_labyrinth_from_csv: initializes a labyrinth from a .csv file
    - __repr__, policy_str: methods to show the labyrinth and the learned policy of the agent in it
    - new_episode: resets the agent position to the starting point
    - perform action: given the action to perform, chosen by the agent, update the agent position (the agent state) and returns it togheter with the reward obtained and if the end goal was reached (is_over flag)
    - other support methods
    """

    LAB = 0
    WALL = 1
    START = 2
    FINISH = 3

    REWARD = -1
    WALL_REWARD = -30

    def __init__(self, import_maze_csv=False):
        self.from_csv = import_maze_csv  # flag for importing the maze from a csv
        self.lab_matrix = None
        self.start_pos = tuple()
        self.finish_pos = tuple()
        self.agent_pos = tuple()

        if self.from_csv:
            self.init_labyrinth_from_csv()
        else:
            self.init_default_labyrinth()

        # initialize the start, finish and agent initial positions
        self.start_pos = tuple(np.argwhere(self.lab_matrix == self.START)[0])
        self.finish_pos = tuple(np.argwhere(self.lab_matrix == self.FINISH)[0])
        self.agent_pos = self.start_pos

        # set the number of rows and columns for the maze
        self.rows, self.cols = self.lab_matrix.shape

    def init_default_labyrinth(self):
        """
        Initialized the default labyrinth (lab_matrix).
        """

        self.lab_matrix = np.array([
            [self.WALL, self.WALL, self.WALL, self.WALL,
                self.WALL, self.WALL, self.WALL, self.WALL],
            [self.WALL, self.LAB, self.LAB, self.LAB,
                self.LAB, self.LAB, self.LAB, self.WALL],
            [self.START, self.LAB, self.WALL, self.WALL,
                self.LAB, self.WALL, self.LAB, self.WALL],
            [self.WALL, self.LAB, self.LAB, self.WALL,
                self.WALL, self.LAB, self.LAB, self.WALL],
            [self.WALL, self.WALL, self.LAB, self.LAB,
                self.WALL, self.LAB, self.WALL, self.WALL],
            [self.WALL, self.LAB, self.WALL, self.LAB,
                self.WALL, self.LAB, self.LAB, self.WALL],
            [self.WALL, self.LAB, self.LAB, self.LAB,
                self.LAB, self.WALL, self.LAB, self.FINISH],
            [self.WALL, self.WALL, self.WALL, self.WALL,
                self.WALL, self.WALL, self.WALL, self.WALL]
        ])

        return

    def init_labyrinth_from_csv(self, path=LABYRINT_FILE):
        """
        Import the labirynth from a csv file.
        The file must have no headers,must use only the 4 allowed symbols,must have only one start and only one finish value.
        """
        df_lab = pd.read_csv(path, header=None)
        self.lab_matrix = df_lab.to_numpy()

        unique, counts = np.unique(self.lab_matrix, return_counts=True)
        assert len(
            unique) == 4, "Only 4 variables are allowed in the csv: \nLAB = 0; WALL = 1; START = 2;FINISH = 3;"

        dict_values = dict(zip(unique, counts))
        assert dict_values[self.START] == 1 and dict_values[self.FINISH] == 1, "Labyrint must have only one start and one finish!\n"

        return

    def __repr__(self):
        """
        Represent the labyrinth in ASCII-art

        Returns
        -------
        environment_description: str
            Text representation of the environment.
        """
        environment_description = ''

        for i in range(0, self.rows):
            for j in range(0, self.cols):
                if self.agent_pos == (i, j):
                    if self.lab_matrix[i, j] == self.FINISH:
                        environment_description += '🏁'
                    else:
                        environment_description += '🤖'
                elif self.lab_matrix[i, j] == self.LAB or self.lab_matrix[i, j] == self.START:
                    environment_description += '. '
                elif self.lab_matrix[i, j] == self.WALL:
                    environment_description += '█ '
                elif self.lab_matrix[i, j] == self.FINISH:
                    environment_description += '✔ '

            environment_description += '\n'
        return environment_description

    def policy_str(self, agent):
        """
        Return the string representation of policy learnt by the agent.

        Parameters
        ----------
        agent: Agent
            Reinforcement learning agent

        Returns
        -------
        policy_description: str
            Text representation of the policy learnt by the agent.
        """

        policy_description = ''

        for i in range(0, self.rows):
            for j in range(0, self.cols):
                if self.lab_matrix[i, j] == self.WALL:
                    policy_description += '█ '
                elif self.lab_matrix[i, j] == self.FINISH:
                    policy_description += '✔ '
                else:
                    action_symb = agent.get_action_symb(state=(i, j))
                    policy_description += action_symb + ' '
            policy_description += '\n'
        return policy_description

    def new_episode(self):
        """
        Starts a new episode, setting the position of the agent to the starting point.

        Returns
        -------
        agent_pos: tuple
            Tuple in the format (y_rows, x_cols) of the agent position in the starting point.
        """
        # resets the agent position in the environment
        self.agent_pos = self.start_pos
        return self.agent_pos

    def perform_action(self, action):
        """
        Perform an action in the Environment

        Parameters
        ----------
        action: Action
            Possible action number in the ACTIONS of the Agent:

            "UP": 0, "LEFT": 1, "DOWN": 2, "RIGHT": 3

        Returns
        -------
        current_state (agent position): tuple
            Current state of the Environment -> Agent position
        reward: int
            Reward for action performed computed with the get_reward method
        is_over: bool
            Flag to signal the end of the episode, computed with the is_over method
        """
        (y, x) = self.agent_pos

        if action == TDAgent.ACTIONS["UP"]:
            y -= 1
        elif action == TDAgent.ACTIONS["LEFT"]:
            x -= 1
        elif action == TDAgent.ACTIONS["DOWN"]:
            y += 1
        elif action == TDAgent.ACTIONS["RIGHT"]:
            x += 1

        # Check if we are inside the labyrinth
        if 0 < y < self.rows and 0 < x < self.cols:
            self.agent_pos = (y, x)

        return self.agent_pos, self.get_reward, self.is_over

    @property
    def is_over(self):
        """
        Flag to signal whether the agent reached the exit.

        Returns
        -------
        Bool:
            True if the agent is in the finish position, False otherwise.

        """
        return self.agent_pos == self.finish_pos

    @property
    def get_reward(self):
        """
        Computes the reward given the agent position

        Normal reward = -1
        Wall reward = -30 (penalty)

        Returns
        -------
        Reward: int
            The computed reward given the agent position (state)
        """
        y, x = self.agent_pos

        if self.lab_matrix[y, x] == self.WALL:
            return self.WALL_REWARD
        else:
            return self.REWARD

    def get_maze_shape(self):
        """
        Computes how many rows and columns the labyrinth matrix has.

        Returns
        -------
        labyrinth.shape: tuple
            Tuple in the format (n_rows, n_columns) of the labyrinth matrix
        """
        return self.lab_matrix.shape


def main(n_episodes=150, alpha=1, gamma=1, epsilon=0.01, import_maze_csv=False, show_training=False):
    """
    Execute the RL algorithm for an agent that must find and exit in a maze, given a starting position.

    Parameters
    ----------
    n_episodes: int (default is 150)
        number of episoded for training the RL algorithm

    alpha: int (default is 1)
        coefficients for the weight of the update of the policy

    gamma: int (default is 1)
        discounting factor for the TD (Q-learning) algorithm that gives more or less importance to the future state reward.

    espilon: float (default is 0.01)
        coefficient for setting the probability of choosing a random action while training the agent.
        Default is 0.01 and means a 10% prob. of selecting a random action.

    import_maze_csv: Bool (default is False)
        if set to True, tries to import a maze from a csv file, whose path is defined in the LABYRINT_FILE constant at the start of the script

    show_training: Bool (default is False)
        if set to True, shows the agent moves in the training phase during all the epochs.
        If n_episodes is a big number and/or the maze space is big, it may require some time.
    """
    # objects instantiation
    environment = Environment(import_maze_csv=import_maze_csv)

    maze_shape = environment.get_maze_shape()
    agent = TDAgent(alpha=alpha, gamma=gamma, epsilon=epsilon,
                    lab_matrix_shape=maze_shape)

    # shows the maze to solve and the agent starting position
    print("Maze to solve and Agent start position:\n\n")
    print(environment)

    # RL 1-Step TD LOOP
    for e in range(n_episodes):

        # cumulative reward counter, to evaluate performance of the RL alg.
        tot_reward = 0
        steps = 0  # episode steps counter
        state = environment.new_episode()  # reset agent state (position)
        is_over = False  # termination flag

        while True:
            if show_training:
                # clear screen
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"Step: {steps}")
                print(environment)

            # use epsilon greedy policy to explore/exploit and select an action
            action = agent.get_action_eps_greedy(state=state)

            # perform the selected action on the environment and compute the new_state (position),
            # the reward obtained doing the move, and if the destination is reached (is_over)
            new_state, reward, is_over = environment.perform_action(
                action=action)

            # given the previous state, the new state, the action and the respective reward
            # update the Q function (policy) using the TD0 (Q-learing) formula
            agent.update_Q_function(
                prev_state=state, new_state=new_state, reward=reward, action=action)

            tot_reward += reward  # update the cumulative reward
            state = new_state  # update the state (state transition)
            steps += 1  # update number of step for the single episode

            if is_over:  # if the agent reaches the end, terminate the episode
                break

        print(f'Episode: {e} - Tot Steps: {steps} Tot Reward: {tot_reward}')
        # shows the total reward obtained in every episode
        # if the algorithm works, we expect a maximization of the reward over each episode

    # show the learned policy as preferred actions in the labyrinth
    print('\n\nLearned Policy:\n')
    print(environment.policy_str(agent))

    # test the learned policy using only the greedy approach
    input('\n\nPress `ENTER` to start testing the learned policy.')

    # reset the agent position to the starting point
    state = environment.new_episode()
    is_over = False  # termination flag
    counter = 0  # counter for computing the step numbers
    while True:

        os.system('cls' if os.name == 'nt' else 'clear')  # clear screen
        # show the agent steps when reaching the end of the maze
        print(f"step {counter}\n")
        print(environment)

        # Choose agent's action (greedy policy)
        action = agent.get_action_greedy(state=state)

        # Perform action in the environment
        new_state, reward, is_over = environment.perform_action(action=action)

        # Update state and actio counter
        state = new_state
        counter += 1

        if is_over:  # if the agent reaches the end, end the loop
            os.system('cls' if os.name == 'nt' else 'clear')  # clear screen
            print(f"step {counter}\n")
            print(environment)
            print("\nFinish!")
            return

        time.sleep(0.3)


if __name__ == '__main__':
    main(n_episodes=150, alpha=1, gamma=1,
         epsilon=0.01, import_maze_csv=False, show_training=False)
