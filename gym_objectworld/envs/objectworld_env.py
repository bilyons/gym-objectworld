from .adapted_minigrid import *
from itertools import product
np.random.seed(0)
class ObjectWorldEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, grid_size, n_objects, n_colours, p_slip, discrete = False, agent_start_pos=None,):

        self.agent_start_pos = agent_start_pos
        self.n_objects = n_objects
        self.n_colours = n_colours
        self.discrete = discrete

        super().__init__(grid_size=grid_size, p_slip = p_slip, max_steps=8,)

        if self.discrete == False:
            self.observation_space = spaces.Box(
                low = 0,
                high = np.sqrt((2*(self.grid_size-2)**2)),
                shape=(2*self.n_colours, ),
                dtype='float64'
            )

    def _gen_grid(self, grid_size):
        # Create an empty grid
        self.grid = Grid(grid_size)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, grid_size, grid_size)

        # Place a goal square in the bottom-right corner
        self.objects = {}
        for _ in range(self.n_objects):
            colour1 = np.random.randint(0, self.n_colours)
            colour2 = np.random.randint(0, self.n_colours)

            while True:
                y, x = self._rand_int(1, grid_size-1), self._rand_int(1, grid_size-1)
                if (y,x) not in self.objects:
                    break

            self.put_obj(OWObject(colour1, colour2), y, x)

            self.objects[y,x] = (OWObject(colour1, colour2), (y, x))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def _reward(self, state):
        """
        In ObjectWorld, the agent receives a reward depending on its proximity to
        the colours.

        +1 reward if:
            within distance 3 of colour 0 (red) and colour 1 (green)
        -1 reward if:
            within distance 3 of colour 0 (red) only
        0 reward if:
            otherwise
        """
        y, x = state
        near_c0 = False
        near_c1 = False

        for (dx, dy) in product(range(-3,4), range(-3,4)):
            if 1<= x+dx < self.grid_size-1 and 1<= y+dy < self.grid_size-1:
                if (abs(dx) + abs(dy)) <= 3 and (x+dx, y+dy) in self.objects and self.objects[(x+dx, y+dy)][0].colour1 == 'red':
                    near_c0 = True
                if (abs(dx) + abs(dy)) <= 3 and (x+dx, y+dy) in self.objects and self.objects[(x+dx, y+dy)][0].colour2 == 'green':
                    near_c1 = True
        if near_c0 and near_c1:
            return 1
        if near_c0:
            return -1
        return 0

    def _gen_obs(self, state=None):
        """
        Feature vector of the state continuous and discrete
        """
        if state is None:
            sy, sx = self.agent_pos
        else:
            sy, sx = state

        nearest_inner = {}
        nearest_outer = {}

        for x in range(1, self.grid_size-1):
            for y in range(1, self.grid_size-1):
                if (y,x) in self.objects:
                    dist = math.hypot((x-sx), (y-sy))
                    obj = self.objects[(y,x)]
                    if COLOUR_TO_INDEX[obj[0].colour1] in nearest_inner:
                        if dist < nearest_inner[COLOUR_TO_INDEX[obj[0].colour1]]:
                            nearest_inner[COLOUR_TO_INDEX[obj[0].colour1]] = dist
                    else:
                        nearest_inner[COLOUR_TO_INDEX[obj[0].colour1]] = dist
                    if COLOUR_TO_INDEX[obj[0].colour2] in nearest_outer:
                        if dist < nearest_outer[COLOUR_TO_INDEX[obj[0].colour2]]:
                            nearest_outer[COLOUR_TO_INDEX[obj[0].colour2]] = dist
                    else:
                        nearest_outer[COLOUR_TO_INDEX[obj[0].colour2]] = dist

        # Ensure all colors are represented
        for c in range(self.n_colours):
            if c not in nearest_inner:
                nearest_inner[c] = 0
            if c not in nearest_outer:
                nearest_outer[c] = 0

        if self.discrete:
            state = np.zeros((2*self.n_colours*self.grid_size-2,))
            i=0
            for c in range(self.n_colours):
                for d in range(1, self.grid_size-1):
                    if nearest_inner[c] < d:
                        state[i] = 1
                    i+=1
                    if nearest_outer[c] < d:
                        state[i] = 1
                    i+=1
            assert i==2*self.n_colours*(self.grid_size-2)
            assert (state >= 0).all()
        else:
            # Continuous features
            state = np.zeros((2*self.n_colours))
            i=0
            for c in range(self.n_colours):
                state[i] = nearest_inner[c]
                i+=1
                state[i] = nearest_outer[c]
                i+=1
        return state

    def _feature_matrix(self, discrete=True):

        return np.array([self._gen_obs((y,x), discrete) for 
            (y,x) in product(range(1, self.grid_size-1), range(1, self.grid_size-1))])

    def reset(self):
        # Current position and direction of the agent
        self.agent_pos = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.grid_size)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Generate observation
        obs = self._gen_obs(self.agent_pos)

        return obs