from gym_objectworld.adapted_minigrid import *
from itertools import product

class ObjectWorld(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        grid_size,
        n_objects,
        n_colours,
        p_slip,
        agent_start_pos=None,
    ):
        self.agent_start_pos = agent_start_pos
        self.n_objects = n_objects
        self.n_colours = n_colours

        super().__init__(
            grid_size=grid_size,
            p_slip = p_slip,
            max_steps=4*grid_size*grid_size,
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

    def _gen_obs(self, discrete=True):
        """
        Feature vector of the state continuous and discrete
        """

        sx, sy = self.agent_pos

        nearest_inner = {}
        nearest_outer = {}

        for y in range(1, self.grid_size-1):
            for x in range(1, self.grid_size-1):
                if (x,y) in self.objects:
                    dist = math.hypot((x-sx), (y-sy))
                    obj = self.objects[(x,y)]
                    if obj[0].colour1 in nearest_inner:
                        if dist < nearest_inner[obj[0].colour1]:
                            nearest_inner[obj[0].colour1] = dist
                    else:
                        nearest_inner[obj[0].colour1] = dist
                    if obj[0].colour2 in nearest_outer:
                        if dist < nearest_outer[obj[0].colour2]:
                            nearest_outer[obj[0].colour2] = dist
                    else:
                        nearest_outer[obj[0].colour2] = dist
        # Ensure all colors are represented
        for c in range(self.n_colours):
            if c not in nearest_inner:
                nearest_inner[c] = 0
            if c not in nearest_outer:
                nearest_outer[c] = 0

        if discrete:
            state = np.zeros((2*self.n_colours*self.grid_size,))
            i=0
            for c in range(self.n_colours):
                for d in range(1, self.grid_size+1):
                    if nearest_inner[c] < d:
                        state[i] = 1
                    i+=1
                    if nearest_outer[c] < d:
                        state[i] = 1
                    i+=1
            assert i==2*self.n_colours*self.grid_size
            assert (state >= 0).all()
        else:
            pass

        return state


test = ObjectWorld(32, 50, 4, 0.3)
test.step(0)

for i in test.objects:
    x = test.objects[i][1]