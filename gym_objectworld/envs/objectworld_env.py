import math
import hashlib
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from gym_objectworld import rendering as r

# Pixel size
TILE_PIXELS = 32

# Map of colour names to RGB
COLOURS = {
	'red'   : np.array([255, 0, 0]),
	'green' : np.array([0, 255, 0]),
	'blue'  : np.array([0, 0, 255]),
	'yellow': np.array([255, 255, 0]),
	'grey'  : np.array([100, 100, 100]),
	'black' : np.array([0,0,0]),
	'white' : np.array([255,255,255])
}

COLOUR_TO_INDEX = {
	'red'   : 0,
	'green' : 1,
	'blue'  : 2,
	'yellow': 3,
	'grey'  : 4,
	'black' : 5,
	'white' : 6
}

INDEX_TO_COLOUR = dict(zip(COLOUR_TO_INDEX.values(), COLOUR_TO_INDEX.keys()))

# Map of object type to integers
OBJECT_TO_INDEX = {
	'wall'  : 0,
	'floor' : 1,
	'object': 2,
	'agent' : 3,
	'empty' : 4
}

INDEX_TO_OBJECT = dict(zip(OBJECT_TO_INDEX.values(), OBJECT_TO_INDEX.keys()))

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class WorldObject:
	"""
	Base class for objects
	"""

	def __init__(self, type, colour1, colour2):
		assert type in OBJECT_TO_INDEX, type
		assert colour1 in COLOUR_TO_INDEX, colour1
		assert colour2 in COLOUR_TO_INDEX, colour2
		self.type = type
		self.colour1 = colour1
		self.colour2 = colour2

		# Initial position of object
		self.init_pos = None

		# Current position of object
		self.cur_pos = None

	def encode(self):
		"""
		Encode the object as a tuple
		"""
		return (OBJECT_TO_INDEX[self.type], COLOUR_TO_INDEX[self.colour1], 
										COLOUR_TO_INDEX[self.colour2], 0)

	def decode(type_idx, colour1, colour2, state):
		"""
		Create the object from the tuple
		"""

		obj_type = INDEX_TO_OBJECT[type_idx]
		colour1 = INDEX_TO_COLOUR[colour1]
		colour2 = INDEX_TO_COLOUR[colour2]

		is_open = state == 0


		if obj_type == 'wall':
			v = Wall(colour1)
		elif obj_type == 'floor':
			v = Floor(colour1)
		elif obj_type == 'object':
			v = OWObject(colour1, colour2)
		else:
			assert False, "unknown object"
		return v

class Floor(WorldObject):
	"""
	Walkable floor tile
	"""

	def __init__(self, colour='black'):
		super().__init__('floor', colour, colour)

	def can_overlap(self):
		return True

	def render(self, img):
		colour = COLOURS[self.colour1]
		r.fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), colour)

class Wall(WorldObject):
	"""
	Impassable wall tile
	"""

	def __init__(self, colour='grey'):
		super().__init__('floor', colour, colour)

	def can_overlap(self):
		return False

	def render(self, img):
		colour = COLOURS[self.colour1]
		r.fill_coords(img, point_in_rect(0, 1, 0, 1), colour)

class OWObject(WorldObject):
	"""
	This is an object in the object world. Each object consists
	of an inner colour and an outer colour, the number of which
	is defined in the object world environment.
	"""

	def __init__(self, colour1, colour2):
		super().__init__('object', colour1, colour2)

	def can_overlap(self):
		return True

	def render(self, img):
		colour1 = COLOURS[self.colour1]
		colour1 = COLOURS[self.colour2]
		r.fill_coords(img, point_in_rect(0, 1, 0, 1), colour1)
		r.fill_coords(img, point_in_rect(0.1, 1, 0.1, 1), colour2)

class Grid:
	"""
	Represent a grid and operations on it
	"""

	# Static cache of pre-renderer tiles
	tile_cache = {}

	def __init__(self, grid_size):
		self.width = grid_size
		self.height = grid_size

		self.grid = [None] * grid_size * grid_size

	def __contains__(self, key):
		if isinstance(key, WorldObj):
			for e in self.grid:
				if e is key:
					return True
		elif isinstance(key, tuple):
			for e in self.grid:
				if e is None:
					continue
				if (e.color, e.type) == key:
					return True
				if key[0] is None and key[1] == e.type:
					return True
		return False

	def __eq__(self, other):
		grid1  = self.encode()
		grid2 = other.encode()
		return np.array_equal(grid2, grid1)

	def __ne__(self, other):
		return not self == other

	def copy(self):
		from copy import deepcopy
		return deepcopy(self)

	def set(self, i, j, v):
		assert i >= 0 and i < self.width
		assert j >= 0 and j < self.height
		self.grid[j * self.width + i] = v

	def get(self, i, j):
		assert i >= 0 and i < self.width
		assert j >= 0 and j < self.height
		return self.grid[j * self.width + i]

	def horz_wall(self, x, y, length=None, obj_type=Wall):
		if length is None:
			length = self.width - x
		for i in range(0, length):
			self.set(x + i, y, obj_type())

	def vert_wall(self, x, y, length=None, obj_type=Wall):
		if length is None:
			length = self.height - y
		for j in range(0, length):
			self.set(x, y + j, obj_type())

	def wall_rect(self, x, y, w, h):
		self.horz_wall(x, y, w)
		self.horz_wall(x, y+h-1, w)
		self.vert_wall(x, y, h)
		self.vert_wall(x+w-1, y, h)

	def rotate_left(self):
		"""
		Rotate the grid to the left (counter-clockwise)
		"""

		grid = Grid(self.height, self.width)

		for i in range(self.width):
			for j in range(self.height):
				v = self.get(i, j)
				grid.set(j, grid.height - 1 - i, v)

		return grid

	def slice(self, topX, topY, width, height):
		"""
		Get a subset of the grid
		"""

		grid = Grid(width, height)

		for j in range(0, height):
			for i in range(0, width):
				x = topX + i
				y = topY + j

				if x >= 0 and x < self.width and \
				   y >= 0 and y < self.height:
					v = self.get(x, y)
				else:
					v = Wall()

				grid.set(i, j, v)

		return grid

	@classmethod
	def render_tile(
		cls,
		obj,
		agent_pos,
		tile_size=TILE_PIXELS,
		subdivs=3
	):

		"""
		Render a tile and cache the result
		"""
		# Hash map lookup key for the cache
		print("cls", cls)
		print("obj", obj)
		print(agent_pos)
		print("tile_size", tile_size)

		key = (tile_size)
		print(key)
		key = obj.encode() + key if obj else key

		if key in cls.tile_cache:
			return cls.tile_cache[key]
		print(tile_size)
		exit()
		img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

		# Draw the grid lines (top and left edges)
		r.fill_coords(img, r.point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
		r.fill_coords(img, r.point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

		if obj != None:
			obj.render(img)

		# Overlay agent on top
		if agent_dir is not None:
			r.fill_coords(img, point_in_circle(0.5,0.5,0.31, COLOURS[self.colour1]))

		# Highlight the cell if needed
		if highlight:
			r.highlight_img(img)

		# Downsample the image to perform supersampling/anti-aliasing
		img = r.downsample(img, subdivs)

		# Cache the rendered tile
		cls.tile_cache[key] = img

		return img

	def render(
		self,
		tile_size,
		agent_pos=None
	):
		"""
		Render this grid at a given scale
		:param r: target renderer object
		:param tile_size: tile size in pixels
		"""
		# Compute the total grid size
		width_px = self.width * tile_size
		height_px = self.height * tile_size

		img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

		# Render the grid
		for j in range(0, self.height):
			for i in range(0, self.width):
				cell = self.get(i, j)
				print(cell)
				print(tile_size)
				agent_here = np.array_equal(agent_pos, (i, j))
				print(agent_here)
				tile_img = Grid.render_tile(
					cell,
					0 if agent_here else None,
					agent_pos,
					tile_size
					
				)

				ymin = j * tile_size
				ymax = (j+1) * tile_size
				xmin = i * tile_size
				xmax = (i+1) * tile_size
				img[ymin:ymax, xmin:xmax, :] = tile_img

		return img

	def encode(self, vis_mask=None):
		"""
		Produce a compact numpy encoding of the grid
		"""

		if vis_mask is None:
			vis_mask = np.ones((self.width, self.height), dtype=bool)

		array = np.zeros((self.width, self.height, 4), dtype='uint8')

		for i in range(self.width):
			for j in range(self.height):
				if vis_mask[i, j]:
					v = self.get(i, j)

					if v is None:
						array[i, j, 0] = OBJECT_TO_INDEX['empty']
						array[i, j, 1] = 0
						array[i, j, 2] = 0

					else:
						array[i, j, :] = v.encode()

		return array

	@staticmethod
	def decode(array):
		"""
		Decode an array grid encoding back into a grid
		"""

		width, height, channels = array.shape
		assert channels == 3

		grid = Grid(width, height)
		for i in range(width):
			for j in range(height):
				type_idx, color_idx, state = array[i, j]
				v = WorldObj.decode(type_idx, color_idx, state)
				grid.set(i, j, v)

		return grid

class MiniGridEnv(gym.Env):
	"""
	2D grid world game environment
	"""

	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 10
	}

	# Enumeration of possible actions
	class Actions(IntEnum):
		# Turn left, turn right, move forward
		left = 0
		right = 1
		forward = 2

		# Done completing task
		done = 6

	def __init__(
		self,
		grid_size=None,
		max_steps=100,
		see_through_walls=False,
		seed=1337,
		agent_view_size=7
	):

		# Action enumeration for this environment
		self.actions = MiniGridEnv.Actions

		# Actions are discrete integer values
		self.action_space = spaces.Discrete(len(self.actions))

		# Number of cells (width and height) in the agent view
		assert agent_view_size % 2 == 1
		assert agent_view_size >= 3
		self.agent_view_size = agent_view_size

		# Observations are dictionaries containing an
		# encoding of the grid and a textual 'mission' string
		self.observation_space = spaces.Box(
			low=0,
			high=255,
			shape=(self.agent_view_size, self.agent_view_size, 3),
			dtype='uint8'
		)
		self.observation_space = spaces.Dict({
			'image': self.observation_space
		})

		# Range of possible rewards
		self.reward_range = (0, 1)

		# Window to use for human rendering mode
		self.window = None

		# Environment configuration
		self.grid_size = grid_size
		self.width = grid_size
		self.height = grid_size
		self.max_steps = max_steps
		self.see_through_walls = see_through_walls

		# Current position and direction of the agent
		self.agent_pos = None

		# Initialize the RNG
		self.seed(seed=seed)

		# Initialize the state
		self.reset()

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


	def seed(self, seed=1337):
		# Seed the random number generator
		self.np_random, _ = seeding.np_random(seed)
		return [seed]

	def hash(self, size=16):
		"""Compute a hash that uniquely identifies the current state of the environment.
		:param size: Size of the hashing
		"""
		sample_hash = hashlib.sha256()

		to_encode = [self.grid.encode().tolist(), self.agent_pos, self.agent_dir]
		for item in to_encode:
			sample_hash.update(str(item).encode('utf8'))

		return sample_hash.hexdigest()[:size]

	@property
	def steps_remaining(self):
		return self.max_steps - self.step_count

	def __str__(self):
		"""
		Produce a pretty string of the environment's grid along with the agent.
		A grid cell is represented by 2-character string, the first one for
		the object and the second one for the color.
		"""

		# Map of object types to short string
		OBJECT_TO_STR = {
			'wall'          : 'W',
			'floor'         : 'F',
			'door'          : 'D',
			'key'           : 'K',
			'ball'          : 'A',
			'box'           : 'B',
			'goal'          : 'G',
			'lava'          : 'V',
		}

		# Short string for opened door
		OPENDED_DOOR_IDS = '_'

		# Map agent's direction to short string
		AGENT_DIR_TO_STR = {
			0: '>',
			1: 'V',
			2: '<',
			3: '^'
		}

		str = ''

		for j in range(self.grid.height):

			for i in range(self.grid.width):
				if i == self.agent_pos[0] and j == self.agent_pos[1]:
					str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
					continue

				c = self.grid.get(i, j)

				if c == None:
					str += '  '
					continue

				if c.type == 'door':
					if c.is_open:
						str += '__'
					elif c.is_locked:
						str += 'L' + c.color[0].upper()
					else:
						str += 'D' + c.color[0].upper()
					continue

				str += OBJECT_TO_STR[c.type] + c.color[0].upper()

			if j < self.grid.height - 1:
				str += '\n'

		return str

	def _gen_grid(self, grid_size):
		assert False, "_gen_grid needs to be implemented by each environment"

	def _reward(self):
		"""
		Compute the reward to be given upon success
		"""

		return 1 - 0.9 * (self.step_count / self.max_steps)

	def _rand_int(self, low, high):
		"""
		Generate random integer in [low,high[
		"""

		return self.np_random.randint(low, high)

	def _rand_float(self, low, high):
		"""
		Generate random float in [low,high[
		"""

		return self.np_random.uniform(low, high)

	def _rand_bool(self):
		"""
		Generate random boolean value
		"""

		return (self.np_random.randint(0, 2) == 0)

	def _rand_elem(self, iterable):
		"""
		Pick a random element in a list
		"""

		lst = list(iterable)
		idx = self._rand_int(0, len(lst))
		return lst[idx]

	def _rand_subset(self, iterable, num_elems):
		"""
		Sample a random subset of distinct elements of a list
		"""

		lst = list(iterable)
		assert num_elems <= len(lst)

		out = []

		while len(out) < num_elems:
			elem = self._rand_elem(lst)
			lst.remove(elem)
			out.append(elem)

		return out

	def _rand_color(self):
		"""
		Generate a random color name (string)
		"""

		return self._rand_elem(COLOR_NAMES)

	def _rand_pos(self, xLow, xHigh, yLow, yHigh):
		"""
		Generate a random (x,y) position tuple
		"""

		return (
			self.np_random.randint(xLow, xHigh),
			self.np_random.randint(yLow, yHigh)
		)

	def place_obj(self,
		obj,
		top=None,
		size=None,
		reject_fn=None,
		max_tries=math.inf
	):
		"""
		Place an object at an empty position in the grid
		:param top: top-left position of the rectangle where to place
		:param size: size of the rectangle where to place
		:param reject_fn: function to filter out potential positions
		"""

		if top is None:
			top = (0, 0)
		else:
			top = (max(top[0], 0), max(top[1], 0))

		if size is None:
			size = (self.grid.width, self.grid.height)

		num_tries = 0

		while True:
			# This is to handle with rare cases where rejection sampling
			# gets stuck in an infinite loop
			if num_tries > max_tries:
				raise RecursionError('rejection sampling failed in place_obj')

			num_tries += 1

			pos = np.array((
				self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
				self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
			))

			# Don't place the object on top of another object
			if self.grid.get(*pos) != None:
				continue

			# Don't place the object where the agent is
			if np.array_equal(pos, self.agent_pos):
				continue

			# Check if there is a filtering criterion
			if reject_fn and reject_fn(self, pos):
				continue

			break

		self.grid.set(*pos, obj)

		if obj is not None:
			obj.init_pos = pos
			obj.cur_pos = pos

		return pos

	def put_obj(self, obj, i, j):
		"""
		Put an object at a specific position in the grid
		"""

		self.grid.set(i, j, obj)
		obj.init_pos = (i, j)
		obj.cur_pos = (i, j)

	def place_agent(
		self,
		top=None,
		size=None,
		max_tries=math.inf
	):
		"""
		Set the agent's starting point at an empty position in the grid
		"""

		self.agent_pos = None
		pos = self.place_obj(None, top, size, max_tries=max_tries)
		self.agent_pos = pos

		return pos

	def get_view_coords(self, i, j):
		"""
		Translate and rotate absolute grid coordinates (i, j) into the
		agent's partially observable view (sub-grid). Note that the resulting
		coordinates may be negative or outside of the agent's view size.
		"""

		ax, ay = self.agent_pos
		dx, dy = self.dir_vec
		rx, ry = self.right_vec

		# Compute the absolute coordinates of the top-left view corner
		sz = self.agent_view_size
		hs = self.agent_view_size // 2
		tx = ax + (dx * (sz-1)) - (rx * hs)
		ty = ay + (dy * (sz-1)) - (ry * hs)

		lx = i - tx
		ly = j - ty

		# Project the coordinates of the object relative to the top-left
		# corner onto the agent's own coordinate system
		vx = (rx*lx + ry*ly)
		vy = -(dx*lx + dy*ly)

		return vx, vy

	def relative_coords(self, x, y):
		"""
		Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
		"""

		vx, vy = self.get_view_coords(x, y)

		if vx < 0 or vy < 0 or vx >= self.agent_view_size or vy >= self.agent_view_size:
			return None

		return vx, vy

	def in_view(self, x, y):
		"""
		check if a grid position is visible to the agent
		"""

		return self.relative_coords(x, y) is not None

	def agent_sees(self, x, y):
		"""
		Check if a non-empty grid position is visible to the agent
		"""

		coordinates = self.relative_coords(x, y)
		if coordinates is None:
			return False
		vx, vy = coordinates

		obs = self.gen_obs()
		obs_grid, _ = Grid.decode(obs['image'])
		obs_cell = obs_grid.get(vx, vy)
		world_cell = self.grid.get(x, y)

		return obs_cell is not None and obs_cell.type == world_cell.type

	def step(self, action):
		self.step_count += 1

		reward = 0
		done = False

		# Get the position in front of the agent
		fwd_pos = self.front_pos

		# Get the contents of the cell in front of the agent
		fwd_cell = self.grid.get(*fwd_pos)

		# Rotate left
		if action == self.actions.left:
			self.agent_dir -= 1
			if self.agent_dir < 0:
				self.agent_dir += 4

		# Rotate right
		elif action == self.actions.right:
			self.agent_dir = (self.agent_dir + 1) % 4

		# Move forward
		elif action == self.actions.forward:
			if fwd_cell == None or fwd_cell.can_overlap():
				self.agent_pos = fwd_pos
			if fwd_cell != None and fwd_cell.type == 'goal':
				done = True
				reward = self._reward()
			if fwd_cell != None and fwd_cell.type == 'lava':
				done = True

		# Pick up an object
		elif action == self.actions.pickup:
			if fwd_cell and fwd_cell.can_pickup():
				if self.carrying is None:
					self.carrying = fwd_cell
					self.carrying.cur_pos = np.array([-1, -1])
					self.grid.set(*fwd_pos, None)

		# Drop an object
		elif action == self.actions.drop:
			if not fwd_cell and self.carrying:
				self.grid.set(*fwd_pos, self.carrying)
				self.carrying.cur_pos = fwd_pos
				self.carrying = None

		# Toggle/activate an object
		elif action == self.actions.toggle:
			if fwd_cell:
				fwd_cell.toggle(self, fwd_pos)

		# Done action (not used by default)
		elif action == self.actions.done:
			pass

		else:
			assert False, "unknown action"

		if self.step_count >= self.max_steps:
			done = True

		obs = self.gen_obs()

		return obs, reward, done, {}

	def render(self, mode='human', close=False, tile_size=TILE_PIXELS):
		"""
		Render the whole-grid human view
		"""

		if close:
			if self.window:
				self.window.close()
			return

		if mode == 'human' and not self.window:
			import gym_objectworld.window
			self.window = gym_objectworld.window.Window('gym_objectworld')
			self.window.show(block=False)

		# Render the whole grid
		print(self.agent_pos)
		img = self.grid.render(
			tile_size,
			self.agent_pos
		)

		if mode == 'human':
			self.window.set_caption(self.mission)
			self.window.show_img(img)

		return img

	def close(self):
		if self.window:
			self.window.close()
		return

class EmptyEnv(MiniGridEnv):
	"""
	Empty grid environment, no obstacles, sparse reward
	"""

	def __init__(
		self,
		size=8,
		agent_start_pos=(1,1)
	):
		self.agent_start_pos = agent_start_pos

		super().__init__(
			grid_size=size,
			max_steps=4*size*size,
			# Set this to True for maximum speed
			see_through_walls=True
		)

	def _gen_grid(self, grid_size):
		# Create an empty grid
		self.grid = Grid(grid_size)

		# Generate the surrounding walls
		self.grid.wall_rect(0, 0, self.width, self.height)

		# Place a goal square in the bottom-right corner
		self.put_obj(OWObject(INDEX_TO_COLOUR[0], INDEX_TO_COLOUR[1]), self.width - 2, self.height - 2)

		# Place the agent
		if self.agent_start_pos is not None:
			self.agent_pos = self.agent_start_pos
		else:
			self.place_agent()

		self.mission = "get to the green goal square"

test = EmptyEnv()

test.render()