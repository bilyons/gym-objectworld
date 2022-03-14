import math
import hashlib
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from gym_objectworld import rendering as r
from itertools import product

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

	def __str__(self):
		return "<{} (In: {}) (Out: {})>".format(self.type, self.colour1, self.colour2)

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
		r.fill_coords(img, r.point_in_rect(0.031, 1, 0.031, 1), colour)

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
		r.fill_coords(img, r.point_in_rect(0, 1, 0, 1), colour)

class OWObject(WorldObject):
	"""
	This is an object in the object world. Each object consists
	of an inner colour and an outer colour, the number of which
	is defined in the object world environment.
	"""

	def __init__(self, colour1, colour2):
		super().__init__('object', INDEX_TO_COLOUR[colour1], INDEX_TO_COLOUR[colour2])

	def can_overlap(self):
		return True

	def render(self, img):
		colour1 = COLOURS[self.colour1]
		colour2 = COLOURS[self.colour2]
		r.fill_coords(img, r.point_in_rect(0, 1, 0, 1), colour2)
		r.fill_coords(img, r.point_in_rect(0.2, 0.8, 0.2, 0.8), colour1)

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
		agent_here,
		tile_size=TILE_PIXELS,
		subdivs=3
	):

		"""
		Render a tile and cache the result
		"""
		# Hash map lookup key for the cache
		key=(agent_here, tile_size)
		key = obj.encode() + key if obj else key

		if key in cls.tile_cache:
			return cls.tile_cache[key]

		img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

		# Draw the grid lines (top and left edges)
		r.fill_coords(img, r.point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
		r.fill_coords(img, r.point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

		if obj != None:
			obj.render(img)

		# Overlay agent on top
		if agent_here:
			r.fill_coords(img, r.point_in_circle(0.5,0.5,0.31), (255,255,255))


		# Downsample the image to perform supersampling/anti-aliasing
		img = r.downsample(img, subdivs)

		# Cache the rendered tile
		cls.tile_cache[key] = img

		return img

	def render(self, tile_size,	agent_pos=None):
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
				agent_here = np.array_equal(agent_pos, (i, j))
				tile_img = Grid.render_tile(
					cell,
					agent_here=agent_here if agent_here else None,
					tile_size=tile_size
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
		back = 3
		stay = 4

	def __init__(self, grid_size=None, p_slip = 0.0, max_steps=100):

		# Action enumeration for this environment
		self.actions = MiniGridEnv.Actions

		# Actions are discrete integer values
		self.action_space = spaces.Discrete(len(self.actions))

		# Range of possible rewards
		self.reward_range = (-1, 1)

		# Window to use for human rendering mode
		self.window = None

		# Environment configuration
		self.grid_size = grid_size
		self.width = grid_size
		self.height = grid_size
		self.max_steps = max_steps
		self.p_slip = p_slip
		self.corners = [np.array((1,1)), np.array((1, self.grid_size-2)), np.array((self.grid_size-2, 1)), np.array((self.grid_size-2, self.grid_size-2))]
		self.off_grid = np.array((0, self.grid_size-1))
		self.edges = np.array((1, self.grid_size-2))
		
		# Transition probability
		self.P = {(y_i, x_i): {a: {(y_k, x_k): [] for (y_k, x_k) in product(range(1, self.grid_size-1), range(1, self.grid_size-1))}
				for a in range(len(self.actions))}
				for (y_i, x_i) in product(range(1, self.grid_size-1), range(1, self.grid_size-1))
			}

		# Update the probability matrix
		for (y_i, x_i) in product(range(1, self.grid_size-1), range(1, self.grid_size-1)):
			for a in range(len(self.actions)):
				for (y_k, x_k) in product(range(1, self.grid_size-1), range(1, self.grid_size-1)):
					self.P[(y_i, x_i)][a][(y_k, x_k)] = self._transition_probability((y_i, x_i), a, (y_k, x_k))

		# Current position and direction of the agent
		self.agent_pos = None

		# Initialize the state
		self.reset()


	def _transition_probability(self, i, j, k):
		"""
		Get the transition probability from state i to state k given action j
		
		i: (y,x) coordinate pair
		j: 0-4 integer action value
		k: (y,x) coordinate pair
		"""
		# Initial state
		i = np.array(i)

		# Action array
		neighbours = self.get_neighbours(i)

		# Desired state
		k = np.array(k)

		neighbour = False

		# If not a neighbour
		if not np.any(np.all(k == neighbours, axis=1)):
			return 0.0

		# Is it the intended move
		if (neighbours[j] == k).all():
			# Was the move to stay where you are?		
			if (neighbours[j] == i).all():
				# Are you in a corner
				if np.any(np.all(i == self.corners, axis=1)):
					return 1-self.p_slip + 3*self.p_slip/len(self.actions)
				# Are you at an edge
				if np.in1d(neighbours[j], self.edges).any():
					return 1-self.p_slip + 2*self.p_slip/len(self.actions)
				# Are you free roaming
				return 1-self.p_slip + self.p_slip/len(self.actions)
			return 1 - self.p_slip + self.p_slip/len(self.actions)

		# If we are in a corner or wall
		# If these are still not the same point, we can only attend them
		# by moving off the grid
		# Corners
		if np.any(np.all(i == self.corners, axis=1)):
			# Corner
			# Can move off in two directions
			# Did we intend to move off the grid?
			if (0 in neighbours[j] or self.grid_size-1 in neighbours[j]) and (k == i).all():
				# 3 ways to stay slip into either corner or stay
				return 1-self.p_slip + 3*self.p_slip/len(self.actions)

			elif (i == k).all():
				# We didn't mean to but we could blow off in two directions or stay
				return 3*self.p_slip/len(self.actions)

		elif np.in1d(i, self.edges).any():
			# Not a corner, is it an edge?
			if not (i != self.edges).any():
				# Not an edge
				return 0.0

			# Edge
			# Can only move off the edge in one direction
			# Did we intend to move off the grid
			if np.in1d(neighbours[j], self.off_grid).any() and (i==k).all():
				return 1-self.p_slip + 2*self.p_slip/len(self.actions)
			else:
				# Can blow or stay
				if (i == k).all():
					return 2*self.p_slip/len(self.actions)
				return self.p_slip/len(self.actions)

		# If these are not the same point, then we can move there by wind
		if (neighbours != i).any():
			return self.p_slip/len(self.actions)


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

		return obs

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
			'object'          : 'OW',
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

	def _gen_obs(self):
		assert False, "gen_obs needs to be implemented by each environment"

	def _reward(self):
		assert False, "_reward needs to be implemented for each environment"

	def _rand_int(self, low, high):
		"""
		Generate random integer in [low,high[
		"""

		return np.random.randint(low, high)

	def _rand_float(self, low, high):
		"""
		Generate random float in [low,high[
		"""

		return np.random.uniform(low, high)

	def _rand_bool(self):
		"""
		Generate random boolean value
		"""

		return (np.random.randint(0, 2) == 0)

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
			np.random.randint(xLow, xHigh),
			np.random.randint(yLow, yHigh)
		)

	def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
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

	def get_neighbours(self, state=None):
		"""
		Get cells to left, right, up, down
		"""
		if state is None:
			return self.agent_pos + np.array((1,0)), self.agent_pos + np.array((-1,0)), self.agent_pos + np.array((0,1)), self.agent_pos + np.array((0,-1)), self.agent_pos
		else:
			return state + np.array((1,0)), state + np.array((-1,0)), state + np.array((0,1)), state + np.array((0,-1)), state

	def step(self, action):
		self.step_count += 1

		reward = 0
		done = False

		# Confirm action
		if np.random.rand() < self.p_slip:
			action = self.action_space.sample()

		# Get move pos
		mov_pos = self.get_neighbours()[action]

		# Get contents
		mov_cell = self.grid.get(*mov_pos)

		if mov_cell == None or mov_cell.can_overlap():
			self.agent_pos = mov_pos
			reward = self._reward(self.agent_pos)

		if self.step_count > self.max_steps:
			done = True

		obs = self._gen_obs()

		return self.agent_pos, reward, done, {}

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
