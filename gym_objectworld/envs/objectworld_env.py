import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.toy_text import discrete

import numpy as np
from io import StringIO
import math

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

def generate_map(size, n_rewards):

	if n_rewards == 2:
		res = np.chararray((size, size))
		res[0][math.ceil(size/2)] == "S"
		res[-1][-1] == "G"
		res[-1][0] == "g"
		return ["".join(x) for x in res]
	elif n_rewards == 4:
		pass
	else:
		pass

class GridWorldEnv(discrete.DiscreteEnv):
	metadata={'render.modes': ['human', 'ansi']}

	def __init__(self, size=5, p_slip = 0.3, n_rewards = 2):
		self.viewer = None

		desc = generate_map()

		self.desc = desc = np.asarray(desc, dtype="c")
		self.n_row, self.n_col = nrow, ncol = desc.shape

		self.reward_range = (0,1)

		nA = 4
		nS = nrow*ncol

		isd = np.array(desc == b"S").astype("float64").ravel()
		isd /= isd.sum()

		P = {s: {a: [] for a in range(nA)} for s in range(nS)}

		def to_s(row, col):
			return row*ncol + col

		def inc(row, col, a):
			if a==LEFT:
				col = max(col-1, 0)
			elif a == DOWN:
				row = min(row+1, nrow-1)
			elif a == RIGHT:
				col = min(col + 1, ncol-1)
			elif a == UP:
				row = max(row-1,0)
			return (row, col)

		def update_probability_matrix(row, col, action):
			newrow, newcol = inc(row, col, action)
			newstate = to_s(newrow, newcol)
			newletter = desc[newrow, newcol]
			done = bytes(newletter) in b"Gg"
			reward = float(newletter == b"Gg")
			return newstate, reward, done
			
		for row in range(nrow):
			for col in range(ncol):
				s = to_s(row, col)
				for a in range(4):
					li = P[s][a]
					letter = desc[row, col]
					if letter in b"GH":
						li.append((1.0, s, 0, True))
					else:
						if is_slippery:
							for b in [(a - 1) % 4, a, (a + 1) % 4]:
								li.append(
									(1.0 / 3.0, *update_probability_matrix(row, col, b))
								)
						else:
							li.append((1.0, *update_probability_matrix(row, col, a)))

		super().__init__(nS, nA, P, isd)


	def render(self, mode="human"):
		outfile = StringIO() if mode == "ansi" else sys.stdout

		row, col = self.s // self.ncol, self.s % self.ncol
		desc = self.desc.tolist()
		desc = [[c.decode("utf-8") for c in line] for line in desc]
		desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
		if self.lastaction is not None:
			outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
		else:
			outfile.write("\n")
		outfile.write("\n".join("".join(line) for line in desc) + "\n")

		if mode != "human":
			with closing(outfile):
				return outfile.getvalue()