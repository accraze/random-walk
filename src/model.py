import numpy as np


def random_walk(dims=1, step_n=100, step_set=[-1, 0, 1]):
    origin = np.zeros((1,dims))
    step_shape = (step_n,dims)
    steps = np.random.choice(a=step_set, size=step_shape)
    path = np.concatenate([origin, steps]).cumsum(0)
    start = path[:1]
    stop = path[-1:]
    return path, start, stop
