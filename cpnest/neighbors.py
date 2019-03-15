import numpy as np
from .parameter import LivePoint

from sklearn.neighbors import KDTree

def constructKDTree(evolution_points, **kwargs):
	n	= len(evolution_points)
	dim = evolution_points[0].dimension
	points = np.array([evolution_points[i].values for i in range(n)])
	tree = KDTree(points, leaf_size = 2)
	return tree

def queryKDTree(tree, point, d):
	p = np.expand_dims(point.values, axis=0)
	return tree.query(p, k = d)
