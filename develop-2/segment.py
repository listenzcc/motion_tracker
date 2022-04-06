# %%

import numpy as np

from scipy.spatial.transform import Rotation as R

from geometric import Vector
from onstart import LOGGER

# %%


class Segment(object):
    def __init__(self, start, stop):
        self.body = Vector(start, stop)
        self.__axes = []
        LOGGER.info("Segment constructor is initialized")
        pass

    @property
    def axes(self):
        return self.__axes

    @axes.setter
    def axes(self, args):
        vector = args['vector']
        name = args.get('name', 'N.A.')
        deg = args.get('deg', 0)
        deg_limit = args.get('deg_limit', (-45, 45))
        children = args.get('children', [])
        self.__axes.append(dict(
            name=name,
            vector=vector,
            deg=deg,
            deg_limit=deg_limit,
            children=children
        ))
        pass

    def get_axes_by_name(self, name, force_single=False):
        axes = [e for e in self.axes
                if e['name'] == name]

        if force_single:
            assert len(axes) == 0, 'Invalid axes: {}'.format(axes)
            return axes[0]

        return axes

    def add_child_to_axis(self, axis_name, seg):
        axis = self.get_axes_by_name(axis_name, force_single=True)
        axis['children'].append(seg)
        return axis

    def rotate(self, orig=None):
        pass


# %%
segment = Segment((-1, -2, -3), (1, 1, 0))
segment.body.dest.xyz = 10, 20, 30

segment.body.translate_by((1, 2, 3))
print(segment.body.orig.xyz, segment.body.dest.xyz)

segment.body.translate_to((1, 2, 3))
print(segment.body.orig.xyz, segment.body.dest.xyz)
# %%
