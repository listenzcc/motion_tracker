# %%
import numpy as np
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation as R

# %%
import plotly.express as px
import plotly.graph_objects as go

# %%


def _deg2rad(deg):
    return deg / 180 * np.pi


def _rad2deg(rad):
    return rad / np.pi * 180


def _float(e, format=np.float32):
    return np.array(e, dtype=format)


def _normalize(vec):
    return vec / np.linalg.norm(vec)

# %%


class Segment(object):
    '''
    The segments of the arm.
    '''

    def __init__(self, start_point, end_point, axes=[], axes_limit=None, name='Seg'):
        '''
        Initialize the segment with 3 elements

        Args:
        - start_point: Where it starts, the shape is (3, );
        - end_point: Where it ends, the shape is (3, );
        - axes: The axes of rotation, the shape is (n, 3), n is the number of axes.
                The axes are normalized to unit length;
        - axes_limit: The angle limit of the rotation, the format is (x, x0, x1)
                      - x is the current angle (in degree);
                      - x0 is the lower boundary;
                      - x1 is the upper boundary;
                      If not provided, a very limitted range (-10, 10) is used;
        - name: The name of the segment.
        '''
        self.start_point = _float(start_point)
        self.end_point = _float(end_point)

        self.axes = [_normalize(_float(e)) for e in axes]

        if axes_limit is None:
            self.axes_limit = [_float([0, -10, 10]) for _ in axes]
        else:
            assert(all([e[2] > e[1] for e in axes_limit]))
            self.axes_limit = [_float(e) for e in axes_limit]

        self.name = name
        self.links = []

    def links_to(self, segment):
        '''
        Link the Segment object to another segment.

        Args:
        - segment: The linked segment object, Segment object.

        Returns:
        - The length of the links
        '''
        self.links.append(segment)
        return len(self.links)

    def rotate_inside(self, axis_idx, deg):
        '''
        Rotate the segment using inside method,

        Args:
        - axis_idx: The rotation axis idx of the rotation, an int.
                    It rotates only along the axes;
        - deg: The angle (in degree) of rotation, an float.
        '''
        # Safe input
        axis = self.axes[axis_idx]

        # self.axes_limit[axis_idx][0] += deg

        if all([
            self.axes_limit[axis_idx][0] + deg <= self.axes_limit[axis_idx][2],
            self.axes_limit[axis_idx][0] + deg >= self.axes_limit[axis_idx][1],
        ]):
            self.axes_limit[axis_idx][0] += deg
        else:
            return
            print('-' * 300)

        # if self.axes_limit[axis_idx][0] + deg > self.axes_limit[axis_idx][2]:
        #     raise ValueError('Deg is {}, {}'.format(deg, self.axes_limit))
        #     print('-' * 10, deg, oifjwoijef)
        #     deg = self.axes_limit[axis_idx][2] - self.axes_limit[axis_idx][0]
        #     self.axes_limit[axis_idx][0] = self.axes_limit[axis_idx][2]
        #     print('-' * 10, deg)

        # if self.axes_limit[axis_idx][0] + deg < self.axes_limit[axis_idx][1]:
        #     deg = self.axes_limit[axis_idx][1] - self.axes_limit[axis_idx][0]
        #     self.axes_limit[axis_idx][0] = self.axes_limit[axis_idx][1]

        r = R.from_rotvec(axis * _deg2rad(deg))

        # Rotation self
        self.end_point = self.start_point + \
            r.apply(self.end_point - self.start_point)

        for j, ax in enumerate(self.axes):
            self.axes[j] = r.apply(ax)

        # Rotation links
        for segment in self.links:
            segment.rotate_outside(self.start_point, axis, deg)

        return

    def rotate_outside(self, origin_point, axis, deg):
        '''
        Rotate the segment using outside method,

        Args:
        - origin_point: The origin_point of the rotation;
        - axis: The rotation axis, the shape is (3,);
        - deg: The angle (in degrees) of rotation, an float.
        '''
        # Safe input
        origin_point = _float(origin_point)
        axis = _normalize(_float(axis))
        r = R.from_rotvec(axis * _deg2rad(deg))

        # Rotation self
        self.start_point, self.end_point = r.apply([
            self.start_point - origin_point,
            self.end_point - origin_point
        ])
        self.start_point += origin_point
        self.end_point += origin_point

        for j, ax in enumerate(self.axes):
            self.axes[j] = r.apply(ax)

        # Rotation links
        for segment in self.links:
            segment.rotate_outside(origin_point, axis, deg)

        return

    def get_xyz(self):
        '''
        Get the x-, y- and z- parameters of the segment and the axes.

        Returns:
        - segment: The xyz coordinate of the start_ and end_point, the format is {'x': [x0, x1]}, and so as the y and z;
        - axes: The list of the xyz coordinates of the axes, the format of the elements is the same as the segment.
        '''
        segment = dict(
            x=[self.start_point[0], self.end_point[0]],
            y=[self.start_point[1], self.end_point[1]],
            z=[self.start_point[2], self.end_point[2]]
        )

        axes = []
        for a in self.axes:
            ax = dict(
                x=[self.start_point[0], self.start_point[0] + a[0]],
                y=[self.start_point[1], self.start_point[1] + a[1]],
                z=[self.start_point[2], self.start_point[2] + a[2]]
            )
            axes.append(ax)

        return segment, axes


# %%
