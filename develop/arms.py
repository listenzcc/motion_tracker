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

        self.axes_limit[axis_idx][0] += deg

        if self.axes_limit[axis_idx][0] > self.axes_limit[axis_idx][2]:
            deg -= (self.axes_limit[axis_idx][0] -
                    self.axes_limit[axis_idx][2])
            self.axes_limit[axis_idx][0] = self.axes_limit[axis_idx][2]

        if self.axes_limit[axis_idx][0] < self.axes_limit[axis_idx][1]:
            deg += (self.axes_limit[axis_idx][1] -
                    self.axes_limit[axis_idx][0])
            self.axes_limit[axis_idx][0] = self.axes_limit[axis_idx][1]

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
style_segment = dict(
    marker=dict(size=5, color=['#BBBBBB', 'black']),
    line=dict(width=7, color=['#BBBBBB', 'black']),
    showlegend=True,
    name='Segment-1'
)

style_axes = dict(
    marker=dict(size=1),
    line=dict(width=4, color='cyan'),
    showlegend=False,
    name='Axe'
)

axes_colors = px.colors.qualitative.Plotly

fig_layout = dict(
    width=800,
    height=800,
    autosize=False,
    scene=dict(
        aspectratio=dict(x=1, y=1, z=1),
        aspectmode='manual',
        camera=dict(up=dict(x=0, y=1, z=0)),
        xaxis=dict(range=[-30, 20]),
        yaxis=dict(range=[-30, 10]),
        zaxis=dict(range=[-20, 20]),
    ),
)


def display(segment, fig=None, opacity=0.7, showlegend=False):
    if fig is None:
        fig = px.scatter_3d()

    seg, axes = segment.get_xyz()

    _style = style_segment.copy()
    _style['name'] = segment.name
    _style['opacity'] = opacity
    _style['showlegend'] = showlegend
    fig.add_trace(go.Scatter3d(
        x=seg['x'], y=seg['y'], z=seg['z'], **_style))

    for j, axe in enumerate(axes):
        _style = style_axes.copy()
        _style['line']['color'] = axes_colors[j % len(axes_colors)]
        _style['name'] = '{} Axe {}'.format(segment.name, j)
        _style['showlegend'] = showlegend
        fig.add_trace(go.Scatter3d(
            x=axe['x'], y=axe['y'], z=axe['z'], **_style))

    fig.update_layout(
        **fig_layout
    )

    for seg in segment.links:
        display(seg, fig, opacity, showlegend)

    return fig


# %%
# segment = Segment([0, 0, 0], [0, 0, 5], [[0, 1, 1], [1, 0, 0]], 'Seg 1')
# segment2 = Segment([0, 0, 5], [0, 5, 0], [[0, 1, 1], [1, 0, 0]], 'Seg 2')
# segment3 = Segment([0, 5, 0], [0, 5, 10], [[0, 1, 1], [1, 0, 0]], 'Seg 3')

# segment.links_to(segment2)
# segment2.links_to(segment3)

# seg, axes = segment.get_xyz()
# seg, axes

# fig = display(segment, opacity=0.2)
# # fig.show()

# segment.rotate_inside(1, 30)
# segment2.rotate_inside(0, 30)
# fig = display(segment, fig, opacity=0.2)
# # fig.show()

# segment.rotate_outside([-1, 0, 0], [0, 1, 0], 45)
# fig = display(segment, fig, opacity=1)
# fig.show()

# %%
# shoulder = Segment([0, 0, 0], [5, 0, 0],
#                    axes=[],
#                    name='Shoulder')

# upper_arm = Segment(shoulder.end_point, shoulder.end_point + [0, -10, 0],
#                     axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
#                     name='upperArm')

# lower_arm = Segment(upper_arm.end_point, upper_arm.end_point + [0, -10, 0],
#                     axes=[[1, 0, 0], [0, 0, 1]],
#                     name='lowerArm')

# shoulder.links_to(upper_arm)
# upper_arm.links_to(lower_arm)
# fig = display(shoulder, opacity=0.2)

# res = 10
# for idx in tqdm(range(len(upper_arm.axes)), 'Axes'):
#     target_angle = np.random.randint(-30, -10)
#     for _ in tqdm(range(res), 'Res'):
#         upper_arm.rotate_inside(idx, target_angle / res)
#         fig = display(shoulder, fig, opacity=0.2)

# for idx in tqdm(range(len(lower_arm.axes)), 'Axes'):
#     target_angle = np.random.randint(-30, -10)
#     for _ in tqdm(range(res), 'Res'):
#         lower_arm.rotate_inside(idx, target_angle / res)
#         fig = display(shoulder, fig, opacity=0.2)

# fig = display(shoulder, fig, opacity=1, showlegend=True)
# fig.show()

# %%
shoulder = Segment([0, 0, 0], [5, 0, 0],
                   axes=[],
                   name='Shoulder')

upper_arm = Segment(shoulder.end_point, shoulder.end_point + [0, -10, 0],
                    axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    axes_limit=[[0, -90, 45], [0, -10, 10], [0, 0, 145]],
                    name='upperArm')

lower_arm = Segment(upper_arm.end_point, upper_arm.end_point + [0, -10, 0],
                    axes=[[1, 0, 0], [0, 0, 1]],
                    axes_limit=[[0, -145, 0], [0, -90, 0]],
                    name='lowerArm')

shoulder.links_to(upper_arm)
upper_arm.links_to(lower_arm)
fig = display(shoulder, opacity=0.2)

for _ in tqdm(range(200), 'Simulation'):
    idx = np.random.randint(len(upper_arm.axes))
    deg = np.random.random() * 2 - 0.5
    upper_arm.rotate_inside(idx, deg)

    idx = np.random.randint(len(lower_arm.axes))
    deg = np.random.random() * 2 - 1.5
    lower_arm.rotate_inside(idx, deg)

    if _ % 10 == 0:
        fig = display(shoulder, fig, opacity=0.2)

fig = display(shoulder, fig, opacity=1, showlegend=True)
fig.show()

# %%
upper_arm.axes_limit, lower_arm.axes_limit
# %%
