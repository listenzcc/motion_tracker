# %%
from geometric import Vector
from onstart import LOGGER

import plotly.express as px
import plotly.graph_objects as go

# %%


class Segment(object):
    '''
    The segment object

    The body is the vector from orig point to dest point,
    The axes is the branches of the segment.

    The body design is like following:

               |
               |:branch-1
               |
     ==body================
            |
            |:branch-2
            |                   |
            |==body===================
                                |

     body: [orig]==========[dest]

    '''

    def __init__(self, orig, dest, name='Segment'):
        self.body = Vector(orig, dest)
        self.name = name
        self.__children = []
        self.__axes = []
        # LOGGER.info("Segment constructor is initialized")
        pass

    @property
    def children(self):
        return self.__children

    @property
    def axes(self):
        return self.__axes

    def append_child(self, segment):
        if not isinstance(segment, Segment):
            LOGGER.error(
                'Invalid input type for segment, it should be Segment')
            return

        self.__children.append(segment)

        # LOGGER.debug('Append new child {}, it has {} children'.format(
        #     segment, len(self.children)))

        return self.children

    def append_axis(self, vector, name=None, deg=0, deg_limit=(-45, 45)):
        # If vector is not a really Vector,
        # then we will try to convert it into a Vector.
        # Of course, it may raise some errors.
        if not isinstance(vector, Vector):
            if not len(vector) == 2:
                LOGGER.error('Invalid vector: {}'.format(vector))
                return
            vector = Vector(vector[0], vector[1])

        if name is None:
            name = len(self.axes)

        self.__axes.append(dict(
            vector=vector,
            name=name,
            deg=deg,
            deg_limit=deg_limit,
        ))

        return self.axes

    def get_axes_name(self):
        '''
        Get the [name]s of the axes

        Returns:
            - The name list of the axes.
        '''

        names = [e['name'] for e in self.axes]

        return names

    def get_axes_by_name(self, name, force_single=True):
        '''
        Get the axes by the [name].

        Args:
            - name: The name of the selected axes;
            - force_single: If True, it assumes the selection has only one axis, default is True.

        Returns:
            - The selected axes, if force_single is False;
            - The single axis, if force_single is True;
            - None, if force_single is True but the condition is not satisfied.
        '''

        axes = [e for e in self.axes
                if e['name'] == name]

        if force_single:
            if not len(axes) == 1:
                LOGGER.error(
                    'Invalid axes: {} in force_single mode'.format(axes))
                return

            return axes[0]

        return axes

    def rotate(self, axis, deg, orig=None, mode='inner'):
        '''
        Rotate the segment around the [axis] by [deg]rees,
        the [orig]in point of the rotation is optional.

        The mode has the values of 'inner' and 'outer',
        'inner' refers the rotation is original for the segment,
        'outer' refers the rotation is comming from another rotation.

        If in the mode of 'inner', orig will be ignored, and will use the body's orig for instead.

        If mode is not 'inner', the deg will not be checked for whether it is in the deg_limit.
        Other wise, it will be checked. If the result is not-in-range, we will raise an ERROR and do nothing.

        Args:
            - axis: The axis of rotation;
            - deg: The rotation angle in degrees;
            - orig: The origin point or the rotation,
                    if mode is 'linear', it will be ignored,
                    if mode is 'outer', it has to be valid 3d point,
                    if not given, we will use the body's orig for instead;
            - mode: 'inner' or 'outer',
                    'inner' means rotating around the axis by deg from the orig, deg checking will be operated,
                    'outer' means rotating around the direction axis from "far" origin;

        Returns:
            - None for failure of deg checking.
            - The self for success.
        '''

        if mode == 'inner':
            orig = axis['vector'].orig
            if any([axis['deg'] + deg > axis['deg_limit'][1],
                    axis['deg'] + deg < axis['deg_limit'][0]]):
                LOGGER.error('Invalid rotate by {} degrees, limit is {}:{}'.format(
                    deg, axis['deg'], axis['deg_limit']
                ))
                return

            axis['deg'] += deg

        else:
            if orig is None:
                orig = self.body.orig

        self.body.rotate(axis['vector'], deg, orig)

        for _axis in self.axes:
            _axis['vector'].rotate(axis['vector'], deg, orig)

        for child in self.children:
            child.rotate(axis, deg, orig, mode='outer')

        return self


# %%
class SegmentDisplayer(object):
    def __init__(self):
        self.segment_style = dict(
            marker=dict(size=5, color=['#BBBBBB', 'black']),
            line=dict(width=7, color=['#BBBBBB', 'black']),
            showlegend=True,
            name='Segment-?'
        )

        self.axis_style = dict(
            marker=dict(size=1),
            line=dict(width=4, color='cyan'),
            showlegend=False,
            name='Axis-?'
        )

        self.axes_colors = px.colors.qualitative.Plotly

        self.fig_layout = dict(
            width=800,
            height=800,
            autosize=False,
            scene=dict(
                aspectratio=dict(x=.5, y=.8, z=.4),
                # aspectmode='manual',
                aspectmode='data',
                camera=dict(up=dict(x=0, y=1, z=0)),
                xaxis=dict(range=[-30, 20]),
                yaxis=dict(range=[-30, 50]),
                zaxis=dict(range=[-20, 20]),
            ),
        )

        self.showlegend = False
        self.opacity = 0.7
        pass

    def plot(self, segment, fig=None, opacity=None, showlegend=None, axes_colors=None, segment_style=None, axis_style=None, fig_layout=None):
        if opacity is None:
            opacity = self.opacity

        if showlegend is None:
            showlegend = self.showlegend

        if axes_colors is None:
            axes_colors = self.axes_colors

        if segment_style is None:
            segment_style = self.segment_style

        if axis_style is None:
            axis_style = self.axis_style

        if fig_layout is None:
            fig_layout = self.fig_layout

        def _2ptr_from_segment(segment):
            _orig_xyz = segment.body.orig.xyz
            _dest_xyz = segment.body.dest.xyz
            seg = dict(
                x=[_orig_xyz[0], _dest_xyz[0]],
                y=[_orig_xyz[1], _dest_xyz[1]],
                z=[_orig_xyz[2], _dest_xyz[2]],
            )
            return seg

        def _2ptr_from_vector(vector):
            _orig_xyz = vector.orig.xyz
            _dest_xyz = vector.dest.xyz
            seg = dict(
                x=[_orig_xyz[0], _dest_xyz[0]],
                y=[_orig_xyz[1], _dest_xyz[1]],
                z=[_orig_xyz[2], _dest_xyz[2]],
            )
            return seg

        # Create fig if not provide
        if fig is None:
            fig = px.scatter_3d()

        # Draw the body
        _seg = _2ptr_from_segment(segment)
        _style = segment_style.copy()
        _style['name'] = segment.name
        _style['opacity'] = opacity
        _style['showlegend'] = showlegend
        fig.add_trace(go.Scatter3d(
            x=_seg['x'], y=_seg['y'], z=_seg['z'], **_style))

        # Draw the axes
        for j, axis in enumerate(segment.axes):
            # Draw the axis
            _seg = _2ptr_from_vector(axis['vector'])
            _style = axis_style.copy()
            _style['line']['color'] = axes_colors[j % len(axes_colors)]
            _style['name'] = '{} Axe {}'.format(segment.name, j)
            _style['showlegend'] = showlegend
            fig.add_trace(go.Scatter3d(
                x=_seg['x'], y=_seg['y'], z=_seg['z'], **_style))

        # Draw the children of the axis
        for child in segment.children:
            self.plot(child, fig, opacity, showlegend, axes_colors,
                      segment_style, axis_style, fig_layout)
            pass

        fig.update_layout(**fig_layout)

        return fig


# %%
