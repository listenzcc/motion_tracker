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
            vector=vector,
            name=name,
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
        assert isinstance(
            seg, Segment), 'Invalid input type for seg, it should be Segment'

        axis = self.get_axes_by_name(axis_name, force_single=True)
        axis['children'].append(seg)
        return axis

    def rotate(self, axis, deg, orig=None):
        if orig is None:
            orig = self.body.orig

        self.body.rotate(axis['vector'], deg, orig)
        for _axis in self.axes:
            _axis.rotate(axis['vector'], deg, orig)
            for child in _axis['children']:
                child.rotate(axis['vector'], deg, orig)

        return self


# %%
segment_style = dict(
    marker=dict(size=5, color=['#BBBBBB', 'black']),
    line=dict(width=7, color=['#BBBBBB', 'black']),
    showlegend=True,
    name='Segment-?'
)

axis_style = dict(
    marker=dict(size=1),
    line=dict(width=4, color='cyan'),
    showlegend=False,
    name='Axis-?'
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


def display(segment,
            fig=None,
            opacity=0.7,
            showlegend=False,
            segment_style=segment_style,
            axis_style=axis_style,
            fig_layout=fig_layout):

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
    for j, axe in enumerate(segment.axes):
        # Draw the axis
        _seg = _2ptr_from_vector(axe)
        _style = axis_style.copy()
        _style['line']['color'] = axes_colors[j % len(axes_colors)]
        _style['name'] = '{} Axe {}'.format(segment.name, j)
        _style['showlegend'] = showlegend
        fig.add_trace(go.Scatter3d(
            x=_seg['x'], y=_seg['y'], z=_seg['z'], **_style))

        # Draw the children of the axis
        for child in axe['children']:
            display(child, fig, opacity, showlegend,
                    segment_style, axis_style, fig_layout)

        pass

    fig.update_layout(**fig_layout)

    return fig


# %%
segment = Segment((-1, -2, -3), (1, 1, 0))
segment.body.dest.xyz = 10, 20, 30

segment.body.translate_by((1, 2, 3))
print(segment.body.orig.xyz, segment.body.dest.xyz)

segment.body.translate_to((1, 2, 3))
print(segment.body.orig.xyz, segment.body.dest.xyz)

# %%
