# %%
import numpy as np
from tqdm.auto import tqdm

# %%
import plotly.express as px
import plotly.graph_objects as go

# %%
from segment import Segment

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
