# %%
from matplotlib.pyplot import hexbin
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA

from tqdm.auto import tqdm

from segment import Segment

# %%


def mk_segment():
    shoulder = Segment([0, 40, 0], [10, 40, 0], axes=[], name='Shoulder')

    upper_arm = Segment(shoulder.end_point,
                        shoulder.end_point + [0, -20, 0],
                        axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        axes_limit=[[0, -90, 45], [0, -60, 10], [0, 0, 145]],
                        name='upperArm')

    lower_arm = Segment(upper_arm.end_point,
                        upper_arm.end_point + [0, -20, 0],
                        axes=[[1, 0, 0], [0, 0, 1]],
                        axes_limit=[[0, -145, 0], [0, -20, 20]],
                        name='lowerArm')

    shoulder.links_to(upper_arm)
    upper_arm.links_to(lower_arm)

    return shoulder, upper_arm, lower_arm


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
        aspectratio=dict(x=60 / 130, y=1, z=70 / 130),
        # aspectmode='manual',
        camera=dict(up=dict(x=0, y=1, z=0)),
        xaxis=dict(range=[-10, 50]),
        yaxis=dict(range=[-30, 100]),
        zaxis=dict(range=[-20, 50]),
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


def random_values(ranges):
    return [
        np.random.uniform(e[0], e[1], size=1)[0]
        for e in ranges
    ]

# %%


records = []

shoulder, upper_arm, lower_arm = mk_segment()

ranges = [
    upper_arm.axes_limit[0][1:],
    upper_arm.axes_limit[1][1:],
    upper_arm.axes_limit[2][1:],
    lower_arm.axes_limit[0][1:],
    lower_arm.axes_limit[1][1:],
]

count = 10000 * 1
for _ in tqdm(range(count)):
    shoulder, upper_arm, lower_arm = mk_segment()

    angles = random_values(ranges)

    upper_arm.rotate_inside(0, angles[0])
    upper_arm.rotate_inside(1, angles[1])
    upper_arm.rotate_inside(2, angles[2])

    lower_arm.rotate_inside(0, angles[3])
    lower_arm.rotate_inside(1, angles[4])

    positions = [e for e in lower_arm.end_point]

    records.append(positions + angles)


records = pd.DataFrame(
    records, columns=['x', 'y', 'z', 'a00', 'a01', 'a02', 'a10', 'a11'])
records

# %%
pca = PCA(n_components=3)
X = np.array(records[[e for e in records.columns if e.startswith('a')]])
pca.fit(X)
print(pca.explained_variance_ratio_)
x = pca.transform(X)


def _scale(x):
    m, M = np.min(x), np.max(x)
    return (x - m) / (M - m)


for c in range(x.shape[1]):
    x[:, c] = _scale(x[:, c])

records[['pc0', 'pc1', 'pc2']] = x
records

# %%


def _color(se):
    r = int(se['pc0'] * 255)
    g = int(se['pc1'] * 255)
    b = int(se['pc2'] * 255)
    return '#' + ('{:02X}' * 3).format(r, g, b)


records['color'] = records.apply(_color, axis=1)
records

# %%
kwargs = dict(marker=dict(size=2,
                          opacity=0.1,
                          line=None),
              selector=dict(mode='markers'))

for col in ['pc0', 'pc1', 'pc2']:
    fig = px.scatter_3d(records, x='x', y='y', z='z', color=col, title=col)
    fig.update_traces(**kwargs)
    fig.show()

# %%
fig = px.scatter_3d(records, x='x', y='y', z='z', title='Colored by PCs')
kwargs['marker']['color'] = records['color']
fig.update_traces(**kwargs)
# fig.show()

shoulder, upper_arm, lower_arm = mk_segment()
# display(shoulder, fig)

se = records.iloc[np.random.choice(records.index)]
# print(se)

for r in np.linspace(0, 1, 8, endpoint=True):
    shoulder, upper_arm, lower_arm = mk_segment()
    upper_arm.rotate_inside(0, r * se['a00'])
    upper_arm.rotate_inside(1, r * se['a01'])
    upper_arm.rotate_inside(2, r * se['a02'])
    lower_arm.rotate_inside(0, r * se['a10'])
    lower_arm.rotate_inside(1, r * se['a11'])
    display(shoulder, fig)

fig.show()

fig = px.scatter_3d(records, x='x', y='y', z='z', title='Colored by PCs')
kwargs['marker']['color'] = records['color']
fig.update_traces(**kwargs)
# fig.show()

shoulder, upper_arm, lower_arm = mk_segment()
# display(shoulder, fig)

a = np.array([se['x'], se['y'], se['z']])
b = np.array(records[['x', 'y', 'z']])
c = np.linalg.norm((b-a), axis=1)
d = np.array(records[['a00', 'a01', 'a02', 'a10', 'a11']])
e = np.linalg.norm(d, axis=1)
select = np.argmin(c * 5 + e)

se = records.iloc[select]
# print(se)

# shoulder, upper_arm, lower_arm = mk_segment()
for r in np.linspace(0, 1, 8, endpoint=True):
    shoulder, upper_arm, lower_arm = mk_segment()
    upper_arm.rotate_inside(0, r * se['a00'])
    upper_arm.rotate_inside(1, r * se['a01'])
    upper_arm.rotate_inside(2, r * se['a02'])
    lower_arm.rotate_inside(0, r * se['a10'])
    lower_arm.rotate_inside(1, r * se['a11'])
    display(shoulder, fig)

fig.show()

# %%
records


# %%
