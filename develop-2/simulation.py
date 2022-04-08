# %%
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go

from segment import Segment, SegmentDisplayer

# %%


def mk_segments():
    # Shoulder
    shoulder = Segment([0, 40, 0], [10, 40, 0], name='shoulder')

    shoulder.append_axis([shoulder.body.dest.xyz,
                          shoulder.body.dest.xyz + [1, 0, 0]],
                         name='0', deg=0, deg_limit=(-90, 45))

    shoulder.append_axis([shoulder.body.dest.xyz,
                          shoulder.body.dest.xyz + [0, 1, 0]],
                         name='1', deg=0, deg_limit=(-60, 10))

    shoulder.append_axis([shoulder.body.dest.xyz,
                          shoulder.body.dest.xyz + [0, 0, 1]],
                         name='2', deg=0, deg_limit=(0, 145))

    # Upper Arm
    upper_arm = Segment(shoulder.body.dest.xyz,
                        shoulder.body.dest.xyz + [0, -20, 0], name='upperArm')

    upper_arm.append_axis([upper_arm.body.dest.xyz,
                           upper_arm.body.dest.xyz + [1, 0, 0]],
                          name='0', deg=0, deg_limit=(-145, 0))

    upper_arm.append_axis([upper_arm.body.dest.xyz,
                           upper_arm.body.dest.xyz + [0, 0, 1]],
                          name='2', deg=0, deg_limit=(-20, 20))

    # Lower Arm
    lower_arm = Segment(upper_arm.body.dest.xyz,
                        upper_arm.body.dest.xyz + [0, -20, 0], name='lowerArm')

    # Link them
    shoulder.append_child(upper_arm)
    upper_arm.append_child(lower_arm)

    return shoulder, upper_arm, lower_arm


segments = mk_segments()
segments

# %%


def random_values(ranges):
    return [
        np.random.uniform(e[-1][0], e[-1][1], size=1)[0]
        for e in ranges
    ]


ranges = []
for seg_idx, seg in enumerate(segments):
    for name in seg.get_axes_name():
        axis = seg.get_axes_by_name(name)
        ranges.append((seg_idx, name, axis['deg_limit']))
ranges

# %%
records = []

count = 10000 * 1
for _ in tqdm(range(count)):
    segments = mk_segments()
    degrees = random_values(ranges)

    for j, rng in enumerate(ranges):
        deg = degrees[j]
        seg = segments[rng[0]]
        name = rng[1]

        seg.rotate(seg.get_axes_by_name(name), deg)

    xyz = segments[-1].body.dest.xyz

    records.append([e for e in xyz] + degrees)

pd_records = pd.DataFrame(records,
                          columns=['x', 'y', 'z', 'a00', 'a01', 'a02', 'a10', 'a11'])
pd_records

# %%
pca = PCA(n_components=3)
X = np.array(pd_records[[e for e in pd_records.columns if e.startswith('a')]])
pca.fit(X)
print(pca.explained_variance_ratio_)
x = pca.transform(X)


def _scale(x):
    m, M = np.min(x), np.max(x)
    return (x - m) / (M - m)


for c in range(x.shape[1]):
    x[:, c] = _scale(x[:, c])

pd_records[['pc0', 'pc1', 'pc2']] = x
pd_records

# %%


def _color(se):
    r = int(se['pc0'] * 255)
    g = int(se['pc1'] * 255)
    b = int(se['pc2'] * 255)
    return '#' + ('{:02X}' * 3).format(r, g, b)


pd_records['color'] = pd_records.apply(_color, axis=1)
pd_records


# %%
kwargs = dict(marker=dict(size=2,
                          opacity=0.1,
                          line=None),
              selector=dict(mode='markers'))

for col in ['pc0', 'pc1', 'pc2']:
    fig = px.scatter_3d(pd_records, x='x', y='y', z='z', color=col, title=col)
    fig.update_traces(**kwargs)
    fig.show()

# %%
fig = px.scatter_3d(pd_records, x='x', y='y', z='z', title='Colored by PCs')
kwargs['marker']['color'] = pd_records['color']
fig.update_traces(**kwargs)
fig.show()
# %%
