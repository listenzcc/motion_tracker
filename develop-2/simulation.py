# %%
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go

from segment import Segment, SegmentDisplayer

# %%

angle_ranges = dict(
    a00=(-90, 45),
    a01=(-60, 10),
    a02=(0, 145),
    a10=(-145, 0),
    a12=(-20, 20),
)

angle_columns = [e for e in angle_ranges]


def mk_segments():
    # Shoulder
    shoulder = Segment([0, 40, 0], [10, 40, 0], name='shoulder')

    shoulder.append_axis([shoulder.body.dest,
                          shoulder.body.dest + [1, 0, 0]],
                         name='0', deg=0, deg_limit=(-90, 45))

    shoulder.append_axis([shoulder.body.dest,
                          shoulder.body.dest + [0, 1, 0]],
                         name='1', deg=0, deg_limit=(-60, 10))

    shoulder.append_axis([shoulder.body.dest,
                          shoulder.body.dest + [0, 0, 1]],
                         name='2', deg=0, deg_limit=(0, 145))

    # Upper Arm
    upper_arm = Segment(shoulder.body.dest,
                        shoulder.body.dest + [0, -20, 0], name='upperArm')

    upper_arm.append_axis([upper_arm.body.dest,
                           upper_arm.body.dest + [1, 0, 0]],
                          name='0', deg=0, deg_limit=(-145, 0))

    upper_arm.append_axis([upper_arm.body.dest,
                           upper_arm.body.dest + [0, 0, 1]],
                          name='2', deg=0, deg_limit=(-20, 20))

    # Lower Arm
    lower_arm = Segment(upper_arm.body.dest,
                        upper_arm.body.dest + [0, -20, 0], name='lowerArm')

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
for _ in tqdm(range(count), 'Simulation by {}'.format(count)):
    segments = mk_segments()
    degrees = random_values(ranges)

    for j, rng in enumerate(ranges):
        deg = degrees[j]
        seg = segments[rng[0]]
        name = rng[1]

        seg.rotate(seg.get_axes_by_name(name), deg)

    xyz = segments[-1].body.dest

    records.append([e for e in xyz] + degrees)

pd_records = pd.DataFrame(records,
                          columns=['x', 'y', 'z'] + angle_columns)
pd_records

# %%
pca = PCA(n_components=3)
X = np.array(pd_records[angle_columns])
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
d = pd_records[angle_columns].to_numpy()
angle_distances = np.linalg.norm(d, axis=1)
pd_records['angleDistance'] = angle_distances
pd_records

# %%
d = pd_records[['x', 'y', 'z']].to_numpy()
n = len(d)
coef_matrix = np.zeros((n, n))
print(d.shape, coef_matrix.shape)
for j in tqdm(range(n), 'Compute Coefficients'):
    coef_matrix[j] = np.linalg.norm(d - d[j], axis=1)

np.fill_diagonal(coef_matrix, np.inf)

coef_matrix

# %%
trace_kwargs = dict(marker=dict(size=2,
                                opacity=0.3,
                                line=None),
                    selector=dict(mode='markers'))

# %%
nearest_values = np.min(coef_matrix, axis=0)
pd_records['nearestValue'] = nearest_values
pd_records

# %%

pd_records['selected'] = False

ignores = []
for j in tqdm(range(n), 'Shrinking Nodes'):
    if j in ignores:
        continue

    coef_vec = coef_matrix[j]
    indexes = np.nonzero(coef_vec < 2)[0].tolist()

    if len(indexes) > 1:
        pd = pd_records.loc[indexes]
        m = pd['angleDistance'].min()
        pd['selected'] = pd['angleDistance'] == m
        pd_records.loc[pd.query(
            'angleDistance == {}'.format(m)).index, 'selected'] = True

    if len(indexes) == 1:
        pd_records.loc[indexes[0], 'selected'] = True

    ignores += indexes

print(len(ignores))
new_pd_records = pd_records.query('selected == True')
new_pd_records


# %%
for col in ['pc0', 'pc1', 'pc2', 'angleDistance', 'nearestValue']:
    fig = px.scatter_3d(pd_records, x='x', y='y', z='z', color=col, title=col)
    fig.update_traces(**trace_kwargs)
    fig.show()

# %%
fig = px.scatter_3d(pd_records, x='x', y='y', z='z', title='Colored by PCs')
trace_kwargs['marker']['color'] = pd_records['color']
fig.update_traces(**trace_kwargs)
fig.show()

# %%
fig = px.scatter_3d(new_pd_records, x='x', y='y',
                    z='z', title='Colored by PCs')
trace_kwargs['marker']['color'] = new_pd_records['color']
fig.update_traces(**trace_kwargs)
fig.show()

# %%
_new_pd_records = new_pd_records.copy()

for col in angle_columns:
    m, M = angle_ranges[col]
    _new_pd_records[col] = _new_pd_records[col].map(
        lambda e: (e - m) / (M - m))

barmode = 'overlay'  # 'relative'
barmode = 'relative'

for col in angle_columns:
    fig = px.histogram(_new_pd_records, x=col, title=col, barmode=barmode)
    fig.show()

barmode = 'overlay'  # 'relative'
fig = px.histogram(_new_pd_records, x=angle_columns, barmode=barmode)
fig.show()

# %%
vars = dict()
for col in angle_columns:
    d = _new_pd_records[col]
    h, b = np.histogram(d)
    vars[col] = np.std(h)
    print(h)

print(vars)

# %%
