# %%
from sklearn import linear_model
from sklearn import svm
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go

from segment import Segment, SegmentDisplayer

# %%

''' md
Direction defines:
- x, 0, right
- y, 1, up
- z, 2, front
'''

angle_ranges = dict(
    a00=(-90, 45),
    a01=(-60, 10),
    a02=(0, 145),
    a10=(-145, 0),
    a12=(-20, 20),
    a22=(-90, 90),
)

angle_columns = [e for e in angle_ranges]


# %%
def mk_segments():
    # Shoulder
    shoulder = Segment([0, 40, 0], [10, 40, 0], name='shoulder')

    shoulder.append_axis([shoulder.body.dest,
                          shoulder.body.dest + [1, 0, 0]],
                         name='0', deg=0, deg_limit=angle_ranges['a00'])

    shoulder.append_axis([shoulder.body.dest,
                          shoulder.body.dest + [0, 1, 0]],
                         name='1', deg=0, deg_limit=angle_ranges['a01'])

    shoulder.append_axis([shoulder.body.dest,
                          shoulder.body.dest + [0, 0, 1]],
                         name='2', deg=0, deg_limit=angle_ranges['a02'])

    # Upper Arm
    upper_arm = Segment(shoulder.body.dest,
                        shoulder.body.dest + [0, -20, 0], name='upperArm')

    upper_arm.append_axis([upper_arm.body.dest,
                           upper_arm.body.dest + [1, 0, 0]],
                          name='0', deg=0, deg_limit=angle_ranges['a10'])

    upper_arm.append_axis([upper_arm.body.dest,
                           upper_arm.body.dest + [0, 0, 1]],
                          name='2', deg=0, deg_limit=angle_ranges['a12'])

    # Lower Arm
    lower_arm = Segment(upper_arm.body.dest,
                        upper_arm.body.dest + [0, -20, 0], name='lowerArm')

    lower_arm.append_axis([lower_arm.body.dest,
                           lower_arm.body.dest + [0, 0, 1]],
                          name='0', deg=0, deg_limit=angle_ranges['a22'])

    # Hand
    hand = Segment(lower_arm.body.dest,
                   lower_arm.body.dest + [0, -10, 0], name='hand')

    # Link them
    shoulder.append_child(upper_arm)
    upper_arm.append_child(lower_arm)
    lower_arm.append_child(hand)

    return shoulder, upper_arm, lower_arm, hand


# Make the left arm
segments = mk_segments()

# Display the left arm
displayer = SegmentDisplayer()
fig = displayer.plot(segments[0], title='Left arm')
fig.show()


# %%

def simulate(segments, angle_columns=angle_columns, count=10000):
    '''
    Simulation by random sampling

    Inputs:
        - segments: The list of segments;
        - angle_columns: The column names of the angle values;
        - count: The repeat count of the simulation.

    Returns
        - The simulation records.
    '''
    # Prepare ranges
    ranges = []
    for seg_idx, seg in enumerate(segments):
        for name in seg.get_axes_name():
            axis = seg.get_axes_by_name(name)
            ranges.append((seg_idx, name, axis['deg_limit']))
    ranges

    def random_values(ranges):
        return [
            np.random.uniform(e[-1][0], e[-1][1], size=1)[0]
            for e in ranges
        ]

    # Random angles
    lst = []

    for _ in tqdm(range(count), 'Simulation by {}'.format(count)):
        segments = mk_segments()
        degrees = random_values(ranges)

        for j, rng in enumerate(ranges):
            deg = degrees[j]
            seg = segments[rng[0]]
            name = rng[1]

            seg.rotate(seg.get_axes_by_name(name), deg)

        xyz = segments[-1].body.dest

        lst.append([e for e in xyz] + degrees)

    records = pd.DataFrame(lst,
                           columns=['x', 'y', 'z'] + angle_columns)
    return records


def compute_pca_columns(records, angle_columns=angle_columns, color=True):
    '''
    Compute pca decomposition of angles,
    the n_components is fixed to 3.

    The ['pc0', 'pc1', 'pc2'] columns will be added **IN-PLACE**.

    If color is True,
    - The pca values will be projected into the range of 0 ~ 1;
    - The scaled pca values will be converted into colors;
    - The 'color' column will be added to the records **IN-PLACE**;
    - The colors is like #FFFF00, for pc0=255, pc1=255, pc2=0.

    Inputs:
        - records: The records of the simulation;
        - angle_columns: The column names of the angle values;
        - color: The option to compute the color column.

    Returns:
        - The explained variance ratio of the pca

    '''

    pca = PCA(n_components=3)
    X = np.array(records[angle_columns])
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    x = pca.transform(X)
    records[['pc0', 'pc1', 'pc2']] = x

    if color:
        def _scale(x):
            m, M = np.min(x), np.max(x)
            return (x - m) / (M - m)

        for c in range(x.shape[1]):
            x[:, c] = _scale(x[:, c])

        color_lst = []

        def _color(value3):
            r = int(value3[0] * 255)
            g = int(value3[1] * 255)
            b = int(value3[2] * 255)
            return '#' + ('{:02X}' * 3).format(r, g, b)

        for value3 in x:
            color_lst.append(_color(value3))

        records['color'] = color_lst

    return pca.explained_variance_ratio_


def compute_angle_distance(records, angle_columns=angle_columns):
    '''
    Compute the angleDistance of the records,
    it is the distance measure in angles from the zero angles.

    The column of angleDistance will be added **IN-PLACE**

    Inputs:
        - records: The records of the simulation;
        - angle_columns: The column names of the angle values.

    Returns:
        - The records.
    '''

    d = records[angle_columns].to_numpy()
    angle_distances = np.linalg.norm(d, axis=1)
    records['angleDistance'] = angle_distances

    return records


# Simulation and compute pca decomposition
full_records = simulate(segments, count=10000)
compute_pca_columns(full_records)
compute_angle_distance(full_records)
full_records

# %%

# %%


def shrink_records(records, radius=2):
    '''
    Shrink the full_records

    Inputs:
        - records: The records of the simulation;
        - radius: The radius of the small range.

    Returns:
        - The shrinked records
    '''
    # Compute distance matrix
    d = records[['x', 'y', 'z']].to_numpy()
    n = len(d)
    coef_matrix = np.zeros((n, n))
    print(d.shape, coef_matrix.shape)
    for j in tqdm(range(n), 'Compute Coefficients'):
        coef_matrix[j] = np.linalg.norm(d - d[j], axis=1)
    np.fill_diagonal(coef_matrix, np.inf)

    # nearest_values = np.min(coef_matrix, axis=0)
    # pd_records['nearestValue'] = nearest_values
    # pd_records

    # Shrink the records
    records['selected'] = False

    ignores = []
    for j in tqdm(range(n), 'Shrink Nodes'):
        if j in ignores:
            continue

        coef_vec = coef_matrix[j]
        indexes = np.nonzero(coef_vec < radius)[0].tolist()

        if len(indexes) > 1:
            pd = records.loc[indexes]
            m = pd['angleDistance'].min()
            pd['selected'] = pd['angleDistance'] == m
            records.loc[pd.query(
                'angleDistance == {}'.format(m)).index, 'selected'] = True

        if len(indexes) == 1:
            records.loc[indexes[0], 'selected'] = True

        ignores += indexes

    print('Ignore {} nodes'.format(len(ignores)))
    new_records = records.query('selected == True').copy()

    return new_records


def shrink_records(records, count=1000):
    angles = records[angle_columns].to_numpy()
    print(angles.shape)

    xyz = records[['x', 'y', 'z']].to_numpy()
    print(xyz.shape)

    regressors = []
    _xyz = xyz * 0

    X = angles[:, (2, 4, 5)]

    for j in tqdm(range(3), 'Compute SVR'):
        regressor = svm.SVR()
        # regressor = linear_model.BayesianRidge(copy_X=True)
        regressor = linear_model.LinearRegression(copy_X=True, n_jobs=10)
        y = xyz[:, j]
        regressor.fit(X, y)
        p = regressor.predict(X)

        # regressor.fit(X, y, np.exp(-np.abs(y-p)))
        # p = regressor.predict(X)

        regressors.append(regressor)
        _xyz[:, j] = y - p

    values = np.linalg.norm(_xyz, axis=1)
    order = np.argsort(values)
    new_records = records.iloc[order[:count]]

    return new_records, regressors


# new_records = shrink_records(full_records)

new_records, regressors = shrink_records(full_records)
new_records


# %%
# Display the simulation

trace_kwargs = dict(marker=dict(size=2,
                                opacity=0.3,
                                line=None),
                    selector=dict(mode='markers'))

for col in ['pc0', 'pc1', 'pc2', 'angleDistance']:
    fig = px.scatter_3d(full_records,
                        x='x', y='y', z='z',
                        color=col, title=col)
    fig.update_traces(**trace_kwargs)
    fig.show()

# -------- %%
# Display the new_records
fig = px.scatter_3d(new_records, x='x', y='y',
                    z='z', title='Colored by PCs')
trace_kwargs['marker']['color'] = new_records['color']
fig.update_traces(**trace_kwargs)
fig.show()


# %%
# Analysis
_new_records = new_records.copy()

for col in angle_columns:
    _min, _max = angle_ranges[col]
    _new_records[col] = _new_records[col].map(
        lambda e: (e - _min) / (_max - _min))

barmode = 'overlay'  # 'relative'
barmode = 'relative'

for col in angle_columns:
    fig = px.histogram(_new_records, x=col, title=col, barmode=barmode)
    fig.show()

barmode = 'overlay'  # 'relative'
fig = px.histogram(_new_records, x=angle_columns, barmode=barmode)
fig.show()

# %%
vars = dict()
for col in angle_columns:
    d = _new_records[col]
    h, b = np.histogram(d)
    vars[col] = np.std(h)
    print(h)

print(vars)

# %%
new_records

# %%

records = new_records.copy()

angles = records[angle_columns].to_numpy()
print(angles.shape)

xyz = records[['x', 'y', 'z']].to_numpy()
print(xyz.shape)


X = angles[:, (2, 4, 5)]

for j in range(3):
    y = xyz[:, j]

    regressor = regressors[j]
    p = regressor.predict(X)

    order = np.argsort(y)
    _y = y[order]
    _p = p[order]

    fig = px.scatter(_y, title='xyz'[j])
    fig.add_trace(go.Scatter(y=_p, name='p'))
    fig.show()


# %%
