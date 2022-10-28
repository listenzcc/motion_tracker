#! python
# name: app.py

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

# %%
import pandas as pd
import plotly.express as px

from dash import Dash, html, dcc
from segment import Segment, SegmentDisplayer

# %%
app = Dash(__name__)

# %%

''' md
Direction defines:
- x, 0, left
- y, 1, up
- z, 2, front
'''

angle_ranges = dict(
    a00=(-90, 45, 'shoulder', '0'),
    a01=(-60, 10, 'shoulder', '1'),
    a02=(0, 145, 'shoulder', '2'),
    a10=(-145, 0, 'upperArm', '0'),
    a12=(-20, 20, 'upperArm', '2'),
    a22=(-90, 90, 'lowerArm', '0'),
)

angle_columns = [e for e in angle_ranges]


def mk_segments():
    # Shoulder
    shoulder = Segment([0, 40, 0], [10, 40, 0], name='shoulder')

    shoulder.append_axis([shoulder.body.dest,
                          shoulder.body.dest + [1, 0, 0]],
                         name='0', deg=0, deg_limit=angle_ranges['a00'][:2])

    shoulder.append_axis([shoulder.body.dest,
                          shoulder.body.dest + [0, 1, 0]],
                         name='1', deg=0, deg_limit=angle_ranges['a01'][:2])

    shoulder.append_axis([shoulder.body.dest,
                          shoulder.body.dest + [0, 0, 1]],
                         name='2', deg=0, deg_limit=angle_ranges['a02'][:2])

    # Upper Arm
    upper_arm = Segment(shoulder.body.dest,
                        shoulder.body.dest + [0, -20, 0], name='upperArm')

    upper_arm.append_axis([upper_arm.body.dest,
                           upper_arm.body.dest + [1, 0, 0]],
                          name='0', deg=0, deg_limit=angle_ranges['a10'][:2])

    upper_arm.append_axis([upper_arm.body.dest,
                           upper_arm.body.dest + [0, 0, 1]],
                          name='2', deg=0, deg_limit=angle_ranges['a12'][:2])

    # Lower Arm
    lower_arm = Segment(upper_arm.body.dest,
                        upper_arm.body.dest + [0, -20, 0], name='lowerArm')

    lower_arm.append_axis([lower_arm.body.dest,
                           lower_arm.body.dest + [0, 0, 1]],
                          name='2', deg=0, deg_limit=angle_ranges['a22'][:2])

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

# %%
# # assume you have a "long-form" data frame
# # see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

# %%

# %%
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    html.Div(
        dcc.Slider(id='slider-1', min=0, max=20, value=10,)
    ),

    dcc.Graph(
        id='main-graph',
        figure=fig
    )
])

# %%
if __name__ == '__main__':
    app.run_server(debug=True)

# %%
