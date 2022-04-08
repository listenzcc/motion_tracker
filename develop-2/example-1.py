# %%
import pandas as pd
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


shoulder, upper_arm, lower_arm = mk_segments()

# %%
print(shoulder.get_axes_name())

displayer = SegmentDisplayer()
fig = displayer.plot(shoulder)

upper_arm.rotate(shoulder.get_axes_by_name('0', force_single=True), 20)
displayer.plot(shoulder, fig)

upper_arm.rotate(shoulder.get_axes_by_name('0', force_single=True), 20)
displayer.plot(shoulder, fig)

fig.show()

# %%
