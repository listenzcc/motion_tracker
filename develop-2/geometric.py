# %%
import numpy as np
from scipy.spatial.transform import Rotation as R

# %%
_dtype = np.float32

# %%


def _iterable(x):
    return hasattr(x, '__iter__')


def _canBePtr3(x):
    if _iterable(x):
        return len(x) == 3
    return False


def _deg2rad(deg):
    return deg / 180 * np.pi


def _regular(x, y='N.A.', z='N.A.'):
    if _canBePtr3(x):
        x, y, z = x
    v = np.array([x, y, z], dtype=_dtype)
    return v


def _normalize(x, y='N.A.', z='N.A.'):
    v = _regular(x, y, z)
    return v / np.linalg.norm(v)


# %%

class Point(object):
    '''
    Class of Point
    '''

    def __init__(self, x, y=0, z=0):
        '''
        Init the point by x, y, z or x=(x, y, z)
        '''
        self.xyz = _regular(x, y, z)
        pass

    @property
    def xyz(self):
        '''
        Get (x, y, z) of the point
        '''
        return np.array([self.x, self.y, self.z], dtype=_dtype)

    @xyz.setter
    def xyz(self, x_in_3):
        '''
        Set (x, y, z) of the point as x_in_3,
        it is the 3 elements tuple of the point coordinates.
        '''
        x, y, z = _regular(x_in_3)
        self.x = x
        self.y = y
        self.z = z
        pass

    def translate_by(self, dx, dy=0, dz=0):
        '''
        Translate the point by dx, dy, dz or dx=(dx, dy, dz)
        '''
        dx, dy, dz = _regular(dx, dy, dz)
        self.x += dx
        self.y += dy
        self.z += dz
        return self

    def translate_to(self, x, y=0, z=0):
        '''
        Translate the point to x, y, z or x=(x, y, z)
        '''
        self.xyz(_regular(x, y, z))
        return self


# %%


class Vector(object):
    '''
    Class of Vector
    '''

    def __init__(self, orig, dest):
        '''
        Init the vector by the orig and dest points
        '''
        self.orig = Point(orig)
        self.dest = Point(dest)

        self.relative = 'Auto set'

        pass

    @property
    def relative(self):
        '''
        Get the relative position of the dest from the orig
        '''
        return self.__relative

    @relative.setter
    def relative(self, _):
        '''
        Set the relative position of the dest from the orig.
        It needs nothing for input.
        '''
        self.__relative = self.dest.xyz - self.orig.xyz
        self.__normalized = _normalize(self.__relative)
        self.__norm = np.linalg.norm(self.__relative)
        pass

    @property
    def normalized(self):
        '''
        Get the normalization of the vector with the unit length
        '''
        return self.__normalized

    @property
    def norm(self):
        '''
        Get the norm of the vector,
        in 3D space,
        it is the length of the vector.
        '''
        return self.__norm

    def rotate(self, vec, deg, orig=None):
        '''
        Rotate the vector by the deg (in degrees),
        around the vec as the axis.

        The rotation center is orig point.
        If not provided, the rotation center uses the orig of the vector.
        '''
        vec = _regular(vec)
        rad = _deg2rad(deg)
        r = R.from_rotvec(vec * rad)

        if orig is None:
            # Rotate from the self's orig
            self.dest = self.orig + r.apply(self.dest - self.orig)
            self.relative = None
        else:
            # Rotate from the given orig
            orig = _regular(orig)
            self.orig = orig + r.apply(self.orig - orig)
            self.dest = orig + r.apply(self.dest - orig)

        return self

    def translate_by(self, dx, dy=0, dz=0):
        '''
        Translate the vector by dx, dy, dz or dx=(dx, dy, dz)
        '''
        self.orig.translate_by(dx, dy, dz)
        self.dest.translate_by(dx, dy, dz)
        return self

    def translate_to(self, x, y=0, z=0):
        '''
        Translate the orig point to x, y, z and x=(x, y, z),
        the dest point is tranlated accordingly.
        '''
        d = _regular(x, y, z) - self.orig.xyz
        self.orig.translate_by(d)
        self.dest.translate_by(d)
        return self

    def product(self, vec, normalize=False):
        '''
        Get the production between the vector and the given vec.

        If normalize: Both the vectors are normalized to unit length.
        '''
        assert isinstance(
            vec, Vector), 'Invalid production for {}'.format(type(vec))
        if normalize:
            prod = np.prod(self.norm, vec.norm)
        else:
            prod = np.prod(self.relative, vec.relative)
        return prod


# %%
