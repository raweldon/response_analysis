import numpy as np

# Create data with x and y random in the [-2, 2] segment, and z a
# Gaussian function of x and y.
np.random.seed(12345)
x = 4 * (np.random.random(500) - 0.5)
y = 4 * (np.random.random(500) - 0.5)


def f(x, y):
    return np.exp(-(x ** 2 + y ** 2))

z = f(x, y)

from mayavi import mlab
mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

# Visualize the points
pts = mlab.points3d(x, y, z, z, scale_mode='none', scale_factor=0.2)

# Create and visualize the mesh
mesh = mlab.pipeline.delaunay2d(pts)
surf = mlab.pipeline.surface(mesh)

mlab.view(47, 57, 8.2, (0.1, 0.15, 0.14))
mlab.show()

## Another example

# Standard library imports
from os.path import join, abspath, dirname

# Mayavi imports.
from mayavi.scripts import mayavi2
from mayavi.sources.api import VTKXMLFileReader
from mayavi.filters.contour import Contour
from mayavi.filters.api import PolyDataNormals
from mayavi.filters.set_active_attribute import SetActiveAttribute
from mayavi.modules.api import Surface, Outline

@mayavi2.standalone
mayavi.new_scene()

# Read the example data: fire_ug.vtu.
r = VTKXMLFileReader()
filename = join(mayavi2.get_data_dir(dirname(abspath(__file__))),
                'fire_ug.vtu')
r.initialize(filename)
mayavi.add_source(r)
# Set the active point scalars to 'u'.
r.point_scalars_name = 'u'

# Simple outline for the data.
o = Outline()
mayavi.add_module(o)

# Branch the pipeline with a contour -- the outline above is
# directly attached to the source whereas the contour below is a
# filter and will branch the flow of data.   An isosurface in the
# 'u' data attribute is generated and normals generated for it.

c = Contour()
mayavi.add_filter(c)
n = PolyDataNormals()
mayavi.add_filter(n)

# Now we want to show the temperature 't' on the surface of the 'u'
# iso-contour.  This is easily done by using the SetActiveAttribute
# filter below.

aa = SetActiveAttribute()
mayavi.add_filter(aa)
aa.point_scalars_name = 't'

# Now view the iso-contours of 't' with a Surface filter.
s = Surface(enable_contours=True)
mayavi.add_module(s)