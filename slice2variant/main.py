from var2 import sympy_part
from var2 import numpy_part
from var2 import matplotlib_part

sympy_part()

t, x, v, envelope = numpy_part()

matplotlib_part(t, x, v, envelope)