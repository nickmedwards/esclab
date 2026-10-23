"""
Solar receiver component library.
"""

from esclab.simulate import *
from eeslib import fluid_properties

class PlateReceiver(Component):
    fluid = 'Water'

    N = Component.Parameter(3.0) # [-], number of nodes in either discretization direction 

    H = Component.Parameter(2.0) # [m], in z direction in SolTrace
    W = Component.Parameter(2.0) # [m], in x direction in SolTrace
    T_max = Component.Parameter(95.0 + 273.15) # [K], maximum water temperature
    T_amb = Component.Parameter(20.0 + 273.15) # [K], ambient temperature

    def __init__(self):
        super().__init__()
        # make Ts, Ti, and Tm output matrices
        for _x in range(0, int(self.N)):
            for _z in range(0, int(self.N)):
                idx_str = f'_{_x}{_z}'
                default_T = float(self.T_max + self.T_amb) / 2
                object.__setattr__(self, f'Ts{idx_str}', Component.Output())

    def presim_setup(self, **kwargs):
        # depth of duct [m] set to make ensure thermally developed by 1% of H
        # in y direction in SolTrace
        # .01 * H = .05 * Re_DH * Pr * DH
        # enforing mass flow thru receiver such that Re_DH = 2000
        # DH = 2 * a * W / (a + W)
        # a = H * W / (5 * 2000 * Pr * 2 * W - H)
        Pr = fluid_properties.prandtl(self.fluid, T = float(self.T_max + self.T_amb) / 2, X = 1.0)
        self.a = float(self.H * self.W) / (20000 * Pr * self.W - self.H)

        # nodal distances 
        self.DELTAx = self.W / self.N
        self.DELTAz = self.H / self.N

        # cross sectional areas
        self.Axy = self.DELTAx * self.a
        self.Axz = self.DELTAx * self.DELTAz
        self.Ayz = self.a * self.DELTAz

        