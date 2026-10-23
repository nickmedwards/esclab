from esclab.simulate import Component, Model
from eeslib import fluid_properties
import numpy as np

class HasFluid(Component):
    # default fluid override with set_fluid 
    fluid = 'Water'

    def set_fluid(self, f: str):
        if f not in fluid_properties.get_fluids_list():
            raise ValueError(f'Fluid ({f}) not found in:\n{fluid_properties.get_fluids_list()}')
        self.fluid = f

class Pump(HasFluid):
    S = Component.Parameter(350)

    m_dot_in = Component.Input()
    T_in = Component.Input()

    W_pump = Component.Output()
    m_dot_out = Component.Output()
    T_out = Component.Output()


    def calculate(self):
        # W_pump is a reported output, not a connected matrix unknown.
        self.W_pump = 1000.*(1+0.25*np.sin(self.model.time/50.))
        # Catch any equation values that haven't been initialized yet.
        if not np.isfinite(self.m_dot_out):
            self.m_dot_out = self.W_pump**0.5 / self.S
        
        if self.coupled_eqs is not None:

            # Power and mass flow relationship
            self.coupled_eqs.add_equation({
                    self.m_dot_out: 1,
                }, rhs=self.W_pump**0.5 / self.S)
            
            # energy conservation
            # T_out = T_in + W / (m_dot_out * cp(T_out))
            self.coupled_eqs.add_equation({
                    self.T_out: 1,
                    self.coupled_eqs.source(self.T_in): -1,
                }, rhs=self.W_pump / (self.m_dot_out * self.cp))
        pass