from esclab.simulate import Component, Model
from eeslib import fluid_properties
import numpy as np

class HasFluid(Component):
    # default fluid override with set_fluid 
    fluid = 'Water'
    FLUIDS = fluid_properties.get_fluids_list()

    def set_fluid(self, f: str):
        assert f in HasFluid.FLUIDS, f'Fluid ({f}) not found in:\n{HasFluid.FLUIDS}'
        self.fluid = f

class Pump(HasFluid):
    S = Component.Parameter(350)

    m_dot_in = Component.Input()
    T_in = Component.Input()

    W_pump = Component.Output()
    m_dot_out = Component.Output()
    T_out = Component.Output()

    def calculate(self):
        # TODO: maybe make output setting functions to make more generic?
        # bigger TODO: maybe Output and Input become Port and I/O is determined by connections?

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
                }, rhs=self.W_pump / (self.m_dot_out * self.cp)) # TODO: make this a call to a HasFluid function 
        pass

class Pipe(HasFluid):
    D = Component.Parameter(0.05)
    L = Component.Parameter(10.)
    h_ext = Component.Parameter(10.)
    T_ambient = Component.Parameter(25.)

    m_dot_in = Component.Input()
    T_in = Component.Input()

    m_dot_out = Component.Output()
    T_out = Component.Output()
    Q_loss = Component.Output()

    def presim_setup(self, **kwargs):
        self.A = self.pi * self.D * self.L
        self.m_dot_ref = 10.

    def calculate(self):
        # Heat loss 
        # Q_loss = h_ext * A * (T_surface - T_ambient)
        # T_surface = (T_in + T_out) / 2   << T_out??
        
        # handle un-initialized equation values
        if not np.isfinite(self.T_out):
            self.T_out = self.T_in-10.
        if not np.isfinite(self.m_dot_out):
            self.m_dot_out = self.m_dot_in

        if self.coupled_eqs is not None:
            # mass conservation
            self.coupled_eqs.add_equation({
                    self.m_dot_out: 1,
                    self.coupled_eqs.source(self.m_dot_in): -1,
                }, rhs=0.)
            
            # energy conservation
            # Positive Q_loss removes energy from the fluid.
            # T_out = T_in - Q_loss / (m_dot_out * cp(T_out))

            # Linearized method
            # shorthand coefficients
            m0 = self.m_dot_out
            Ti0 = self.T_in
            To0 = self.T_out
            # TODO: make this a call to a HasFluid function
            cp = self.cp
            H = self.h_ext * self.A

            # Energy imbalance at the current guesses.
            R0 = m0 * cp * (To0 - Ti0) + H * (
                (Ti0 + To0) / 2.0 - self.T_ambient
            )

            # Derivatives of the residual.
            drdm = cp * (To0 - Ti0)
            drdTi = -m0 * cp + H / 2.0
            drdTo =  m0 * cp + H / 2.0

            self.coupled_eqs.add_equation({
                self.m_dot_out: drdm,
                self.coupled_eqs.source(self.T_in): drdTi,
                self.T_out: drdTo,
            }, rhs=drdm * m0 + drdTi * Ti0 + drdTo * To0 - R0)
        # ------

        # compute other outputs
        T_surface = (self.T_in + self.T_out) / 2.
        self.Q_loss = self.h_ext * self.A * (T_surface - self.T_ambient)
        
        return

class TeeOut(HasFluid):
    m_dot_in = Component.Input()
    T_in = Component.Input()

    m_dot_out_1 = Component.Output()
    m_dot_out_2 = Component.Output()
    T_out = Component.Output()
    
    def calculate(self):
        if self.coupled_eqs is not None:
            # mass conservation
            self.coupled_eqs.add_equation({
                    self.m_dot_out_1: 1,
                    self.m_dot_out_2: 1,
                    self.coupled_eqs.source(self.m_dot_in): -1,
                }, rhs=0.)
            # T_out = T_in (assuming no heat loss in the tee)
            self.coupled_eqs.add_equation({
                    self.T_out: 1,
                    self.coupled_eqs.source(self.T_in): -1,
                }, rhs=0.)
        pass
# ---------------------------------------------------------------------

class TeeReturn(HasFluid):
    m_dot_in_1 = Component.Input()
    m_dot_in_2 = Component.Input()
    T_in_1 = Component.Input()
    T_in_2 = Component.Input()

    m_dot_out = Component.Output()
    T_out = Component.Output()
    m_res = Component.Output()
    E_res = Component.Output()


    def calculate(self):
        if self.coupled_eqs is not None:

            # mass conservation
            self.coupled_eqs.add_equation({
                    self.m_dot_out: 1,
                    self.coupled_eqs.source(self.m_dot_in_1): -1,
                    self.coupled_eqs.source(self.m_dot_in_2): -1,
                }, rhs=0.)
            # energy conservation
            # m_dot_out*T_out = (m_dot_1*T_in_1 + m_dot_2*T_in_2) (assuming no heat loss in the tee)
            self.coupled_eqs.add_equation({
                    self.T_out: self.m_dot_out,
                    self.coupled_eqs.source(self.T_in_1): -self.m_dot_in_1,
                    self.coupled_eqs.source(self.T_in_2): -self.m_dot_in_2,
                }, rhs=0.)

            # Control choice: relate outlet temperatures from each branch
            self.coupled_eqs.add_equation({
                    self.coupled_eqs.source(self.T_in_1): 1,
                    self.coupled_eqs.source(self.T_in_2): -1,
                }, rhs=2.)

        # Report residuals
        self.m_res = (
            self.m_dot_out - self.m_dot_in_1 - self.m_dot_in_2
        )

        # TODO: make this a call to a HasFluid function
        self.E_res = self.cp * (
            self.m_dot_out * self.T_out
            - self.m_dot_in_1 * self.T_in_1
            - self.m_dot_in_2 * self.T_in_2
        )
        return