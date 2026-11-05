from esclab.simulate import Component, Model
from eeslib import fluid_properties
import numpy as np

class Fluid:
    # default fluid override with set_fluid 
    fluid = 'Water'
    FLUIDS = fluid_properties.get_fluids_list()

    def set_fluid(self, f: str):
        assert f in Fluid.FLUIDS, f'Fluid ({f}) not found in:\n{Fluid.FLUIDS}'
        self.fluid = f

# ---------------------------------------------------------------------

class Pump(Fluid):
    S = Component.Parameter(350)

    m_dot_in = Component.Input()
    T_in     = Component.Input()

    W_pump    = Component.Output()
    m_dot_out = Component.Output()
    T_out     = Component.Output()

    def calculate(self):
        # TODO: maybe make output setting functions to make more generic?
        # bigger TODO: maybe Output and Input become Port and I/O is determined by connections?

        # W_pump is a reported output, not a connected matrix unknown.
        self.W_pump = 1000.*(1+0.25*np.sin(self.model.time/50.))
        # Catch any equation values that haven't been initialized yet.
        if not np.isfinite(self.m_dot_out):
            self.m_dot_out = self.W_pump**0.5 / self.S
        
        if self.coupled_eqs is not None:
            # short name
            _src = self.coupled_eqs.source
            # Power and mass flow relationship
            self.coupled_eqs.add_equation({
                self.m_dot_out: 1,
            }, rhs=self.W_pump**0.5 / self.S)
            
            # energy conservation
            # T_out = T_in + W / (m_dot_out * cp(T_out))
            self.coupled_eqs.add_equation({
                self.T_out:       1,
                _src(self.T_in): -1,
            }, rhs=self.W_pump / (self.m_dot_out * self.cp)) # TODO: make this a call to a Fluid function 
        pass

# ---------------------------------------------------------------------

class Pipe(Fluid):
    D         = Component.Parameter(0.05)
    L         = Component.Parameter(10.)
    h_ext     = Component.Parameter(10.)
    T_ambient = Component.Parameter(25.)

    m_dot_in = Component.Input()
    T_in     = Component.Input()

    m_dot_out = Component.Output()
    T_out     = Component.Output()
    Q_loss    = Component.Output()

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
            # short name
            _src = self.coupled_eqs.source
            # mass conservation
            self.coupled_eqs.add_equation({
                self.m_dot_out:       1,
                _src(self.m_dot_in): -1,
            }, rhs=0.)
            
            # energy conservation
            # Positive Q_loss removes energy from the fluid.
            # T_out = T_in - Q_loss / (m_dot_out * cp(T_out))

            # Linearized method
            # shorthand coefficients
            m0 = self.m_dot_out
            Ti0 = self.T_in
            To0 = self.T_out
            # TODO: make this a call to a Fluid function
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
                self.m_dot_out:  drdm,
                _src(self.T_in): drdTi,
                self.T_out:      drdTo,
            }, rhs=drdm * m0 + drdTi * Ti0 + drdTo * To0 - R0)
        # ------

        # compute other outputs
        T_surface = (self.T_in + self.T_out) / 2.
        self.Q_loss = self.h_ext * self.A * (T_surface - self.T_ambient)
        
        return
    
# ---------------------------------------------------------------------

class TeeOut(Fluid):
    m_dot_in = Component.Input()
    T_in     = Component.Input()

    m_dot_out_1 = Component.Output()
    m_dot_out_2 = Component.Output()
    T_out       = Component.Output()
    
    def calculate(self):
        if self.coupled_eqs is not None:
            # short name
            _src = self.coupled_eqs.source
            # mass conservation
            self.coupled_eqs.add_equation({
                self.m_dot_out_1:     1,
                self.m_dot_out_2:     1,
                _src(self.m_dot_in): -1,
            }, rhs=0.)
            # T_out = T_in (assuming no heat loss in the tee)
            self.coupled_eqs.add_equation({
                self.T_out:       1,
                _src(self.T_in): -1,
            }, rhs=0.)
        pass

# ---------------------------------------------------------------------

class TeeReturn(Fluid):
    m_dot_in_1 = Component.Input()
    m_dot_in_2 = Component.Input()
    T_in_1     = Component.Input()
    T_in_2     = Component.Input()

    m_dot_out = Component.Output()
    T_out     = Component.Output()
    m_res     = Component.Output()
    E_res     = Component.Output()


    def calculate(self):
        if self.coupled_eqs is not None:
            # short name
            _src = self.coupled_eqs.source
            # mass conservation
            self.coupled_eqs.add_equation({
                self.m_dot_out:         1,
                _src(self.m_dot_in_1): -1,
                _src(self.m_dot_in_2): -1,
            }, rhs=0.)
            # energy conservation
            # m_dot_out*T_out = (m_dot_1*T_in_1 + m_dot_2*T_in_2) (assuming no heat loss in the tee)
            self.coupled_eqs.add_equation({
                self.T_out:         self.m_dot_out,
                _src(self.T_in_1): -self.m_dot_in_1,
                _src(self.T_in_2): -self.m_dot_in_2,
            }, rhs=0.)

            # Control choice: relate outlet temperatures from each branch
            self.coupled_eqs.add_equation({
                _src(self.T_in_1):  1,
                _src(self.T_in_2): -1,
            }, rhs=2.)

        # Report residuals
        self.m_res = (
            self.m_dot_out - self.m_dot_in_1 - self.m_dot_in_2
        )

        # TODO: make this a call to a Fluid function
        self.E_res = self.cp * (
            self.m_dot_out * self.T_out
            - self.m_dot_in_1 * self.T_in_1
            - self.m_dot_in_2 * self.T_in_2
        )
        return

# ---------------------------------------------------------------------

class Tank(Fluid):
    """
    Model the temperature and fill volume of a tank storing an incompressible fluid.
    """
    # Parameters 
    m_init = Component.Parameter(40.)  # kg
    T_init = Component.Parameter(400)  # K
    # Inputs
    T_in      = Component.Input()
    m_dot_in  = Component.Input()
    m_dot_out = Component.Input()
    # Outputs
    dT      = Component.Output()
    dm      = Component.Output()
    T_out   = Component.Output()
    m_final = Component.Output()  # report the current mass of the tank

    def presim_setup(self, **kwargs):
        # We need to keep track of the component state information -- mass and temperature
        # Before the simulation starts, set the current state of mass and temperature to equal
        # the initial values specified in the parameters
        self.m = self.m_init 
        self.T = self.T_init

        # Add defaults to outputs to avoid nan's
        self.dT.v      = 0.
        self.dm.v      = 0.
        self.T_out.v   = self.T_init
        self.m_final.v = self.m_init
        
    def calculate(self):
        """
        Solve for the new mass using a mass balance on the tank.
        Mass balance:
            IN = OUT + STORED
            m_dot_in = m_dot_out + dm/dt

        Approximating the derivative at the current timestep:
            m_dot_in = m_dot_out + m/dt - m0/dt

        Reorganizing:
            m_dot_in - m_dot_out - m/dt = m0/dt
            
        Solve for the new temperature using an energy balance on the tank.
        Energy balance: 
            IN = OUT + STORED
            m_dot_in*h_in = m_dot_out*h_out + dU/dt
        
        Assuming incompressible fluid, h = cp*T, and U = m*cp*T, we can write:
            T_in*m_dot_in*cp = T*m_dot_out*cp + cp*d(m*T)/dt 

            R(x)  = T*m_dot_out*cp - T_in*m_dot_in*cp + cp*(m*dT/dt + T*dm/dt)
            x     = [m_dot_out, T_in, m_dot_in, T,  m,  dT/dt, dm/dt]
            x0    = [mo0,       Ti0,  mi0,      T0, m0, dT0,   dm0]
            R(x0) = T0*mo0*cp - Ti0*mi0*cp + cp*(m0*dT0 + T0*dm0)
            
            R(x0) + dR/dmo(mo-mo0) + dR/dTi(Ti-Ti0) + dR/dmi(mi-mi0) 
                + dR/dT(T-T0) + dR/dm(m-m0) + dR/ddT(dT-dT0) + dR/ddm(dm-dm0) = 0
        
        Reorganizing:
            dR/dmo*mo + dR/dTi*Ti + dR/dmi*mi + dR/dT*T + dR/dm*m + dR/ddT*dT + dR/ddm*dm
            = -R(x0) + dR/dmo*mo0 + dR/dTi*Ti0 + dR/dmi*mi0 + dR/dT*T0 + dR/dm*m0 + dR/ddT*dT0 + dR/ddm*dm0
            NOTE: can divide through by cp

        Approximating derivatives at x0:
            dR/dmo = T0
            dR/dTi = -mi0
            dR/dmi = -Ti0
            dR/dT  = mo0 + dm0
            dR/dm  = dT0
            dR/ddT = m0
            dR/ddm = T0
        """

        if self.coupled_eqs is not None:
            # short name
            _src = self.coupled_eqs.source
            # Get the timestep
            dt = self.model.settings.timestep
            # short names
            m0  = self.m.v
            T0  = self.T.v
            dm0 = self.m_final.v - self.m.v
            dT0 = self.T_out.v - self.T.v
            mo0 = self.m_dot_out.v
            mi0 = self.m_dot_in.v
            Ti0 = self.T_in.v

            # mass balance
            # m_dot_in - m_dot_out - m/dt = m0/dt
            self.coupled_eqs.add_equation({
                _src(self.m_dot_in):   1,
                _src(self.m_dot_out): -1,
                self.m_final:         -1/dt,
            }, rhs = m0/dt)

            dR_dmo = T0
            dR_dTi = -mi0
            dR_dmi = -Ti0
            dR_dT  = mo0 + dm0
            dR_dm  = dT0
            dR_ddT = m0
            dR_ddm = T0

            # R(x0) = T0*mo0*cp - Ti0*mi0*cp + cp*(m0*dT0 + T0*dm0)
            R0 = T0*mo0 - Ti0*mi0 + m0*dT0 + T0*dm0

            # energy balance
            # dR/dmo*mo + dR/dTi*Ti + dR/dmi*mi + dR/dT*T + dR/dm*m + dR/ddT*dT + dR/ddm*dm
            # = -R(x0) + dR/dmo*mo0 + dR/dTi*Ti0 + dR/dmi*mi0 + dR/dT*T0 + dR/dm*m0 + dR/ddT*dT0 + dR/ddm*dm0
            self.coupled_eqs.add_equation({
                _src(self.m_dot_out): dR_dmo,
                _src(self.T_in):      dR_dTi,
                _src(self.m_dot_in):  dR_dmi,
                self.T_out:           dR_dT,
                self.m_final:         dR_dm,
                self.dT:              dR_ddT,
                self.dm:              dR_ddm,
            }, rhs = -R0 + dR_dmo*mo0 + dR_dTi*Ti0 + dR_dmi*mi0 + dR_dT*T0 + dR_dm*m0 + dR_ddT*dT0 + dR_ddm*dm0)


        # # Calculate the change in mass in the tank (dm) over the timestep
        # dm = (self.m_dot_in - self.m_dot_out)*dt
        # m_new = self.m + dm

        # # Calculate the temperature of the tank at the end of the timestep
        # T_new = ((self.T_in*self.m_dot_in - self.T*self.m_dot_out)*dt + self.m*self.T)/m_new  

        # # Assign the output temperature to the value that was calculated 
        # # for the end of the timestep. This is the temperature that will be reported 
        # # to other components in the model.
        # self.T_out = T_new
        # # Assign output mass
        # self.m_final = m_new

        # Update the state of the tank for the next timestep
        if self.model.is_converged:
            # Only execute if the model has converged for this timestep
            self.T = self.T_out
            self.m = self.m_final
        return

# ---------------------------------------------------------------------

class Turbine(Fluid):
    eta_s    = Component.Parameter(0.9)
    PR       = Component.Parameter(4.0)
    mdot_in  = Component.Input()
    T_in     = Component.Input()
    P_in     = Component.Input()
    h_in     = Component.Output() # added for use in network equation of energy balance
    h_out    = Component.Output()
    T_out    = Component.Output()
    P_out    = Component.Output()
    mdot_out = Component.Output()
    W        = Component.Output()

    def calculate(self):
        """
        Calculate the turbine outlet conditions and work output based on the inlet conditions and isentropic efficiency.

        Mass balance: 
            mdot_out - mdot_in = 0
            mdot_out = mdot_in = m

        Efficiency relationship:
            h_out = h_in - (h_in-h_out_s)*eta_s
            h_out = (1 - eta_s) * h_in + eta_s * h_out_s
            h_out - (1 - eta_s) * h_in = eta_s * h_out_s

        Energy balance:
            W = m*(h_in - h_out)

            R(x) = W - m*(h_in - h_out)
            x    = [W,  m,  h_in, h_out]
            x0   = [W0, m0, hi0,  ho0]
            R(x0) = W0 - m0*(hi0 - ho0)

            R(x0) + dR/dW(W-W0) + dR/dm(m-m0) + dR/dhi(hi-hi0) + dR/dho(ho-ho0) = 0

        Reorganizing:
            dR/dW*W + dR/dm*m + dR/dhi*hi + dR/dho*ho = -R(x0) + dR/dW*W0 + dR/dm*m0 + dR/dhi*hi0 + dR/dho*ho0

        Approximating derivatives at x0:
            dR/dW  = 1
            dR/dm  = -(hi0 - ho0)
            dR/dhi = -m0
            dR/dho = m0
        """

        # determine inlet and isentropic properties
        self.P_out = self.P_in / self.model.design.PR
        self.h_in  = fluid_properties.enthalpy(self.fluid, T=self.T_in, P=self.P_in)
        s_in       = fluid_properties.entropy(self.fluid, T=self.T_in, P=self.P_in)
        h_out_s    = fluid_properties.enthalpy(self.fluid, P=self.P_out, s=s_in)

        if self.coupled_eqs is not None:
            # short name
            _src = self.coupled_eqs.source

            # mass conservation
            self.coupled_eqs.add_equation({
                self.m_dot_out: 1,
                _src(self.m_dot_in): -1,
            }, rhs=0.)

            # efficency relationship
            self.coupled_eqs.add_equation({
                self.h_out: 1,
                self.h_in: -(1 - self.eta_s),
            }, rhs=self.eta_s * h_out_s)

            # energy conservation
            # short names
            m0  = self.mdot_in.v
            hi0 = self.h_in.v
            ho0 = self.h_out.v
            W0  = self.W.v

            # R(x0) = W0 - m0*(hi0 - ho0)
            R0 = W0 - m0*(hi0 - ho0)

            # Derivatives of the residual.
            dR_dW  = 1
            dR_dm  = -(hi0 - ho0)
            dR_dhi = -m0
            dR_dho = m0

            self.coupled_eqs.add_equation({
                self.W:     dR_dW,
                self.m_dot: dR_dm,
                self.h_in:  dR_dhi,
                self.h_out: dR_dho
            }, rhs=-R0 + dR_dW*W0 + dR_dm*m0 + dR_dhi*hi0 + dR_dho*ho0)

        # compute other outputs
        self.T_out = fluid_properties.temperature(self.fluid, P=self.P_out, h=self.h_out)

        # self.mdot_out = self.mdot_in
        # h_in = fluid_properties.enthalpy(self.fluid, T=self.T_in, P=self.P_in)
        # s_in = fluid_properties.entropy(self.fluid, T=self.T_in, P=self.P_in)
        # self.P_out = self.P_in / self.model.design.PR
        # h_out_s = fluid_properties.enthalpy(self.fluid,P=self.P_out, s=s_in)
        # self.h_out = h_in - (h_in-h_out_s)*self.eta_s
        # self.T_out = fluid_properties.temperature(self.fluid, P=self.P_out, h=self.h_out)
        # self.W = (h_in - self.h_out)*self.mdot_in        
        return 

# ---------------------------------------------------------------------

class HeatExchanger:
    UA = Component.Parameter(1000.)
    N  = Component.Parameter(10.) # number of passes

    P_in_shell  = Component.Input()
    P_in_tube   = Component.Input()
    T_in_shell  = Component.Input()
    T_in_tube   = Component.Input()
    m_dot_shell = Component.Input()
    m_dot_tube  = Component.Input()

    T_out_shell = Component.Output()
    T_out_tube  = Component.Output()
    q           = Component.Output()

    def presim_setup(self, **kwargs):
        self.shell = Fluid()
        self.tube  = Fluid()

    def calculate(self):
        # determine effectiveness
        # frac_1 = 1 + exp(-NTU_1 * sqrt(1 + C_r^2)) / 1 - exp(-NTU_1 * sqrt(1 + C_r^2))
        # epsilon_1 = 2 * (1 + C_r * sqrt(1 + C_r^2) * frac_1) ^ -1
        # frac_2 = (1 - epsilon_1 * C_r) / (1 - epsilon_1)
        # epsilon = (frac_2 ^ N - 1) / (frac_2 ^ N - C_r)

        # determine outputs
        # q = epsilon * Cmin * (T_hot_in - T_cold_in)
        # T_out_hot = T_hot_in - q / (C_hot)
        pass

# ---------------------------------------------------------------------