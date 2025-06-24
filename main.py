import math

import matplotlib.pyplot as plt
import numpy as np

TANK_DRY_MASS_RATIO = 0.05
STANDARD_ISP = 300.0  # s, specific impulse of the engine
SHIP_DRY_MASS = 100.0  # kg, dry mass of the ship without fuel
SEPARATOR_MASS = 7  # kg, mass of the separator for each tank
TOTAL_SHIP_MASS = 1000.0  # kg, total mass of the ship


def ln(x):
    """Natural logarithm function."""
    return math.log(x) if x > 0 else float('-inf')


class Tank:
    def __init__(self, capacity, stageable=True, separator_mass=SEPARATOR_MASS):
        self.capacity = capacity
        self.current_level = capacity
        self.stageable = bool(stageable)
        self.mass = capacity + capacity * TANK_DRY_MASS_RATIO + separator_mass
        self.empty = False

    def drain(self, amount):
        if amount > self.current_level:
            amount = self.current_level
            self.empty = True
        self.current_level -= amount
        self.mass -= amount
        return amount

    def __repr__(self):
        return f"Tank(capacity={self.capacity}, current_level={self.current_level}, mass={self.mass})"


class Engine:
    def __init__(self, mass, isp=STANDARD_ISP):
        self.mass = mass
        self.isp = isp

    def thrust_step(self, ship_mass, fuel_burned_mass):
        delta_v = self.isp * 9.81 * ln(ship_mass / (ship_mass - fuel_burned_mass))
        return delta_v

    def __repr__(self):
        return f"Engine(mass={self.mass}, isp={self.isp})"


class Rocket:
    def __init__(self, engine, tanks):
        self.engine = engine
        self.tanks = tanks
        self.velocity = 0

    @property
    def mass(self):
        return sum(tank.mass for tank in self.tanks) + self.engine.mass

    def drain(self, amount):
        total_drain = 0
        for tank in self.tanks:
            if tank.current_level > 0:
                drained = tank.drain(amount - total_drain)
                total_drain += drained
                if total_drain >= amount:
                    break
        return total_drain

    def stage(self):
        for tank in self.tanks:
            if tank.empty and tank.stageable:
                self.tanks.remove(tank)
                break

    def burn_step(self, fuel_per_step=1):
        self.stage()
        ship_mass = self.mass
        fuel_to_burn = self.drain(fuel_per_step)
        if fuel_to_burn <= 0:
            return False
        fuel_burned_mass = fuel_to_burn
        delta_v = self.engine.thrust_step(ship_mass, fuel_burned_mass)
        self.velocity += delta_v
        return True

    def __repr__(self):
        return f"Rocket(mass={self.mass}, velocity={self.velocity}, tanks={self.tanks})"


class Simulation:
    def __init__(self, rocket):
        self.rocket = rocket
        self.velocities = []
        self.fuels = []
        self.masses = []

    def run(self, drain_per_step=1.0):
        while self.rocket.burn_step(drain_per_step):
            self.velocities.append(self.rocket.velocity)
            self.fuels.append(sum([tank.current_level for tank in self.rocket.tanks]))
            self.masses.append(self.rocket.mass)

    def __repr__(self):
        return f"Simulation(rocket={self.rocket}, velocities={self.velocities}, fuels={self.fuels}, masses={self.masses})"


def plot_results(data):
    for d in data:
        masses, velocities = d
        plt.plot(masses, velocities)
    plt.gca().invert_xaxis()
    plt.xlabel('Mass (kg)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Rocket Velocity vs Mass')
    plt.show()


def run_simulation(tanks):
    my_engine = Engine(SHIP_DRY_MASS, STANDARD_ISP)

    my_rocket = Rocket(my_engine, tanks)

    simulation = Simulation(my_rocket)
    simulation.run(drain_per_step=.1)

    masses = simulation.masses
    velocities = simulation.velocities
    return masses, velocities, my_rocket


def run_and_plot():
    tanks = [
        Tank(capacity=400, stageable=True),
        Tank(capacity=300, stageable=True),
        Tank(capacity=200, stageable=False, separator_mass=0)
    ]  # Create multiple tanks based on num_tanks
    masses, velocities, rocket = run_simulation(tanks)
    tanks = [
        Tank(capacity=300, stageable=True),
        Tank(capacity=300, stageable=True),
        Tank(capacity=300, stageable=False, separator_mass=0)
    ]  # Create multiple tanks based on num_tanks
    masses2, velocities2, rocket2 = run_simulation(tanks)
    print(rocket)
    plot_results([[masses, velocities], [masses2, velocities2]])


def minimize_test():
    def objective(x):
        tanks = []
        for var in x:
            tanks.append(Tank(capacity=var, stageable=True))
        tanks[-1].stageable = False
        masses, velocities, rocket = run_simulation(tanks)
        print(f"Stepping with tanks: {x}: {-rocket.velocity:.2f} m/s")
        return -rocket.velocity

    def mass_constraint(x):
        return 900 - sum(x)

    import scipy.optimize as opt

    num_tanks = 2
    initial_guess = np.array([10.0] * num_tanks)
    bounds = [(0.0, 900)] * num_tanks  # Each tank must have a minimum capacity of 10.0

    result = opt.minimize(objective,
                          x0=initial_guess,
                          bounds=bounds,
                          constraints={'type': 'ineq', 'fun': mass_constraint},
                          method='SLSQP')
    print(f"Minimum at x = {result.x}, f(x) = {result.fun}")


if __name__ == "__main__":
    minimize_test()
