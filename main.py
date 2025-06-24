import math

import matplotlib.pyplot as plt

FUEL_DENSITY = 0.8  # kg/L
TANK_DRY_MASS_RATIO = 0.1  # Dry mass is 10% of the tank's capacity
STANDARD_ISP = 300.0  # s, specific impulse of the engine


def ln(x):
    """Natural logarithm function."""
    return math.log(x) if x > 0 else float('-inf')


class Tank:
    def __init__(self, capacity, separator_mass=1):
        self.capacity = capacity
        self.current_level = capacity
        self.mass = capacity * FUEL_DENSITY + capacity * TANK_DRY_MASS_RATIO + separator_mass
        self.empty = False

    def drain(self, amount):
        if amount > self.current_level:
            amount = self.current_level
            self.empty = True
        self.current_level -= amount
        self.mass -= amount * FUEL_DENSITY
        return amount


class Engine:
    def __init__(self, mass, isp=STANDARD_ISP):
        self.mass = mass
        self.isp = isp

    def thrust_step(self, ship_mass, fuel_burned_mass):
        delta_v = self.isp * 9.81 * ln(ship_mass / (ship_mass - fuel_burned_mass))
        return delta_v


class Rocket:
    def __init__(self, engine, tanks):
        self.engine = engine
        self.tanks = tanks
        self.velocity = 0

    @property
    def mass(self):
        return sum(tank.mass for tank in self.tanks) + self.engine.mass

    def stage(self):
        if self.tanks and self.tanks[0].empty:
            print(f"Staging {self.tanks[0].capacity} tank.")
            self.tanks.pop(0)

    def burn_step(self, fuel_per_step=1):
        self.stage()
        if not self.tanks:
            return False
        ship_mass = self.mass
        fuel_to_burn = self.tanks[0].drain(fuel_per_step)
        fuel_burned_mass = fuel_to_burn * FUEL_DENSITY
        delta_v = self.engine.thrust_step(ship_mass, fuel_burned_mass)
        self.velocity += delta_v
        return True


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
        print(f"Final velocity: {self.rocket.velocity:.2f} m/s")


def plot_results(masses, velocities):
    plt.plot(masses, velocities)
    plt.gca().invert_xaxis()
    plt.xlabel('Mass (kg)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Rocket Velocity vs Mass')
    plt.show()


def run_simulation():
    my_engine = Engine(10, STANDARD_ISP)

    tank1 = Tank(500, separator_mass=1)  # 1000 L tank
    tank2 = Tank(500, separator_mass=0)  # 1000 L tank
    my_rocket = Rocket(my_engine, [tank1, tank2])

    simulation = Simulation(my_rocket)
    simulation.run(drain_per_step=.1)
    masses = simulation.masses
    velocities = simulation.velocities
    plot_results(masses, velocities)


if __name__ == "__main__":
    run_simulation()
