import mesa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace

class Pedestrian(Agent):
    def __init__(self, unique_id, model, pos, goal, speed, radius):
        super().__init__(unique_id, model)
        self.pos = pos
        self.vel = np.zeros(2)
        self.goal = goal
        self.speed = speed
        self.radius = radius

    def step(self):
        self.update_velocity()
        self.update_position()

    def update_velocity(self):
        pass  # TODO: Implement the social force model rules to update the velocity of the agent

    def update_position(self):
        new_pos = self.pos + self.vel
        self.model.space.move_agent(self, new_pos)
        self.pos = new_pos

class TrainStationModel(Model):
    def __init__(self, num_agents, width, height, exit_pos):
        self.num_agents = num_agents
        self.width = width
        self.height = height
        self.exit_pos = exit_pos
        self.space = ContinuousSpace(width, height, True)
        self.schedule = RandomActivation(self)
        self.datacollector = mesa.DataCollector(
            {
                "Speed": lambda a: np.linalg.norm(a.vel),
                "Density": lambda m: m.num_agents / (m.width * m.height),
                "Flow": lambda m: m.datacollector.get_agent_vars_dataframe()["Exited"].sum() / self.schedule.time
            }
        )
        self.agents = []

        # Create agents
        for i in range(self.num_agents):
            pos = np.random.uniform(low=[0, 0], high=[self.width, self.height])
            goal = np.random.uniform(low=[0, 0], high=[self.width, self.height])
            speed = np.random.normal(loc=1.3, scale=0.3)
            radius = np.random.normal(loc=0.3, scale=0.2)
            a = Pedestrian(i, self, pos, goal, speed, radius)
            self.space.place_agent(a, pos)
            self.schedule.add(a)
            self.agents.append(a)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def run_model(self, steps):
        for i in range(steps):
            self.step()

class Pedestrian(Agent):
    def __init__(self, unique_id, model, pos, goal, speed, radius):
        super().__init__(unique_id, model)
        self.pos = pos
        self.vel = np.zeros(2)
        self.goal = goal
        self.speed = speed
        self.radius = radius

    def step(self):
        self.update_velocity()
        self.update_position()

    def update_velocity(self):
        # TODO: Implement the social force model rules to update the velocity of the agent
        pass

    def update_position(self):
        new_pos = self.pos + self.vel
        self.model.space.move_agent(self, new_pos)
        self.pos = new_pos

    def is_exit_reached(self):
        if np.linalg.norm(self.pos - self.model.exit_pos) < 1:
            self.model.schedule.remove(self)
            self.model.datacollector.add_agent(self, {"Exited": True})

# Define the step function for the model
def social_force_step(model):
    for agent in model.schedule.agents:
        agent.step()
        if agent.is_exit_reached():
            model.schedule.remove(agent)

# Define the visualization function
def social_force_draw(agent):
    return plt.Circle(agent.pos, agent.radius, color="red")

# Define the simulation parameters
num_agents = 50
width = 50
height = 50
exit_pos = np.array([width / 2, height + 1])

# Create the model object
model = TrainStationModel(num_agents, width, height, exit_pos)

# Run the simulation for a certain number of steps
model.run_model(100)

# Plot the graph of speed over density, speed over flow, and density over flow
data = model.datacollector.get_model_vars_dataframe()
plt.scatter(data["Density"], data["Speed"])
plt.xlabel("Density")
plt.ylabel("Speed")
plt.show()

plt.scatter(data["Flow"], data["Speed"])
plt.xlabel("Flow")
plt.ylabel("Speed")
plt.show()

plt.scatter(data["Density"], data["Flow"])
plt.xlabel("Density")
plt.ylabel("Flow")
plt.show()

# Export the data collected in the simulation as a CSV file
df = model.datacollector.get_agent_vars_dataframe()
df.to_csv("pedestrian_simulation_data.csv")

# Visualize the simulation
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

canvas_element = CanvasGrid(social_force_draw, width, height, 500, 500)
server = ModularServer(
    TrainStationModel,
    [canvas_element],
    "Train Station Model",
    {
        "num_agents": num_agents,
        "width": width,
        "height": height,
        "exit_pos": exit_pos,
    },
)
server.port = 8521

server.launch()
