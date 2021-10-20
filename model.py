import numpy as np
import random as r
import math
from numba import jit, prange
from pythonabm import Simulation, record_time, template_params


# @jit(nopython=True, parallel=True)
def get_neighbor_forces(number_edges, edges, edge_forces, locations, center, types, radius, alpha=10, r_e=1.01, u_11=30,
                         u_12=1, u_22=5):
    for index in range(number_edges):
        # get indices of cells in edge
        cell_1 = edges[index][0]
        cell_2 = edges[index][1]

        # get cell positions
        cell_1_loc = locations[cell_1] - center
        cell_2_loc = locations[cell_2] - center

        # get new location position
        vec = cell_2_loc - cell_1_loc
        dist = np.linalg.norm(vec)

        # based on the distance apply force differently
        if dist == 0:
            edge_forces[index][0] = alpha * (2 * np.random.rand(3) - 1) * np.array([1, 1, 0])
            edge_forces[index][1] = alpha * (2 * np.random.rand(3) - 1) * np.array([1, 1, 0])
        elif 0 < dist < 2 * radius:
            edge_forces[index][0] = -1 * (10 ** 4) * (vec / dist)
            edge_forces[index][1] = 1 * (10 ** 4) * (vec / dist)
        else:
            # get the cell type
            cell_1_type = types[cell_1]
            cell_2_type = types[cell_2]

            # get value prior to applying type specific adhesion const
            value = (np.linalg.norm(vec) - r_e) * (vec / dist) + alpha * (2 * np.random.rand(3) - 1) * np.array([1, 1, 0])

            if cell_1_type == 0 and cell_2_type == 0:
                edge_forces[index][0] = u_11 * value
                edge_forces[index][1] = -1 * u_11 * value
            elif cell_1_type == 1 and cell_2_type == 1:
                edge_forces[index][0] = u_22 * value
                edge_forces[index][1] = -1 * u_22 * value
            else:
                edge_forces[index][0] = u_12 * value
                edge_forces[index][1] = -1 * u_12 * value

    return edge_forces


@jit(nopython=True, parallel=True)
def get_gravity_forces(number_cells, locations, center, well_rad, net_forces):
    for index in range(number_cells):
        new_loc = locations[index] - center
        net_forces[index] = -1 * (new_loc / well_rad) * np.sqrt(1 - (np.linalg.norm(new_loc) / well_rad) ** 2)
    return net_forces


@jit(nopython=True)
def convert_edge_forces(number_edges, edges, edge_forces, neighbor_forces):
    for index in range(number_edges):
        # get indices of cells in edge
        cell_1 = edges[index][0]
        cell_2 = edges[index][1]

        #
        neighbor_forces[cell_1] += edge_forces[index][0]
        neighbor_forces[cell_2] += edge_forces[index][1]

    return neighbor_forces


def set_div_thresh(cell_type):
    """ Specify division threshold value for a particular cell.

        Distribution of cell division thresholds modeled by a shifted gamma distribution
        from Stukalin et al., RSIF 2013
    """
    # parameters for gamma distribution
    alpha, a_0, beta = 12.5, 10.4, 0.72

    # based on cell type return division threshold in seconds
    if cell_type == 0:
        alpha, a_0, beta = 12.5, 10.4, 0.72
        hours = r.gammavariate(alpha, beta) + a_0
        # hours = 51
        # CHO cell time < HEK cell time
    else:
        alpha, a_0, beta = 10, 10.4, 0.72
        hours = r.gammavariate(alpha, beta) + a_0

    return hours * 3600


def seed_cells(num_cells, radius, well_dimensions):
    # radius of the circle
    # center of sphere (x, y, z)
    center_x = well_dimensions[0] / 2
    center_y = well_dimensions[1] / 2
    center_z = well_dimensions[2] / 2
    locations = np.zeros((num_cells,3))
    # random angle
    if center_z > 0:
        for i in range(num_cells):
            phi = 2 * math.pi * r.random()
            theta = 2 * math.pi * r.random()
            rad = radius * math.sqrt(r.random())
            x = rad * math.cos(theta) * math.sin(phi) + center_x
            y = rad * math.sin(theta) * math.sin(phi) + center_y
            z = rad * math.cos(phi) + center_z
            locations[i] = np.array([x, y, z])
        return locations
    # random radius
    for i in range(num_cells):
        theta = 2 * math.pi * r.random()
        rad = radius * math.sqrt(r.random())
        x = rad * math.cos(theta) + center_x
        y = rad * math.sin(theta) + center_y
        locations[i] = np.array([x, y, 0])
    return locations


class TestSimulation(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """
    def __init__(self):
        # initialize the Simulation object
        Simulation.__init__(self)

        # read parameters from YAML file and add them to instance variables
        self.yaml_parameters("general.yaml")

        # HEK/CHO ratio
        self.ratio = 0.7

        # scale well size by diameter of cell:
        self.cell_rad = 0.5
        self.well_rad = 325
        self.hek_color = np.array([255, 255, 0], dtype=int)
        self.cho_color = np.array([50, 50, 255], dtype=int)
        self.dim = np.asarray(self.size)
        self.size = self.dim * self.well_rad

    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """
        # determine the number of agents for each cell type
        num_hek = int(self.num_to_start * self.ratio)
        num_cho = self.num_to_start - num_hek

        # add agents to the simulation
        self.add_agents(num_hek, agent_type="HEK")
        self.add_agents(num_cho, agent_type="CHO")

        # indicate agent arrays and create the arrays with initial conditions
        self.indicate_arrays("locations", "radii", "colors", "cell_type", "division_set", "div_thresh")

        # generate random locations for cells
        self.locations = seed_cells(self.number_agents, self.well_rad/8, self.size)
        self.radii = self.agent_array(initial=lambda: self.cell_rad)

        # 1 is HEK293FT Cell (yellow), 0 is CHO K1 Cell (blue)
        self.cell_type = self.agent_array(dtype=int, initial={"HEK": lambda: 1, "CHO": lambda: 0})
        self.colors = self.agent_array(dtype=int, vector=3, initial={"HEK": lambda: self.hek_color, "CHO": lambda: self.cho_color})

        # setting division times (in seconds):
        self.div_thresh = self.agent_array(initial={"HEK": lambda: set_div_thresh(1), "CHO": lambda: set_div_thresh(0)})
        self.division_set = self.agent_array(initial={"HEK": lambda: 19 * 3600 * r.random(), "CHO": lambda: 19 * 3600 * r.random()})

        # indicate agent graphs and create the graphs for holding agent neighbors
        self.indicate_graphs("neighbor_graph")
        self.neighbor_graph = self.agent_graph()

        # reduce overlap during initialization
        for i in range(self.number_agents):
            self.remove_overlap(i)

        # record initial values
        self.step_values()
        self.step_image()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # preform 60 subsets, each 1 second long
        for i in range(60):
            # increase division counter and determine if any cells are dividing
            self.reproduce(1)

            # get all neighbors within threshold (1.6 * diameter)
            self.get_neighbors(self.neighbor_graph, 3.2 * self.cell_rad)

            # move the cells
            self.move_parallel()

            # add/remove agents from the simulation
            self.update_populations()

        # get the following data
        self.step_values()
        self.step_image()
        self.temp()
        self.data()

    def end(self):
        """ Overrides the end() method from the Simulation class.
        """
        self.create_video()

    @record_time
    def update_populations(self):
        """ Adds/removes agents to/from the simulation by adding/removing
            indices from the cell arrays and any graphs.
        """
        # get indices of hatching/dying agents with Boolean mask
        add_indices = np.arange(self.number_agents)[self.hatching]
        remove_indices = np.arange(self.number_agents)[self.removing]

        # count how many added/removed agents
        num_added = len(add_indices)
        num_removed = len(remove_indices)
        # go through each agent array name
        for name in self.array_names:
            # copy the indices of the agent array data for the hatching agents
            copies = self.__dict__[name][add_indices]

            # add indices to the arrays
            self.__dict__[name] = np.concatenate((self.__dict__[name], copies), axis=0)

            # if locations array
            if name == "locations":
                # go through the number of cells added
                for i in range(num_added):
                    # get mother and daughter indices
                    mother = add_indices[i]
                    daughter = self.number_agents + i

                    # move distance of 5 in random direction
                    vec = self.radii[i] * self.random_vector()
                    self.__dict__[name][mother] += vec
                    self.__dict__[name][daughter] -= vec

            # reset division time
            if name == "division_set":
                # go through the number of cells added
                for i in range(num_added):
                    # get mother and daughter indices
                    mother = add_indices[i]
                    daughter = self.number_agents + i

                    # set division counter to zero
                    self.__dict__[name][mother] = 0
                    self.__dict__[name][daughter] = 0

            # set new division threshold
            if name == "division_threshold":
                # go through the number of cells added
                for i in range(num_added):
                    # get daughter index
                    daughter = self.number_agents + i

                    # set division threshold based on cell type
                    self.__dict__[name][daughter] = set_div_thresh(self.cell_type[daughter])

            # remove indices from the arrays
            self.__dict__[name] = np.delete(self.__dict__[name], remove_indices, axis=0)

        # go through each graph name
        for graph_name in self.graph_names:
            # add/remove vertices from the graph
            self.__dict__[graph_name].add_vertices(num_added)
            self.__dict__[graph_name].delete_vertices(remove_indices)

        # change total number of agents and print info to terminal
        self.number_agents += num_added
        # print("\tAdded " + str(num_added) + " agents")
        # print("\tRemoved " + str(num_removed) + " agents")

        # clear the hatching/removing arrays for the next step
        self.hatching[:] = False
        self.removing[:] = False

    def move_parallel(self):
        edges = np.asarray(self.neighbor_graph.get_edgelist())
        num_edges = len(edges)
        edge_forces = np.zeros((num_edges, 2, 3))
        center = self.size / 2
        neighbor_forces = np.zeros((self.number_agents, 3))
        grav_forces = np.zeros((self.number_agents, 3))

        # get adhesive/repulsive forces from neighbors and gravity forces
        edge_forces = get_neighbor_forces(num_edges, edges, edge_forces, self.locations, center, self.cell_type,
                                          self.cell_rad)
        neighbor_forces = convert_edge_forces(num_edges, edges, edge_forces, neighbor_forces)
        grav_forces = get_gravity_forces(self.number_agents, self.locations, center, self.well_rad, grav_forces)

        # get normalized vector of total force
        total_force = neighbor_forces + grav_forces
        for i in range(self.number_agents):
            total_force[i] = total_force[i] / np.linalg.norm(total_force[i])

        # update locations based on forces
        self.locations += 0.3 * self.cell_rad * total_force

        # check that the new location is within the space, otherwise use boundary values
        self.locations = np.where(self.locations > self.well_rad, self.well_rad, self.locations)
        self.locations = np.where(self.locations < 0, 0, self.locations)

    @record_time
    def reproduce(self, ts):
        """ If the agent meets criteria, hatch a new agent.
        """
        # increase division counter by time step for all agents
        self.division_set += ts

        # go through all agents marking for division if over the threshold
        for index in range(self.number_agents):
            if self.division_set[index] > self.div_thresh[index]:
                self.mark_to_hatch(index)

    def remove_overlap(self, index):
        self.get_neighbors(self.neighbor_graph, 2*self.radii[index])
        while len(self.neighbor_graph.neighbors(index)) > 0:
            for neighbor_cell in self.neighbor_graph.neighbors(index):
                mag = np.linalg.norm(self.locations[neighbor_cell] - self.locations[index])
                vec = mag * np.random.rand(3) * self.dim
                self.locations[index] += vec
                self.locations[neighbor_cell] -= vec
            self.get_neighbors(self.neighbor_graph, 2*self.radii[index])

    @classmethod
    def simulation_mode_0(cls, name, output_dir):
        """ Creates a new brand new simulation and runs it through
            all defined steps.
        """
        # make simulation instance, update name, and add paths
        sim = cls()
        sim.name = name
        sim.set_paths(output_dir)

        # set up the simulation agents and run the simulation
        sim.full_setup()
        sim.run_simulation()


if __name__ == "__main__":
    TestSimulation.start("/Users/andrew/PycharmProjects/tordoff_model/Outputs")
    #TestSimulation.start("C:\\Research\\Code\\Tordoff_model_outputs")
