import numpy as np
from numba import jit

from pythonabm import Simulation, record_time


@jit(nopython=True)
def midrange_attraction(r_ij, r_e, alpha, u_i, u_j, u_11=30, u_12=1, u_22=5):
    u_ij = np.array([u_22, u_12, u_11])
    return u_ij[(u_i + u_j)]*(np.linalg.norm(r_ij) - r_e)*(r_ij/np.linalg.norm(r_ij)) + alpha*(2*np.random.rand(3)-1)*np.array([1, 1, 0])


def set_division_threshold(num_cells, alpha=12.5, a_0=10.4, beta=0.72):
    return (np.random.gamma(alpha, beta, num_cells) + a_0) * 3600

@jit(nopython=True)
def pairwise_numba(cell_type, cell_neighbors, cell_neighbors_type, cell_rad, alpha=10, r_e=1.01):
    pairwise_force = np.zeros(3)
    for i in range(len(cell_neighbors)):
        r_ij_mag = np.linalg.norm(cell_neighbors[i])
        if r_ij_mag == 0:
            pairwise_force += alpha*(2*np.random.rand(3)-1)*np.array([1, 1, 0])
        elif 0 < r_ij_mag < 2 * cell_rad:
            pairwise_force -= (10**4)*(cell_neighbors[i]/r_ij_mag)
        elif 2 * cell_rad < r_ij_mag:
            pairwise_force += midrange_attraction(cell_neighbors[i], r_e, alpha, cell_type, cell_neighbors_type[i])
    return pairwise_force

class TestSimulation(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """
    def __init__(self):
        # initialize the Simulation object
        Simulation.__init__(self)
        print(np.__version__)
        # read parameters from YAML file and add them to instance variables
        self.yaml_parameters("general.yaml")

    def setup(self, ratio=.8):
        """ Overrides the setup() method from the Simulation class.
        """
        # add agents to the simulation
        self.add_agents(self.num_to_start)
        # indicate agent arrays and create the arrays with initial conditions
        self.indicate_arrays("locations", "radii", "new_locations", "colors", "cell_type", "division_set", "division_threshold")
        # scale well size by diameter of cell:
        self.well_rad = 2 * self.cell_rad * self.well_rad
        self.size = np.array(self.size) * self.well_rad
        # Generate random locations for cells
        self.locations = np.random.rand(self.number_agents, 3) * self.size
        self.new_locations = self.locations
        self.radii = self.agent_array(initial=lambda: self.cell_rad)
        # 1 is HEK293FT Cell (yellow), 0 is CHO K1 Cell (blue)
        self.cell_type = np.random.choice([1, 0], self.number_agents, p=[ratio, 1 - ratio])
        self.colors[self.cell_type.nonzero()] = [0, 255, 255]

        # Setting division times (in seconds):
        alpha = 12.5
        a_0 = 10.4
        beta = 0.72
        self.division_threshold = set_division_threshold(self.number_agents)
        self.division_set = np.random.rand(self.number_agents) * 19 * 3600

        # indicate agent graphs and create the graphs for holding agent neighbors
        self.indicate_graphs("neighbor_graph")
        self.neighbor_graph = self.agent_graph()
        for i in range(self.number_agents):
            self.remove_overlap(i)
        # record initial values
        self.step_values()
        self.step_image()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # get all neighbors within radius of 2
        # call the following methods that update agent values
        # self.reproduce()
        for i in range(100):
            self.reproduce(1)
            self.get_neighbors(self.neighbor_graph, 3.2*self.cell_rad)
            # self.move()
            self.move_numba(well_rad=self.well_rad)
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

    # @record_time
    # def die(self):
    #     """ Updates an agent based on the presence of neighbors.
    #     """
    #     # determine which agents are being removed
    #     for index in range(self.number_agents):
    #         if r.random() < 0.001:
    #             self.mark_to_remove(index)

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
                for i in range(num_added):
                    # get mother and daughter indices
                    mother = add_indices[i]
                    daughter = self.number_agents + i
                    self.__dict__[name][mother] = 0
                    self.__dict__[name][daughter] = 0

            # set new division threshold
            if name == "division_threshold":
                for i in range(num_added):
                    daughter = self.number_agents + i
                    self.__dict__[name][daughter] = set_division_threshold(1)

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

    @record_time
    def move(self):
        """ Assigns new location to agent.
        """
        for index in range(self.number_agents):
            # get new location position
            self.new_locations[index] = self.locations[index] + .05 * self.radii[index]*\
                                        self.cell_net_force(self.locations[index], index, well_rad=self.well_rad)

            # check that the new location is within the space, otherwise use boundary values
            # Assumes square size.
        self.new_locations = np.where(self.new_locations > self.well_rad, self.well_rad, self.new_locations)
        self.new_locations = np.where(self.new_locations < 0, 0, self.new_locations)
        self.locations = self.new_locations

    @record_time
    def move_numba(self, well_rad=325):
        for index in range(self.number_agents):
            # get new location position
            cell_loc = self.locations[index] - np.array(self.size)/2
            cell_neighbors_loc = self.locations[self.neighbor_graph.neighbors(index)]
            sum_force = np.zeros(3)
            if len(cell_neighbors_loc) > 0:
                cell_neighbors_type = np.array(self.cell_type[self.neighbor_graph.neighbors(index)])
                cell_neighbors = cell_neighbors_loc - np.array(self.size)/2
                cell_neighbors = np.array(cell_neighbors - cell_loc)
                sum_force = pairwise_numba(self.cell_type[index], cell_neighbors, cell_neighbors_type, self.cell_rad)
            net_force = -(cell_loc/well_rad)*np.sqrt(1-(np.linalg.norm(cell_loc)/well_rad)**2)
            total_force = (sum_force + net_force)/np.linalg.norm(sum_force + net_force)
            self.new_locations[index] = self.locations[index] + .05 * self.cell_rad * total_force

            # check that the new location is within the space, otherwise use boundary values
        self.new_locations = np.where(self.new_locations > self.well_rad, self.well_rad, self.new_locations)
        self.new_locations = np.where(self.new_locations < 0, 0, self.new_locations)
        self.locations = self.new_locations

    @record_time
    def reproduce(self, ts):
        """ If the agent meets criteria, hatch a new agent.
        """
        # determine which agents are hatching
        self.division_set = self.division_set + ts
        for index in range(self.number_agents):
            if self.division_set[index] > self.division_threshold[index]:
                self.mark_to_hatch(index)

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

    def cell_net_force(self, cell_loc, index, alpha=10, r_e=1.01, well_rad=325):
        # centering cartesian plane
        center = np.array(self.size)/2
        new_cell_loc = cell_loc - center
        pairwise_force = np.zeros(3)
        for pairwise in self.neighbor_graph.neighbors(index):
            r_ij = (self.locations[pairwise] - center) - new_cell_loc
            r_ij_mag = np.linalg.norm(r_ij)
            if r_ij_mag == 0:
                pairwise_force += alpha*(2*np.random.rand(3)-1)*np.array([1, 1, 0])
            elif 0 < r_ij_mag < 2*self.radii[index]:
                pairwise_force += -(10**4)*(r_ij/r_ij_mag)
            elif 2*self.radii[index] < r_ij_mag:
                pairwise_force += midrange_attraction(r_ij, r_e, alpha, self.cell_type[index], self.cell_type[pairwise])
        # 2D static force field drives cells towards center of simulation domain
        # if np.linalg.norm(pairwise_force) > 0:
        #     pairwise_force = pairwise_force/np.linalg.norm(pairwise_force)
        force_ext = -(new_cell_loc/well_rad)*np.sqrt(1-(np.linalg.norm(new_cell_loc)/well_rad)**2)
        return (force_ext + pairwise_force)/np.linalg.norm(force_ext + pairwise_force)
        # return force_ext + pairwise_force

    def remove_overlap(self, index):
        self.get_neighbors(self.neighbor_graph, 2*self.radii[index])
        while len(self.neighbor_graph.neighbors(index)) > 0:
            for neighbor_cell in self.neighbor_graph.neighbors(index):
                vec = np.random.rand(3)*np.array([1, 1, 0])
                self.locations[index] += vec
                self.locations[neighbor_cell] -= vec
            self.get_neighbors(self.neighbor_graph, 2*self.radii[index])


if __name__ == "__main__":
    TestSimulation.start("/Users/andrew/PycharmProjects/tordoff_model/Outputs")