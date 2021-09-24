import numpy as np
import random as r
import math

from pythonabm import Simulation, record_time


def midrange_attraction(r_ij, r_e, u_i, u_j, alpha=.01, u_11=30, u_12=1, u_22=5):
    if u_i == u_j == 1:
        return u_11*(np.linalg.norm(r_ij) - r_e)*(r_ij/np.linalg.norm(r_ij)) + alpha*np.random.rand(3)*np.array([1, 1, 0])
    elif u_i != u_j:
        return u_12
    else:
        return u_22


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
        # self.add_agents(int(self.num_to_start*ratio), agent_type='cho')
        # self.add_agents(int(self.num_to_start*(1-ratio)), agent_type='hek')
        self.add_agents(self.num_to_start)
        # indicate agent arrays and create the arrays with initial conditions
        self.indicate_arrays("locations", "radii", "colors")
        self.locations = np.random.rand(self.number_agents, 3) * self.size
        self.radii = self.agent_array(initial=lambda: 5)

        # indicate agent graphs and create the graphs for holding agent neighbors
        self.indicate_graphs("neighbor_graph")
        self.neighbor_graph = self.agent_graph()

        # record initial values
        self.step_values()
        self.step_image()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # get all neighbors within radius of 2
        self.get_neighbors(self.neighbor_graph, 5)

        # call the following methods that update agent values
        self.die()
        self.reproduce()
        self.move()

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
    def die(self):
        """ Updates an agent based on the presence of neighbors.
        """
        # determine which agents are being removed
        for index in range(self.number_agents):
            if r.random() < 0.1:
                self.mark_to_remove(index)

    @record_time
    def move(self):
        """ Assigns new location to agent.
        """
        for index in range(self.number_agents):
            # get new location position
            new_location = self.locations[index] + .05 * self.radii[index]*self.cell_net_force(self.locations[index], index)
            # check that the new location is within the space, otherwise use boundary values
            for i in range(3):
                if new_location[i] > self.size[i]:
                    self.locations[index][i] = self.size[i]
                elif new_location[i] < 0:
                    self.locations[index][i] = 0
                else:
                    self.locations[index][i] = new_location[i]

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
                    vec = 5 * self.random_vector()
                    self.__dict__[name][mother] += vec
                    self.__dict__[name][daughter] -= vec

            # remove indices from the arrays
            self.__dict__[name] = np.delete(self.__dict__[name], remove_indices, axis=0)

        # go through each graph name
        for graph_name in self.graph_names:
            # add/remove vertices from the graph
            self.__dict__[graph_name].add_vertices(num_added)
            self.__dict__[graph_name].delete_vertices(remove_indices)

        # change total number of agents and print info to terminal
        self.number_agents += num_added - num_removed
        print("\tAdded " + str(num_added) + " agents")
        print("\tRemoved " + str(num_removed) + " agents")

        # clear the hatching/removing arrays for the next step
        self.hatching[:] = False
        self.removing[:] = False

    @record_time
    def reproduce(self):
        """ If the agent meets criteria, hatch a new agent.
        """
        # determine which agents are hatching
        for index in range(self.number_agents):
            if r.random() < 0.1:
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

    def cell_net_force(self, cell_loc, index, r_o=1.6, r_e=1.01, well_rad=1000):
        # centering cartesian plane
        r_o = r_o * self.radii[index]
        center = np.array(self.size)/2
        new_cell_loc = cell_loc - center
        pairwise_force = np.zeros(3)
        for pairwise in range(self.number_agents):
            if index != pairwise:
                r_ij = new_cell_loc - (np.array(self.locations[pairwise]) - center)
                r_ij_mag = np.linalg.norm(r_ij)
                if self.radii[index] <= r_ij_mag < r_o:
                    pairwise_force += midrange_attraction(r_ij, r_e, 1, 1)
                elif 0 < r_ij_mag < self.radii[index]:
                    pairwise_force += -(10**4)*r_ij/r_ij_mag
        # 2D static force field drives cells towards center of simulation domain
        force_ext = -(new_cell_loc/well_rad)*np.sqrt(1-(np.linalg.norm(new_cell_loc)/well_rad)**2)
        if np.linalg.norm(pairwise_force) > 0:
            pairwise_force = pairwise_force/np.linalg.norm(pairwise_force)
        return force_ext + pairwise_force


if __name__ == "__main__":
    TestSimulation.start("/Users/andrew/PycharmProjects/tordoff_model/Outputs")
