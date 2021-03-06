import numpy as np
import random as r
import math
from numba import jit, prange
from pythonabm import Simulation, record_time, template_params
import cv2
from numba import cuda
import numba
from pythonabm.backend import record_time, check_direct, template_params, check_existing, get_end_step, Graph, \
    progress_bar, starting_params, check_output_dir, assign_bins_jit, get_neighbors_cpu, get_neighbors_gpu


@jit(nopython=True, parallel=True)
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
            value = (dist - r_e) * (vec / dist)

            if cell_1_type == 0 and cell_2_type == 0:
                edge_forces[index][0] = u_22 * value + alpha * (2 * np.random.rand(3) - 1) * np.array([1, 1, 0])
                edge_forces[index][1] = -1 * u_22 * value + alpha * (2 * np.random.rand(3) - 1) * np.array([1, 1, 0])
            elif cell_1_type == 1 and cell_2_type == 1:
                edge_forces[index][0] = u_11 * value + alpha * (2 * np.random.rand(3) - 1) * np.array([1, 1, 0])
                edge_forces[index][1] = -1 * u_11 * value + alpha * (2 * np.random.rand(3) - 1) * np.array([1, 1, 0])
            else:
                edge_forces[index][0] = u_12 * value + alpha * (2 * np.random.rand(3) - 1) * np.array([1, 1, 0])
                edge_forces[index][1] = -1 * u_12 * value + alpha * (2 * np.random.rand(3) - 1) * np.array([1, 1, 0])

    return edge_forces


@jit(nopython=True, parallel=True)
def get_gravity_forces(number_cells, locations, center, well_rad, net_forces):
    for index in range(number_cells):
        new_loc = locations[index] - center
        # net_forces[index] = -1 * (new_loc / well_rad) * np.sqrt((np.linalg.norm(new_loc) / well_rad) ** 2)
        net_forces[index] = -1 * (new_loc / well_rad) * np.sqrt(1 - (np.linalg.norm(new_loc) / well_rad) ** 2)
    return net_forces

# The clusters are getting so big that the net force isn't being applied to them
# TO DO: try to find a way to make relatively small clusters (~ 5-10 cells, the ones in .3 HEK/CHO ratio not
# be heavily influenced by the cluster forces, but the larger clusters do.
# OR: remove?
@jit(nopython=True, parallel=True)
def get_cluster_forces(clusters, num_clusters, cluster_size, centroids, center, well_rad, net_forces, num_hek):
    temp = 0
    for index in range(num_clusters):
        cluster_loc = centroids[index] - center
        cluster_force = -1 * ((cluster_size[index] / num_hek) ** 2) * np.sqrt(1 - (cluster_size[index] / num_hek) ** 2)\
                    * (cluster_loc / well_rad) * np.sqrt(1 - (np.linalg.norm(cluster_loc) / well_rad) ** 2)
        for i in range(cluster_size[index]):
            net_forces[clusters[i + temp]] = cluster_force
        temp += cluster_size[index]
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
        self.hek_color = np.array([255, 255, 0], dtype=int)
        self.cho_color = np.array([50, 50, 255], dtype=int)

        # movement parameters
        self.velocity = 0.2
        self.noise_magnitude = self.velocity/20

        # scale well size by diameter of cell:, size parameters
        self.cell_rad = 0.5
        self.well_rad = 325
        self.initial_seed_rad = 325/8
        self.cell_interaction_rad = 3.2
        self.dim = np.asarray(self.size)
        self.size = self.dim * self.well_rad

        # cluster identification parameters
        self.cluster_threshold = 3
        self.cluster_interaction_threshold = 3.2
        self.cluster_record_interval = 5
        self.cluster_timer = 0

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
        self.locations = seed_cells(self.number_agents, self.initial_seed_rad, self.size)
        self.radii = self.agent_array(initial=lambda: self.cell_rad)

        # 1 is HEK293FT Cell (yellow), 0 is CHO K1 Cell (blue)
        self.cell_type = self.agent_array(dtype=int, initial={"HEK": lambda: 1, "CHO": lambda: 0})
        self.colors = self.agent_array(dtype=int, vector=3, initial={"HEK": lambda: self.hek_color, "CHO": lambda: self.cho_color})

        # setting division times (in seconds):
        self.div_thresh = self.agent_array(initial={"HEK": lambda: set_div_thresh(1), "CHO": lambda: set_div_thresh(0)})
        self.division_set = self.agent_array(initial={"HEK": lambda: 17 * 3600 * r.random(), "CHO": lambda: 16 * 3600 * r.random()})

        # indicate agent graphs and create the graphs for holding agent neighbors
        self.indicate_graphs("neighbor_graph", "cluster_graph")
        self.neighbor_graph = self.agent_graph()
        self.cluster_graph = self.agent_graph()

        # We're going to try to keep track of the clusters
        self.clusters = []
        self.cluster_centroids = np.zeros(1)
        self.cluster_radii = []
        self.cluster_sizes = np.zeros(1, dtype=int)
        # record initial values
        self.step_values()
        self.get_clusters(self.cluster_threshold, self.cluster_record_interval, self.cluster_interaction_threshold)
        self.step_image()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # preform 60 subsets, each 1 second long
        self.cluster_timer += 1
        for i in range(60):
            # increase division counter and determine if any cells are dividing
            self.reproduce(1)

            # get all neighbors within threshold (1.6 * diameter)
            self.get_neighbors(self.neighbor_graph, self.cell_interaction_rad * self.cell_rad)
            # self.clusters, self.cluster_centroids, self.cluster_radii, self.cluster_sizes = self.get_clusters(1)

            self.get_clusters(self.cluster_threshold, self.cluster_record_interval,
                              self.cluster_interaction_threshold)
            # move the cells
            self.move_parallel()
            self.noise(self.noise_magnitude)
            # add/remove agents from the simulation
            self.update_populations()
        # get the following data
        self.step_values()
        print(f"HEK: {np.sum(self.cell_type)}")
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

                    # move distance of radius in random direction
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
        total_force = np.zeros((self.number_agents, 3))

        # get adhesive/repulsive forces from neighbors and gravity forces
        edge_forces = get_neighbor_forces(num_edges, edges, edge_forces, self.locations, center, self.cell_type,
                                          self.cell_rad)
        neighbor_forces = convert_edge_forces(num_edges, edges, edge_forces, neighbor_forces)
        grav_forces = get_gravity_forces(self.number_agents, self.locations, center, self.well_rad, grav_forces)
        # total_force = neighbor_forces + grav_forces
        for i in range(self.number_agents):
            if np.linalg.norm(neighbor_forces[i]) > 1:
                total_force[i] = (neighbor_forces[i]/np.linalg.norm(neighbor_forces[i])) + grav_forces[i]
            elif np.linalg.norm(neighbor_forces[i]) > 0:
                total_force[i] = neighbor_forces[i] + grav_forces[i]
            else:
                total_force[i] = grav_forces[i]
            total_force[i] = total_force[i] / np.linalg.norm(total_force[i])
        # update locations based on forces
        self.locations += 2 * self.velocity * self.cell_rad * total_force
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

    # Not used
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

    def noise(self, alpha):
        self.locations += alpha * 2 * self.cell_rad * np.random.normal(size=(self.number_agents, 3)) * self.dim

    def get_clusters(self, cluster_threshold, time_thresh, cluster_distance):
        # Create graphs of specified distance.
        if self.cluster_timer % time_thresh == 0:
            self.get_neighbors_clusters(self.cluster_graph, cluster_distance * self.cell_rad)
            # Identify unique clusters in graph
            clusters = self.cluster_graph.clusters()
            file_name = f"{self.name}_values_{self.current_step}_clusters.csv"
            cluster_file = open(self.values_path + file_name, "w")
            if len(clusters) > 0:
                centroids = np.zeros([len(clusters),3])
                radius = np.zeros(len(clusters))
                # Calculate Mean
                for i in range(len(clusters)):
                    if len(clusters[i]) > cluster_threshold:
                        location_graph = self.locations[clusters[i]]
                        centroids[i] = np.mean(location_graph,0)
                        max_distance = 0
                        # and Radius for circles
                        for j in range(len(clusters[i])):
                            if np.linalg.norm(location_graph[j]-centroids[i]) > max_distance:
                                max_distance = np.linalg.norm(location_graph[j]-centroids[i])
                        radius[i] = max_distance
                        cluster_file.write(f"{centroids[i][0]}, {centroids[i][1]}, {centroids[i][2]}, {radius[i]}\n")
                if len(centroids > 0):
                    self.step_image_cluster(centroids, radius)
            cluster_file.close()


    @record_time
    def step_image_cluster(self, centroids, radius, background=(0, 0, 0), origin_bottom=True):
        """ Creates an image of the simulation space with a cluster overlay.
        """
        # only continue if outputting images
        if self.output_images:
            # get path and make sure directory exists


            # get the size of the array used for imaging in addition to the scaling factor
            x_size = self.image_quality
            scale = x_size / self.size[0]
            y_size = math.ceil(scale * self.size[1])

            # create the agent space background image and apply background color
            image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
            background = (background[2], background[1], background[0])
            image[:, :] = background

            # go through all of the agents
            for index in range(self.number_agents):
                # get xy coordinates, the axis lengths, and color of agent
                x, y = int(scale * self.locations[index][0]), int(scale * self.locations[index][1])
                major, minor = int(scale * self.radii[index]), int(scale * self.radii[index])
                color = (int(self.colors[index][2]), int(self.colors[index][1]), int(self.colors[index][0]))

                # draw the agent and a black outline to distinguish overlapping agents
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, color, -1)
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, (0, 0, 0), 1)
            for i in range(len(centroids)):
                x = int(scale * centroids[i, 0])
                y = int(scale * centroids[i, 1])
                rad = int(radius[i] * scale)
                image = cv2.ellipse(image, (x, y), (rad, rad), 0, 0, 360, (0, 0, 255), 3)
            # if the origin should be bottom-left flip it, otherwise it will be top-left
            if origin_bottom:
                image = cv2.flip(image, 0)

            # save the image as a PNG
            image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)
            file_name = f"{self.name}_image_{self.current_step}_cluster.png"
            cv2.imwrite(self.images_path + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression])

    @record_time
    def get_neighbors_clusters(self, graph, distance, clear=True):
        """ Finds all neighbors, within fixed radius, for each each agent.
        """
        # get graph object reference and if desired, remove all existing edges in the graph
        if clear:
            graph.delete_edges(None)

        # don't proceed if no agents present
        if np.sum(self.cell_type) == 0:
            return

        # assign each of the agents to bins, updating the max agents in a bin (if necessary)
        bins, bins_help, bin_locations, graph.max_agents = self.assign_bins(graph.max_agents, distance)

        # run until all edges are accounted for
        while True:
            # get the total amount of edges able to be stored and make the following arrays
            # We are only looking at HEK cells here
            length = np.sum(self.cell_type) * graph.max_neighbors
            edges = np.zeros((length, 2), dtype=int)         # hold all edges
            if_edge = np.zeros(length, dtype=bool)                 # say if each edge exists
            edge_count = np.zeros(np.sum(self.cell_type), dtype=int)   # hold count of edges per agent

            # if using CUDA GPU
            if self.cuda:
                # allow the following arrays to be passed to the GPU
                edges = cuda.to_device(edges)
                if_edge = cuda.to_device(if_edge)
                edge_count = cuda.to_device(edge_count)

                # specify threads-per-block and blocks-per-grid values
                tpb = 72
                bpg = math.ceil(self.number_agents / tpb)

                # call the CUDA kernel, sending arrays to GPU
                get_neighbors_gpu[bpg, tpb](cuda.to_device(self.locations), cuda.to_device(bin_locations),
                                            cuda.to_device(bins), cuda.to_device(bins_help), distance, edges, if_edge,
                                            edge_count, graph.max_neighbors)

                # return the following arrays back from the GPU
                edges = edges.copy_to_host()
                if_edge = if_edge.copy_to_host()
                edge_count = edge_count.copy_to_host()

            # otherwise use parallelized JIT function
            else:
                edges, if_edge, edge_count = get_neighbors_cpu(np.sum(self.cell_type),  self.locations[self.cell_type==1], bin_locations, bins,
                                                               bins_help, distance, edges, if_edge, edge_count,
                                                               graph.max_neighbors)

            # break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
            max_neighbors = np.amax(edge_count)
            if graph.max_neighbors >= max_neighbors:
                break
            else:
                graph.max_neighbors = max_neighbors * 2

        # reduce the edges to edges that actually exist and add those edges to graph
        graph.add_edges(edges[if_edge])

        # simplify the graph's edges if not clearing the graph at the start
        if not clear:
            graph.simplify()


if __name__ == "__main__":
    TestSimulation.start("/Users/andrew/PycharmProjects/tordoff_model/Outputs")
    #TestSimulation.start("C:\\Research\\Code\\Tordoff_model_outputs")
