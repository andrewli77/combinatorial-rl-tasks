import numpy as np

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


"""Simple Travelling Salesperson Problem (TSP) between cities."""
def extract_solution(manager, routing, solution):
    """Prints solution on console."""
    # print('Objective: {} miles'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = []#'Route for vehicle 0:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += [manager.IndexToNode(index) - 1]
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += [manager.IndexToNode(index) - 1]

    return np.array(plan_output)[1:-1]


def get_optim_route(env):
    rob_pos = np.asarray(env.world.robot_pos(), dtype='float64')[:-1]
    city_pos = np.vstack(([rob_pos], np.array(env.zones_pos)[:, :-1]))

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(env.num_cities + 1, 1, 0)

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""

        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        from_pos = city_pos[from_node]
        to_pos = city_pos[to_node]

        ret = np.sqrt(np.sum((from_pos - to_pos)**2))

        return ret * 10 # multiplying by 10 just for better readability for debugging

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    return extract_solution(manager, routing, solution)
