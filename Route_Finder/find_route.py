__author__ = 'narayans'
from myqueue import *
from stack import *


def reset(all_cities):
    '''
    Mark all cities in the dictionary as True, i.e. not visited
    :param all_cities: dictionary listing all cities
    :return: None
    '''
    for w in all_cities:
        all_cities.__setitem__(w,True)


def read_flights(filename):
    '''
    Reads the referenced text file
    and returns two dictionaries as described below

    :param filename: file with the flight data
    :return: two dictionaries
                 dictionary 1: key = city, value = list of tuples corresponding to destinations as described in file
                 dictionary 2: key = city, value = True
    '''
    import ast
    file = open(filename, "r")
    flights = {}
    cities = {}
    for line in file:
        line = line.strip()
        source,dest = line.split(':')
        flights[source] = ast.literal_eval(dest)
        cities[source] = True

    return flights, cities

def get_next_cities(current_city,all_flights):
    '''
    Returns a list representing cities that are directly reachable from current city
    :param current_city: some city
    :return: list representing cities with direct flights from current city
    '''

    cities = all_flights[current_city]

    return cities


def find_route(start_city, end_city, all_flights, all_cities):
    '''
    Use the algorithm described in the assignment to find a route between
    start_city and end_city
    :param start_city: the start city
    :param end_city:  the end city
    :param all_flights: dictionary describing all flights
    :param all_cities: dictionary corresponding to all cities
    :return: A stack corresponding to the shortest route between start_city and end_city, or None if no route exists
    '''

    routes = Queue()
    route_1 = Stack()
    start = (start_city, 1)
    route_1.push(start)
    routes.enqueue(route_1)
    all_cities.__setitem__(start_city,False)

    while routes.isempty() == False:
        route = routes.dequeue()
        last = route.peek()
        last_city = last[0]
        if last_city == end_city:
            final = []
            final2 = []
            while route.size() != 0:
                city = route.pop()
                final.append(city[0])
            for item in reversed(final):
                final2.append(item)
            return final2
        else:
            temp = get_next_cities(last_city, all_flights)
            for city in temp:
                temp_route = route.clone()
                temp_route.push(city)
                routes.enqueue(temp_route)
                all_cities.__setitem__(city,False)


def find_all():
    all_flights, all_cities = read_flights("Resources/flights.txt")
    file = open("Resources/city_pairs.txt")
    all_pairs = []
    all_routes = []
    for line in file:
        start,end = line.split(",")
        end = end.strip("\n")
        pair = (start,end)
        all_pairs.append(pair)
    for pair in all_pairs:
        route = find_route(pair[0], pair[1], all_flights, all_cities)
        all_routes.append(route)
    return all_routes

print(find_all())