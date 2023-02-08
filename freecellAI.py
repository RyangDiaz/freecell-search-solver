import time
import csv
from freecell import *
from collections import deque
from copy import deepcopy
from random import *

import sys
sys.path.append('../aima-python')
from search import PriorityQueue

VERBOSE = True

# Utilities for controlled deck generation
HIGHEST_CARD_RANK = 7
VALID_CARD_RANKS = "A23456789TJQK"[:HIGHEST_CARD_RANK]
VALID_CARD_SUITS = "CDHS" # Must have all four suits
NUM_CARDS = len(VALID_CARD_RANKS) * len(VALID_CARD_SUITS)

# Search parameters
NODE_SEARCH_CUTOFF = 500
DISPLAY_INTERVAL = 50
SEED = 133

class GeneratedDeck(object):
    def __init__(self, seed=1):
        self.seed = seed

    # Generates a predetermined sequence of deck strings with each call
    def getDeckString(self):
        def randomGenerator():
            max_int32 = (1 << 31) - 1
            self.seed = self.seed & max_int32

            while True:
                self.seed = (self.seed * 214013 + 2531011) & max_int32
                yield self.seed >> 16
        
        def deal():
            nc = NUM_CARDS
            cards = list(range(nc - 1, -1, -1))
            rnd = randomGenerator()
            for i, r in zip(range(nc), rnd):
                j = (nc - 1) - r % (nc - i)
                cards[i], cards[j] = cards[j], cards[i]
            return cards

        cards = deal()
        ranks = ["10" if VALID_CARD_RANKS[int(c/4)] == "T" else VALID_CARD_RANKS[int(c/4)] for c in cards]
        suits = [VALID_CARD_SUITS[c%4] for c in cards]
        l = [rank + suit for rank, suit in zip(ranks, suits)]
        return ",".join(l[:])

def state_to_string(state):
    return str(state)

# Search functions (FreecellAI Class)
class SolitaireBoardProblem():

    def __init__(self, deckString):
        self.game = FreecellGame(deckString)
        self.initial = deepcopy(self.game.get_state())

    def actions(self, state):
        # totalPossibilities = ["a","s","d","f","j","k","l",";","u","i","o","p","q","w","e","r"]
        source_locs = self.prune_sources(state)
        valid_actions = []
        for i in source_locs:
            dest_locs = self.prune_destinations(state,i)
            for j in dest_locs:
                action = i + j
                if self.move_test(state, action):
                    valid_actions.append(action)
        return valid_actions

    def prune_sources(self, state):
        sources = []
        free_cells = {
            "q": state['freecells'][0],
            "w": state['freecells'][1],
            "e": state['freecells'][2],
            "r": state['freecells'][3]
        }
        # foundations = {
        #
        # }
        columns = {
            "a": state['columns'][0],
            "s": state['columns'][1],
            "d": state['columns'][2],
            "f": state['columns'][3],
            "j": state['columns'][4],
            "k": state['columns'][5],
            "l": state['columns'][6],
            ";": state['columns'][7]
        }
        # Prune empty locs
        sources.extend([key for key,value in free_cells.items() if value is not None])
        sources.extend([key for key,value in columns.items() if len(value) > 0])
        # Prune foundation locs
        return sources

    def prune_destinations(self, state, source):
        dests = []
        free_cells = {
            "q": state['freecells'][0],
            "w": state['freecells'][1],
            "e": state['freecells'][2],
            "r": state['freecells'][3]
        }
        foundations = {
            "u": state['foundation']['S'],
            "i": state['foundation']['H'],
            "o": state['foundation']['D'],
            "p": state['foundation']['C']
        }
        columns = {
            "a": state['columns'][0],
            "s": state['columns'][1],
            "d": state['columns'][2],
            "f": state['columns'][3],
            "j": state['columns'][4],
            "k": state['columns'][5],
            "l": state['columns'][6],
            ";": state['columns'][7]
        }

        # Prune moves that swap around empty columns
        if source in columns and len(columns[source]) == 1:
            dests.extend([key for key, value in columns.items() if len(value) > 0])
        else:
            dests.extend(["a", "s", "d", "f", "j", "k", "l", ";"])

        # Prune non-suit foundation locs
        dests.extend(["u", "i", "o", "p"])

        # If free-cell loc, prune free-cell locs
        if source not in free_cells:
            dests.extend([key for key,value in free_cells.items() if value is None])

        # Prune free-cell locs if empty column available (?)

        return dests

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """
        self.game.set_state(deepcopy(state))
        # Apply the action to the game
        self.game.move(action)
        new_state = self.game.get_state()
        return new_state

    def goal_test(self, state):
        self.game.set_state(state)
        return self.game.complete()

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def move_test(self, state, action):
        self.game.set_state(deepcopy(state))
        # print("STATE BEFORE MOVE", self.game.get_state())
        ret = False
        try:
            ret = self.game.move(action)
        except:
            ret = False
        return ret


class Node():
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return f"<Node {self.state}"

    def __lt__(self, node):
        # return self.state < node.state
        return True

    def expand(self, problem):
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]
    
    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        return [node.action for node in self.path()[1:]]

    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


class FreecellAI():
    def __init__(self, deck_string):
        self.deck_string = deck_string
        self.problem = SolitaireBoardProblem(deck_string)
        self.ida_count = 0

    def reset_problem(self):
        self.problem = SolitaireBoardProblem(self.deck_string)

    def set_deck_string(self, deck_string):
        self.deck_string = deck_string

    def breadth_first_search(self):
        node = Node(self.problem.initial)
        explored = set()
        if self.problem.goal_test(node.state):
            return (len(explored),node)
        frontier = deque([node])
        while frontier:
            node = frontier.popleft()
            explored.add(state_to_string(node.state))
            for child in node.expand(self.problem):
                if state_to_string(child.state) not in explored and child not in frontier:
                    if self.problem.goal_test(child.state):
                        return (len(explored),child)
                    frontier.append(child)
            # For terminal output
            if len(explored) % DISPLAY_INTERVAL == 0:
                print(f"{len(explored)} nodes explored so far")
            if len(explored) > NODE_SEARCH_CUTOFF:
                print("TIMEOUT: Search cutoff reached")
                break
        return (None,None)

    def depth_first_search(self):
        frontier = [(Node(self.problem.initial))]

        explored = set()
        while frontier:
            node = frontier.pop()
            if self.problem.goal_test(node.state):
                return (len(explored), node)
            explored.add(state_to_string(node.state))
            frontier.extend(child for child in node.expand(self.problem) if state_to_string(child.state) not in explored and child not in frontier)
            # For terminal output
            if len(explored) % DISPLAY_INTERVAL == 0:
                print(f"{len(explored)} nodes explored so far")
            if len(explored) > NODE_SEARCH_CUTOFF:
                print("TIMEOUT: Search cutoff reached")
                break
        return (None,None)

    def a_star_search(self, h):
        f = lambda n: n.path_cost + h(n)
        node = Node(self.problem.initial)
        frontier = PriorityQueue('min', f)
        frontier.append(node)
        explored = set()
        while frontier:
            node = frontier.pop()
            if self.problem.goal_test(node.state):
                return (len(explored),node)
            explored.add(state_to_string(node.state))
            for child in node.expand(self.problem):
                if state_to_string(child.state) not in explored and child not in frontier:
                    frontier.append(child)
                elif child in frontier:
                    if f(child) < frontier[child]:
                        del frontier[child]
                        frontier.append(child)
            # For terminal output
            if len(explored) % DISPLAY_INTERVAL == 0:
                print(f"{len(explored)} nodes explored so far")
            if len(explored) > NODE_SEARCH_CUTOFF:
                print("TIMEOUT: Search cutoff reached")
                break
        return (None,None)
    
    # Simple heuristic based on how many cards are NOT in foundation piles
    def num_not_in_foundation_heuristic(self, node):
        state = node.state
        foundation_cards = sum([len(ls) for ls in state['foundation'].values()])
        return NUM_CARDS - foundation_cards

    # Simple heuristic based on number of well placed cards in columns
    def well_placed_heuristic(self, node):
        state = node.state
        val = NUM_CARDS
        for each in state['columns']:
            prev = None
            prevColor = None
            for i in range(0,len(each)):
                if not prev:
                    prev = each[len(each)-i-1].rank.num
                    prevColor = each[len(each)-i-1].suit.color
                else:
                    if each[len(each)-i-1].rank.num == (prev + 1) and prevColor != each[len(each)-i-1].suit.color:
                        val += 1
                        prev = each[len(each)-i-1].rank.num
                        prevColor = each[len(each)-i-1].suit.color
                    else:
                        break
        val -= sum([len(ls) for ls in state['foundation'].values()])
        return val

    # Simple heuristic based on number of freecells and free columns
    def free_heuristic(self, node):
        state = node.state
        num_free_freecells = sum([1 for value in state['freecells'] if value is None])
        num_free_columns = sum([1 for value in state['columns'] if len(value) == 0])
        foundation_cards = sum([len(ls) for ls in state['foundation'].values()])
        return NUM_CARDS - (num_free_freecells + num_free_columns + foundation_cards)
    
    # Simple heuristic based on difference from the top
    def difference_from_top_heuristic(self, node):
        state = node.state
        foundation_cards = sum([len(ls) for ls in state['foundation'].values()])
        avgOne = 0
        avgTwo = 0
        for each in state['columns']:
            if each:
                avgOne += each[-1].rank.num
        avgOne /= 8
        for each in state['foundation'].values():
            if each:
                avgTwo += each[-1].rank.num
        avgTwo /= 4
        return NUM_CARDS + (avgOne - avgTwo) - foundation_cards

    # Simple heuristic based on lowest home card
    def lowest_home_card_heuristic(self, node):
        state = node.state
        foundation_cards = sum([len(ls) for ls in state['foundation'].values()])
        low = 14
        for each in state['foundation'].values():
            if each:
                low = min(each[-1].rank.num,low)
        if low == 14:
            low = 0
        return NUM_CARDS + low - foundation_cards

    # Simple heuristic based on highest home card
    def highest_home_card_heuristic(self, node):
        state = node.state
        foundation_cards = sum([len(ls) for ls in state['foundation'].values()])
        highest = 0
        for each in state['foundation'].values():
            if each:
                highest = max(each[-1].rank.num,highest)
        return NUM_CARDS - highest - foundation_cards

    # For each foundation pile: Locate within the columns the next card that should be
    # placed there, and count the cards found on top of it. The sum of this count for
    # each foundation is what the heuristic returns. This number is multiplied by 2
    # if there are no available free-cells or there are empty foundation piles.
    def hsd_heuristic(self, node):
        state = node.state
        val = 0
        for key, pile in state['foundation'].items():
            target_card = pile[-1] if len(pile) > 0 else FreecellCard(rank=CardRank('Ace'), suit=CardSuit(key))
            for col in state['columns']:
                for i,card in enumerate(col):
                    if target_card == card:
                        val += len(col) - i
                        break

        # Determine if multiplying value by 2
        multiply = True
        for col in state['columns']:
            if len(col) == 0:
                multiply = False
                break
        if multiply:
            for cell in state['freecells']:
                if cell is None:
                    multiply = False
                    break
        if multiply:
            val *= 2
        return val

    # TODO: Define more heuristics here!
    
    def a_star_search_limited(self, h, limit, search_total):
        self.reset_problem() # Clear memory from previous searches
        f = lambda n: n.path_cost + h(n)
        node = Node(self.problem.initial)
        frontier = PriorityQueue('min', f)
        frontier.append(node)
        explored = set()
        while frontier:
            node = frontier.pop()
            if self.problem.goal_test(node.state):
                return (len(explored),node)
            explored.add(state_to_string(node.state))
            for child in node.expand(self.problem):
                if state_to_string(child.state) not in explored and child not in frontier and f(child) <= limit:
                    frontier.append(child)
                elif child in frontier:
                    if f(child) < frontier[child]:
                        del frontier[child]
                        frontier.append(child)
            # For terminal output
            if len(explored) % DISPLAY_INTERVAL == 0:
                print(f"{len(explored)} nodes explored so far")
                print(f"{search_total + len(explored)} nodes explored so far in TOTAL")
            if len(explored) > NODE_SEARCH_CUTOFF:
                # print("TIMEOUT: Search cutoff reached")
                return (None,None)
        return (len(explored),None)

    def iterative_deepening_astar_search(self, h=None):
        total_searched = 0
        for cost_limit in range(sys.maxsize):
            num_explored, result = self.a_star_search_limited(h, cost_limit, total_searched)
            if result is not None:
                return (total_searched+num_explored,result)
            elif num_explored is None:
                print("TIMEOUT: Search cutoff reached")
                return (None,None)
            else:
                total_searched += num_explored
    
    # Order exploration of nodes with h
    def depth_limited_astar_search(self, f, f_limit=50, depth_limit=NODE_SEARCH_CUTOFF):
        def recursive_dls(node, limit):
            self.ida_count += 1
            if self.ida_count % DISPLAY_INTERVAL == 0:
                print(f"{self.ida_count} nodes explored so far")
            if self.ida_count > NODE_SEARCH_CUTOFF:
                return 'TIMEOUT'
            if self.problem.goal_test(node.state):
                return node
            elif limit == 0:
                return 'TIMEOUT'
            else:
                explore_nodes = PriorityQueue('min', f)
                for child in node.expand(self.problem):
                    if f(child) <= f_limit:
                        explore_nodes.append(child)
                while explore_nodes:
                    node = explore_nodes.pop()
                    result = recursive_dls(node, limit - 1)
                    if result == 'TIMEOUT':
                        return 'TIMEOUT'
                    elif result is not None:
                        return result
                return None

        # Body of depth_limited_search:
        return recursive_dls(Node(self.problem.initial), depth_limit)
    
    def recursive_iterative_deepening_astar_search(self, h=None):
        self.ida_count = 0
        f = lambda n: n.path_cost + h(n)
        for cost_limit in range(sys.maxsize):
            self.reset_problem()
            result = self.depth_limited_astar_search(f, cost_limit)
            if result == 'TIMEOUT':
                print("TIMEOUT: Search cutoff reached")
                return (None,None)
            elif result is not None:
                return (self.ida_count,result)
        return (None,None)




def test_search(ai, algorithm, heuristic=None, name=None):
    ai.reset_problem()
    print(f"Performing {name}...")
    print("-----------------------------")
    start = time.time()
    if heuristic is None:
        states_explored, goal_node = algorithm()
    else:
        states_explored, goal_node = algorithm(heuristic)
    end = time.time()

    solution_length = None
    time_elapsed = end - start
    success = False
    if goal_node is None:
        print("The search failed!")
        if VERBOSE:
            print(f"{time_elapsed} to execute")
        success = False
    else:
        solution_length = goal_node.path_cost # Another performance metric
        print("The search succeeded!")
        # print(f"Initial state: {ai.problem.initial}")
        if VERBOSE:
            print(f"The solution is: {goal_node.solution()}")
            print(f"{solution_length} moves to get a win")
            print(f"{states_explored} states explored")
            print(f"{time_elapsed} to execute")
        success = True
    print("-----------------------------\n")
    return (success, solution_length, states_explored, time_elapsed)


if __name__ == '__main__':
    from sys import argv
    seed = int(argv[1]) if len(argv) >= 2 else SEED # 20
    VERBOSE = bool(int(argv[2])) if len(argv) == 3 else True
    print("Hand {}".format(seed))
    deck1 = GeneratedDeck(seed)
    deckString = deck1.getDeckString()
    print(deckString)
    ai = FreecellAI(deckString)

    test_algorithms = {
        "Depth-first search": (ai.depth_first_search, None),
        "Breadth-first search": (ai.breadth_first_search, None),
        "A* search with foundation heuristic": (ai.a_star_search, ai.num_not_in_foundation_heuristic), 
        "A* search with well-placed heuristic": (ai.a_star_search, ai.well_placed_heuristic),
        "A* search with free heuristic": (ai.a_star_search, ai.free_heuristic),
        "A* search with difference-from-top heuristic": (ai.a_star_search, ai.difference_from_top_heuristic),
        "A* search with lowest-home-card heuristic": (ai.a_star_search, ai.lowest_home_card_heuristic),
        "A* search with highest-home-card heuristic": (ai.a_star_search, ai.highest_home_card_heuristic),
        "IDA* search with foundation heuristic": (ai.iterative_deepening_astar_search, ai.num_not_in_foundation_heuristic),
        "IDA* search with well-placed heuristic": (ai.iterative_deepening_astar_search, ai.well_placed_heuristic),
        "IDA* search with free heuristic": (ai.iterative_deepening_astar_search, ai.free_heuristic),
        "IDA* search with difference-from-top heuristic": (ai.iterative_deepening_astar_search, ai.difference_from_top_heuristic),
        "IDA* search with lowest-home-card heuristic": (ai.iterative_deepening_astar_search, ai.lowest_home_card_heuristic),
        "IDA* search with highest-home-card heuristic": (ai.iterative_deepening_astar_search, ai.highest_home_card_heuristic),
        "Recursive IDA* search with foundation heuristic": (ai.recursive_iterative_deepening_astar_search, ai.num_not_in_foundation_heuristic),
        "Recursive IDA* search with well-placed heuristic": (ai.recursive_iterative_deepening_astar_search, ai.well_placed_heuristic),
        "Recursive IDA* search with free heuristic": (ai.recursive_iterative_deepening_astar_search, ai.free_heuristic),
        "Recursive IDA* search with difference-from-top heuristic": (ai.recursive_iterative_deepening_astar_search, ai.difference_from_top_heuristic),
        "Recursive IDA* search with lowest-home-card heuristic": (ai.recursive_iterative_deepening_astar_search, ai.lowest_home_card_heuristic),
        "Recursive IDA* search with highest-home-card heuristic": (ai.recursive_iterative_deepening_astar_search, ai.highest_home_card_heuristic)
    }

    test_algorithm_metrics = { # (successful_searches, solution_length, states_explored, time_elapsed_successful, time_elapsed_failed)
        "Depth-first search": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "Breadth-first search": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "A* search with foundation heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "A* search with well-placed heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "A* search with free heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "A* search with difference-from-top heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "A* search with lowest-home-card heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "A* search with highest-home-card heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "IDA* search with foundation heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "IDA* search with well-placed heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "IDA* search with free heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "IDA* search with difference-from-top heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "IDA* search with lowest-home-card heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "IDA* search with highest-home-card heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "Recursive IDA* search with foundation heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "Recursive IDA* search with well-placed heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "Recursive IDA* search with free heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "Recursive IDA* search with difference-from-top heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "Recursive IDA* search with lowest-home-card heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        },
        "Recursive IDA* search with highest-home-card heuristic": {
            'successes': 0,
            'solution_length': 0,
            'states_explored': 0,
            'time_elapsed_successful': 0,
            'time_elapsed_failed': 0
        }
    }

    num_iterations = 20

    for i in range(1,num_iterations+1):
        print("=============================")
        print('TESTING ITERATION', i)
        print('Deck configuration:', deckString)
        print("=============================\n")
        for search_name, searches in test_algorithms.items():
            search, heuristic = searches
            success, solution_length, states_explored, time_elapsed = test_search(ai=ai, algorithm=search, heuristic=heuristic, name=search_name)
            if success:
                test_algorithm_metrics[search_name]['successes'] += 1
                test_algorithm_metrics[search_name]['solution_length'] += solution_length
                test_algorithm_metrics[search_name]['states_explored'] += states_explored
                test_algorithm_metrics[search_name]['time_elapsed_successful'] += time_elapsed
            else:
                test_algorithm_metrics[search_name]['time_elapsed_failed'] += time_elapsed
        deckString = deck1.getDeckString()
        ai.set_deck_string(deckString)
        
    
    # Print these to file?
    f = open(f"./results/results_{HIGHEST_CARD_RANK}_card_IDA.txt", 'w')
    sys.stdout = f
    print("=============================")
    print('SEARCH ALGORITHM STATISTICS')
    print('Highest rank in deck:', HIGHEST_CARD_RANK)
    print('Seed:', seed)
    print('Search Cutoff:', NODE_SEARCH_CUTOFF)
    print("=============================\n")

    for search_name, search_statistics in test_algorithm_metrics.items():
        success_rate = (search_statistics['successes']/num_iterations)*100
        successes = search_statistics['successes'] if search_statistics['successes'] > 0 else 1
        failures = (num_iterations-search_statistics['successes']) if (num_iterations-search_statistics['successes']) > 0 else 1 
        print("--------------------------------------------------")
        print(f"*** {search_name} ***")
        print(f"Success Rate: {search_statistics['successes']}/{num_iterations} ({success_rate}%)")
        print(f"Average Solution Length: {search_statistics['solution_length']/successes}")
        print(f"Average States Explored: {search_statistics['states_explored']/successes}")
        print(f"Average Success Time: {search_statistics['time_elapsed_successful']/successes}")
        print(f"Average Failure Time: {search_statistics['time_elapsed_failed']/failures}")
        print("--------------------------------------------------\n")
    f.close()
