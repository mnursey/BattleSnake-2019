import math

class MCTS():

    score_map = {}
    sim_count_map = {}
    child_map = {}
    parent_map = {}
    possiblities_map = {}
    explored_possibilities_map = {}
    state_map = {}

    c = 0
    next_node_id = 0

    # values are dependent
    explore_score = 0.825
    search_games = 1

    def __init__(self, c = 1.41):
        self.c = c
        return

    def UBC1(self, child_id, parent_id):
        return (score_map[child_id] / sim_count_map[child_id] + self.c * math.sqrt(math.log(sim_count_map[parent_id]) / sim_count_map[child_id])) if child_id in sim_count_map else explore_score

    def selection(self):

        current_id = 0

        while True:

            # if not explored return
            if not current_id in possiblities_map[current_id]:
                return current_id
               
            # set current id to child with best UBC1 score :)
            best_id = 0
            best_score = -math.inf
            for child_id in possiblities_map[current_id]:
                score = self.UBC1(child_id, current_id)
                if best_score < score:
                    best_score = score
                    best_id = child_id

            current_id = best_id

        return child_id

    def get_possibilities(self, state):
        return

    def expansion(self, id, parent_state, possibility):
        possiblities_map[id] = get_possibilities(state_map[id])
        num_possibilities = len(possiblities_map[id])
        explored_possibilities_map[id] = [False for i in range(num_possibilities)]
        return

    def simulation(self):
        return

    def backpropagation(self):
        return

    def setup(self, state):
        return

    def run(self):
        return

    def restart(self):
        return
