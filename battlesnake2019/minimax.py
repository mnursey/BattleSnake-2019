import math

def minimax(state, depth, alpha, beta, maximizingPlayer):

    if depth == 0: # or i'm dead
        return # eval of score.. neural network?

    if maximizingPlayer:
        maxEval = -math.inf

        for substate in state:
            eval = minimax(substate, depth - 1, alpha, beta, False)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)

            if beta <= alpha:
                break
        return maxEval

    else:
        minEval = math.inf

        for substate in state:
            eval = minimax(substate, depth - 1, alpha, beta, True)
            minEval = min(maxEval, eval)
            alpha = min(alpha, eval)

            if beta <= alpha:
                break
        return minEval