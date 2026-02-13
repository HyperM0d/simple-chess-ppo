from agent import ppo_agent, self_play, evaluate, save_agent, play_game
from env import chess_env
from concurrent.futures import ProcessPoolExecutor
import os


# self play training loop
# https://github.com/Zeta36/chess-alpha-zero
# https://stackoverflow.com/questions/53998282/how-does-the-batch-size-affect-the-training-of-a-neural-network
# multiprocessing for parallel games implemented below
def play_single_game(_):
    """play one game and return result - for parallel execution"""
    agent = ppo_agent()
    result = play_game(agent, agent, temperature=0.7)
    return result


def parallel_self_play(num_games=100, num_workers=None):
    """play multiple games in parallel using multiprocessing"""
    if num_workers is None:
        num_workers = os.cpu_count() or 4
    
    print(f"starting parallel self play with {num_workers} workers")
    
    results = {"wins": 0, "losses": 0, "draws": 0}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        game_results = list(executor.map(play_single_game, range(num_games)))
    
    for result in game_results:
        if result == 1:
            results["wins"] += 1
        elif result == -1:
            results["losses"] += 1
        else:
            results["draws"] += 1
    
    print(f"completed {num_games} games | wins {results['wins']} losses {results['losses']} draws {results['draws']}")
    return results


def main():
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    agent = ppo_agent(lr=1e-4, gamma=0.99, eps_clip=0.2, k_epochs=4)
    
    print("starting self play training")
    # use parallel version for faster training
    parallel_self_play(num_games=200, num_workers=4)
    
    # or use sequential version
    # self_play(agent, num_games=200)
    
    print("\nevaluating against random play")
    results = evaluate(agent, num_games=20)
    print(f"results {results}")
    
    save_agent(agent, "chess_ppo_model.pt")


if __name__ == "__main__":
    main()
