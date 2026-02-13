import chess
from encoder import board_encoder


# chess library wrapper for game logic
# https://python-chess.readthedocs.io/en/latest/
# https://stackoverflow.com/questions/55876336/is-there-a-way-to-convert-a-python-chess-board-into-a-list-of-integers
class chess_env:
    def __init__(self):
        self.board = chess.Board()
    
    def reset(self):
        self.board = chess.Board()
        return board_encoder.encode(self.board)
    
    def step(self, move):
        self.board.push(move)
        reward = 0
        done = False
        
        if self.board.is_game_over():
            result = self.board.result()
            if result == "1-0":
                reward = 1
            elif result == "0-1":
                reward = -1
            else:
                reward = 0
            done = True
        
        return board_encoder.encode(self.board), reward, done
    
    def get_legal_moves(self):
        return list(self.board.legal_moves)
