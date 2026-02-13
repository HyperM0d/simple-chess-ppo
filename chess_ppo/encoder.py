import numpy as np
import chess


# board encoding inspired by alpha zero paper
# https://stackoverflow.com/questions/67480684/chess-board-encoding-for-neural-network
# https://arxiv.org/abs/1712.01815
class board_encoder:
    @staticmethod
    def encode(board):
        planes = np.zeros((18, 8, 8), dtype=np.float32)
        piece_map = board.piece_map()
        for square, piece in piece_map.items():
            row = 7 - (square // 8)
            col = square % 8
            plane = piece.piece_type - 1
            if piece.color == chess.BLACK:
                plane += 6
            planes[plane, row, col] = 1
        
        planes[12] = 1 if board.turn == chess.BLACK else 0
        planes[13] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
        planes[14] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
        planes[15] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
        planes[16] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
        
        if board.ply() > 0:
            planes[17] = min(board.ply(), 100) / 100
        
        return planes
