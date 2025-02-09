# Copyright 2025 Omar Huseynov
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import chess
import chess.pgn
import os
import numpy as np
import random
import math
import struct
import time
import re
import io

PGN_FILE            = './lichess/lichess_db_standard_rated_2013-06.pgn'
TEST_MOVES_FILE     = './lichess_tensorized/lichess-2013-06-moves-idx3-float-white-black-checkmates'
TEST_LABELS_FILE    = './lichess_tensorized/lichess-2013-06-labels-idx2-float-white-black-checkmates'

SQUARE_EMPTY        =   0.0
SQUARE_WHITE_PAWN   =   3.0  / 8.0
SQUARE_WHITE_ROOK   =   4.0  / 8.0
SQUARE_WHITE_KNIGHT =   5.0  / 8.0
SQUARE_WHITE_BISHOP =   6.0  / 8.0
SQUARE_WHITE_QUEEN  =   7.0  / 8.0
SQUARE_WHITE_KING   =   8.0  / 8.0
SQUARE_BLACK_PAWN   =  -3.0  / 8.0
SQUARE_BLACK_ROOK   =  -4.0  / 8.0
SQUARE_BLACK_KNIGHT =  -5.0  / 8.0
SQUARE_BLACK_BISHOP =  -6.0  / 8.0
SQUARE_BLACK_QUEEN  =  -7.0  / 8.0
SQUARE_BLACK_KING   =  -8.0  / 8.0

BOARD_MAP = { 'P' : SQUARE_WHITE_PAWN,
              'p' : SQUARE_BLACK_PAWN,
              'N' : SQUARE_WHITE_KNIGHT,
              'n' : SQUARE_BLACK_KNIGHT,
              'B' : SQUARE_WHITE_BISHOP,
              'b' : SQUARE_BLACK_BISHOP,
              'R' : SQUARE_WHITE_ROOK,
              'r' : SQUARE_BLACK_ROOK,
              'Q' : SQUARE_WHITE_QUEEN,
              'q' : SQUARE_BLACK_QUEEN,
              'K' : SQUARE_WHITE_KING,
              'k' : SQUARE_BLACK_KING }
 
WHITE_CHECKMATE_LABEL = np.array([ 0.99, 0.01 ], dtype=np.float32)
BLACK_CHECKMATE_LABEL = np.array([ 0.01, 0.99 ], dtype=np.float32)

def tensorize_board_fen(move_tensor, board_fen, offset):
    for c in board_fen:
        if str.isdigit(c):
            c_digit = int(c)
            for i in range(c_digit):
                move_tensor[offset] = SQUARE_EMPTY
                offset += 1
        else:
            value = BOARD_MAP.get(c)
            if value is not None:
                move_tensor[offset] = value
                offset += 1
 
def tensorize_move(prev_board_fen, current_board_fen):
    move_tensor = np.zeros(128, dtype=np.float32)
    tensorize_board_fen(move_tensor, prev_board_fen, 0)
    tensorize_board_fen(move_tensor, current_board_fen, 64)
    return move_tensor

def write_testing_data_to_file(testing_data):
    with open(TEST_MOVES_FILE, "wb") as test_moves_file:
        test_moves_file.write(struct.pack('>I', int('0x00000D02', 16)))
        test_moves_file.write(struct.pack('>I', len(testing_data)))
        test_moves_file.write(struct.pack('>I', 8))
        test_moves_file.write(struct.pack('>I', 16))
        for sample in testing_data:
            test_moves_file.write(sample[0].tobytes())
 
def write_testing_labels_to_file(testing_data):
    with open(TEST_LABELS_FILE, "wb") as test_labels_file:
        test_labels_file.write(struct.pack('>I', int('0x00000D02', 16)))
        test_labels_file.write(struct.pack('>I', len(testing_data)))
        test_labels_file.write(struct.pack('>I', 1))
        test_labels_file.write(struct.pack('>I', 2))
        for sample in testing_data:
            test_labels_file.write(sample[1].tobytes())

def main():
    all_moves = []
    pgn = open(PGN_FILE)
    game_cnt = 0
    white_checkmate_cnt = 0
    black_checkmate_cnt = 0

    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        elif game.headers.get('FEN') is not None or game.headers['Termination'] != 'Normal':
            continue
        elif not(game.headers['Result'] == '1-0' or game.headers['Result'] == '0-1'):
            continue
        game_cnt += 1
        board = chess.Board()
        for move in game.mainline_moves():
            prev_pos = str(board.board_fen())
            board.push(move)
            if board.is_checkmate():
                move_tensor = tensorize_move(prev_pos, board.board_fen())
                outcome = board.outcome()
                if outcome.winner:
                    all_moves.append((move_tensor, WHITE_CHECKMATE_LABEL))
                    white_checkmate_cnt += 1
                else:
                    all_moves.append((move_tensor, BLACK_CHECKMATE_LABEL))
                    black_checkmate_cnt += 1
                print(f'game = {game_cnt}, moves = {len(all_moves)}, w = {white_checkmate_cnt}, b = {black_checkmate_cnt}', end='\r')

    print('')
    print(f'game = {game_cnt}, moves = {len(all_moves)}, w = {white_checkmate_cnt}, b = {black_checkmate_cnt}')

    write_testing_data_to_file(all_moves)
    write_testing_labels_to_file(all_moves)
    print('Done')

main()
