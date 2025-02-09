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
import chess.engine
import chess.pgn
import os
import numpy as np
import random
import math
import struct
import threading
import time

PGN_FILE_URL        = './lichess/lichess_db_standard_rated_2024-11.pgn'

TRAIN_MOVES_FILE    = './data/train-white-black-checkmates-idx3-float-2M'
TRAIN_LABELS_FILE   = './data/train-white-black-labels-idx2-float-2M'
TEST_MOVES_FILE     = './data/test-white-black-checkmates-idx3-float-2M'
TEST_LABELS_FILE    = './data/test-white-black-labels-idx2-float-2M'
 
TEST_DATA_RATIO     = 0.05
MAX_MOVES           = 2_000_000
 
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
 
def gen_checkmates(max_white_checkmates, max_black_checkmates):
    white_checkmates = set()
    black_checkmates = set()
    with open(PGN_FILE_URL, 'r') as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            elif game.headers['Termination'] != 'Normal':
                continue
            elif not(game.headers['Result'] == '1-0' or game.headers['Result'] == '0-1'):
                continue
            board = game.board()
            prev_board_fen = None
            for move in game.mainline_moves():
                prev_board_fen = str(board.board_fen())
                board.push(move) 
            if board.is_checkmate():
                move_tensor = tensorize_move(prev_board_fen, board.board_fen())
                outcome = board.outcome()
                if outcome.winner and len(white_checkmates) < max_white_checkmates:
                    white_checkmates.add((move_tensor.tobytes(), WHITE_CHECKMATE_LABEL.tobytes()))
                elif not(outcome.winner) and len(black_checkmates) < max_black_checkmates:
                    black_checkmates.add((move_tensor.tobytes(), BLACK_CHECKMATE_LABEL.tobytes()))
                if len(white_checkmates) >= max_white_checkmates and len(black_checkmates) >= max_black_checkmates:
                    return white_checkmates, black_checkmates
            print(f'w = {len(white_checkmates)}, b = {len(black_checkmates)}', end='\r')
    return white_checkmates, black_checkmates
 
def append_training_data_to_file(train_moves_file, moves):
    for move in moves:
        train_moves_file.write(move[0])
    train_moves_file.flush()
    os.fsync(train_moves_file.fileno())
 
def append_training_labels_to_file(train_labels_file, moves):
    for move in moves:
        train_labels_file.write(move[1])
    train_labels_file.flush()
    os.fsync(train_labels_file.fileno())
 
def append_testing_data_to_file(test_moves_file, moves):
    for move in moves:
        test_moves_file.write(move[0])
    test_moves_file.flush()
    os.fsync(test_moves_file.fileno())
 
def append_testing_labels_to_file(test_labels_file, moves):
    for move in moves:
        test_labels_file.write(move[1])
    test_labels_file.flush()
    os.fsync(test_labels_file.fileno())
 
def create_file(file_name, data_size, rows, cols):
    file_ptr = open(file_name, "wb")
    file_ptr.write(struct.pack('>I', int('0x00000D02', 16)))
    file_ptr.write(struct.pack('>I', data_size))
    file_ptr.write(struct.pack('>I', rows))
    file_ptr.write(struct.pack('>I', cols))
    return file_ptr

def main():
    TEST_COUNT          = int(MAX_MOVES * TEST_DATA_RATIO)
    TRAIN_COUNT         = MAX_MOVES - TEST_COUNT

    train_moves_file    = create_file(TRAIN_MOVES_FILE, TRAIN_COUNT, 8, 16)
    train_labels_file   = create_file(TRAIN_LABELS_FILE, TRAIN_COUNT, 1, 2)
    test_moves_file     = create_file(TEST_MOVES_FILE, TEST_COUNT, 8, 16)
    test_labels_file    = create_file(TEST_LABELS_FILE, TEST_COUNT, 1, 2)

    HALF_MAX_MOVES      = MAX_MOVES // 2

    unique_white_checkmates, unique_black_checkmates = gen_checkmates(HALF_MAX_MOVES, HALF_MAX_MOVES)
    print(f'w = {len(unique_white_checkmates)}, b = {len(unique_black_checkmates)}')

    unique_white_checkmates = list(unique_white_checkmates)
    unique_black_checkmates = list(unique_black_checkmates)

    HALF_TEST_COUNT     = int(HALF_MAX_MOVES * TEST_DATA_RATIO)
    HALF_TRAIN_COUNT    = HALF_MAX_MOVES - HALF_TEST_COUNT

    training_white_checkmates = unique_white_checkmates[:HALF_TRAIN_COUNT]
    print(f'training_white_checkmates = {len(training_white_checkmates)}')

    testing_white_checkmates = unique_white_checkmates[HALF_TRAIN_COUNT:]
    print(f'testing_white_checkmates = {len(testing_white_checkmates)}')

    training_black_checkmates = unique_black_checkmates[:HALF_TRAIN_COUNT]
    print(f'training_black_checkmates = {len(training_black_checkmates)}')

    testing_black_checkmates = unique_black_checkmates[HALF_TRAIN_COUNT:]
    print(f'testing_black_checkmates = {len(testing_black_checkmates)}')

    training_checkmates = training_white_checkmates + training_black_checkmates
    print(f'training_checkmates = {len(training_checkmates)}')

    testing_checkmates = testing_white_checkmates + testing_black_checkmates
    print(f'testing_checkmates = {len(testing_checkmates)}')

    random.shuffle(training_checkmates)
    random.shuffle(testing_checkmates)

    append_training_data_to_file(train_moves_file, training_checkmates)
    append_training_labels_to_file(train_labels_file, training_checkmates)

    append_testing_data_to_file(test_moves_file, testing_checkmates)
    append_testing_labels_to_file(test_labels_file, testing_checkmates)

    train_moves_file.close()
    train_labels_file.close()
    test_moves_file.close()
    test_labels_file.close()
    print('Done.')
 
main()
