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

TRAIN_MOVES_FILE    = './data/train-mate-nomate-moves-idx3-float-2M-lichess'
TRAIN_LABELS_FILE   = './data/train-mate-nomate-labels-idx2-float-2M-lichess'
TEST_MOVES_FILE     = './data/test-mate-nomate-moves-idx3-float-2M-lichess'
TEST_LABELS_FILE    = './data/test-mate-nomate-labels-idx2-float-2M-lichess'
 
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
 
MATE_MOVE_LABEL        = np.array([ 0.99, 0.01 ], dtype=np.float32)
NOMATE_MOVE_LABEL      = np.array([ 0.01, 0.99 ], dtype=np.float32)
 
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
 
def gen_mate_moves(max_mate_moves):
    all_moves = set()
    white_checkmate_count = 0
    black_checkmate_count = 0
    half_max_mate_moves = max_mate_moves // 2
    with open(PGN_FILE_URL, 'r') as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            elif game.headers.get('FEN') is not None or game.headers['Termination'] != 'Normal':
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
                if outcome.winner and white_checkmate_count < half_max_mate_moves:
                    prev_len = len(all_moves)
                    all_moves.add((move_tensor.tobytes(), MATE_MOVE_LABEL.tobytes()))
                    if len(all_moves) > prev_len:
                        white_checkmate_count += 1
                elif not(outcome.winner) and black_checkmate_count < half_max_mate_moves:
                    prev_len = len(all_moves)
                    all_moves.add((move_tensor.tobytes(), MATE_MOVE_LABEL.tobytes()))
                    if len(all_moves) > prev_len:
                        black_checkmate_count += 1
                if len(all_moves) >= max_mate_moves:
                    return all_moves, white_checkmate_count, black_checkmate_count
            print(f'w = {white_checkmate_count}, b = {black_checkmate_count}, unique_mate_moves = {len(all_moves)}', end='\r')
    return all_moves, white_checkmate_count, black_checkmate_count
 
def gen_nomate_moves(max_nomate_moves):
    all_moves = set()
    with open(PGN_FILE_URL, 'r') as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            elif game.headers.get('FEN') is not None or game.headers['Termination'] != 'Normal':
                continue
            board = game.board()
            for move in game.mainline_moves():
                prev_board_fen = str(board.board_fen())
                board.push(move)
                if not(board.is_checkmate()):
                    move_tensor = tensorize_move(prev_board_fen, board.board_fen())
                    all_moves.add((move_tensor.tobytes(), NOMATE_MOVE_LABEL.tobytes()))
                    if len(all_moves) >= max_nomate_moves:
                        return all_moves
                    print(f'unique_nomate_moves = {len(all_moves)}', end='\r')
    return all_moves
 
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

    unique_nomate_moves = gen_nomate_moves(HALF_MAX_MOVES)
    print(f'unique_nomate_moves = {len(unique_nomate_moves)}')

    unique_mate_moves, white_checkmate_count, black_checkmate_count   = gen_mate_moves(HALF_MAX_MOVES)
    print(f'w = {white_checkmate_count}, b = {black_checkmate_count}, unique_mate_moves = {len(unique_mate_moves)}')

    unique_nomate_moves = list(unique_nomate_moves)
    unique_mate_moves   = list(unique_mate_moves)

    HALF_TEST_COUNT     = int(HALF_MAX_MOVES * TEST_DATA_RATIO)
    HALF_TRAIN_COUNT    = HALF_MAX_MOVES - HALF_TEST_COUNT

    training_nomate_moves = unique_nomate_moves[:HALF_TRAIN_COUNT]
    print(f'training_nomate_moves = {len(training_nomate_moves)}')

    testing_nomate_moves = unique_nomate_moves[HALF_TRAIN_COUNT:]
    print(f'testing_nomate_moves = {len(testing_nomate_moves)}')

    training_mate_moves = unique_mate_moves[:HALF_TRAIN_COUNT]
    print(f'training_mate_moves = {len(training_mate_moves)}')

    testing_mate_moves = unique_mate_moves[HALF_TRAIN_COUNT:]
    print(f'testing_mate_moves = {len(testing_mate_moves)}')

    training_moves = training_nomate_moves + training_mate_moves
    print(f'training_moves = {len(training_moves)}')

    testing_moves = testing_nomate_moves + testing_mate_moves
    print(f'testing_moves = {len(testing_moves)}')

    random.shuffle(training_moves)
    random.shuffle(testing_moves)

    append_training_data_to_file(train_moves_file, training_moves)
    append_training_labels_to_file(train_labels_file, training_moves)

    append_testing_data_to_file(test_moves_file, testing_moves)
    append_testing_labels_to_file(test_labels_file, testing_moves)

    train_moves_file.close()
    train_labels_file.close()
    test_moves_file.close()
    test_labels_file.close()
    print('Done.')
 
main()
