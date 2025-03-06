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

import numpy as np
import pygame
import sys

if len(sys.argv) < 2:
    print(f"usage: {sys.argv[0]} <FEN_STRING>", sys.argv[0])
    sys.exit(0)

# constants
FPS                 = 60
WINDOW_DIMS         = (370, 500)
WINDOW_TITLE        = "KV"
EMPTY_BOARD_POS     = (0, 65)
BORDER_COLOR        = "yellow"
BORDER_WIDTH        = 3
TEXT_COLOR          = "white"
SQUARE_X_OFFSET     = 12
SQUARE_Y_OFFSET     = 76
SQUARE_SIDE_LEN     = 45
SQUARES_ON_X_AXIS   = [ pos for pos in range(SQUARE_X_OFFSET, SQUARE_X_OFFSET + SQUARE_SIDE_LEN * 7, SQUARE_SIDE_LEN - 2) ]
SQUARES_ON_Y_AXIS   = [ pos for pos in range(SQUARE_Y_OFFSET, SQUARE_Y_OFFSET + SQUARE_SIDE_LEN * 7, SQUARE_SIDE_LEN - 2) ]
PIECE_DIMS          = (SQUARE_SIDE_LEN, SQUARE_SIDE_LEN)
PIECE_OFFSET        = 2.2
MODEL_ALPHA_URL     = "./models/weights-legal-illegal-10M"
MODEL_BETA_URL      = "./models/weights-mate-nomate-2M"
MODEL_GAMMA_URL     = "./models/weights-white-black-checkmates-2M"
DEFAULT_FONT_NAME   = "arial"
SMALL_FONT_SIZE     = 14
LARGE_FONT_SIZE     = 28

# fp32 representation of the pieces
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

# piece_name -> fp32 representation
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

# parameters
is_model_alpha_selected = True
is_model_beta_selected  = False
is_model_gamma_selected = False

# assets
board_texture           = pygame.image.load("./assets/empty_board.png")
white_pawn_texture      = pygame.transform.scale(pygame.image.load("./assets/white_pawn_processed.png"), PIECE_DIMS)
white_rook_texture      = pygame.transform.scale(pygame.image.load("./assets/white_rook_processed.png"), PIECE_DIMS)
white_knight_texture    = pygame.transform.scale(pygame.image.load("./assets/white_knight_processed.png"), PIECE_DIMS)
white_bishop_texture    = pygame.transform.scale(pygame.image.load("./assets/white_bishop_processed.png"), PIECE_DIMS)
white_queen_texture     = pygame.transform.scale(pygame.image.load("./assets/white_queen_processed.png"), PIECE_DIMS)
white_king_texture      = pygame.transform.scale(pygame.image.load("./assets/white_king_processed.png"), PIECE_DIMS)
black_pawn_texture      = pygame.transform.scale(pygame.image.load("./assets/black_pawn_processed.png"), PIECE_DIMS)
black_rook_texture      = pygame.transform.scale(pygame.image.load("./assets/black_rook_processed.png"), PIECE_DIMS)
black_knight_texture    = pygame.transform.scale(pygame.image.load("./assets/black_knight_processed.png"), PIECE_DIMS)
black_bishop_texture    = pygame.transform.scale(pygame.image.load("./assets/black_bishop_processed.png"), PIECE_DIMS)
black_queen_texture     = pygame.transform.scale(pygame.image.load("./assets/black_queen_processed.png"), PIECE_DIMS)
black_king_texture      = pygame.transform.scale(pygame.image.load("./assets/black_king_processed.png"), PIECE_DIMS)

# piece_name -> texture
piece_textures = { "P": white_pawn_texture,
                   "R": white_rook_texture,
                   "N": white_knight_texture,
                   "B": white_bishop_texture,
                   "Q": white_queen_texture,
                   "K": white_king_texture,
                   "p": black_pawn_texture,
                   "r": black_rook_texture,
                   "n": black_knight_texture,
                   "b": black_bishop_texture,
                   "q": black_queen_texture,
                   "k": black_king_texture }

# initialization
pygame.init()
pygame.display.set_caption(WINDOW_TITLE)
screen = pygame.display.set_mode(WINDOW_DIMS)
clock = pygame.time.Clock()
big_font = pygame.font.SysFont(DEFAULT_FONT_NAME, LARGE_FONT_SIZE)
small_font = pygame.font.SysFont(DEFAULT_FONT_NAME, SMALL_FONT_SIZE)

# generate a tensor out of the given fen string
def create_input_tensor(board_fen):
    offset = 0
    input_tensor = np.zeros((128, 1), dtype=np.float32)
    for c in board_fen:
        if str.isdigit(c):
            c_digit = int(c)
            for i in range(c_digit):
                input_tensor[offset] = SQUARE_EMPTY
                input_tensor[offset + 64] = SQUARE_EMPTY
                offset += 1
        else:
            value = BOARD_MAP.get(c)
            if value is not None:
                input_tensor[offset] = value
                input_tensor[offset + 64] = value
                offset += 1
    return input_tensor

# text rendering
def render_text(screen, font, text, text_color, border_color, is_selected, x, y):
    text_surface = font.render(text, True, text_color)
    screen.blit(text_surface, (x, y))

    if is_selected:
        border_rect = text_surface.get_rect(topleft=(x, y))
        pygame.draw.rect(screen, border_color, border_rect.inflate(8, 8), BORDER_WIDTH)

# render a border around the square over which the mouse is hovering
def render_border_around_square(screen, border_color, mouse_pos):
    x_pos, y_pos = None, None
    for square_x_pos in SQUARES_ON_X_AXIS:
        if mouse_pos[0] > square_x_pos and mouse_pos[0] <= square_x_pos + SQUARE_SIDE_LEN:
            x_pos = square_x_pos
            break
    for square_y_pos in SQUARES_ON_Y_AXIS:
        if mouse_pos[1] > square_y_pos and mouse_pos[1] <= square_y_pos + SQUARE_SIDE_LEN:
            y_pos = square_y_pos
            break

    if x_pos is not None and y_pos is not None:
        pygame.draw.rect(screen, border_color, (x_pos, y_pos, SQUARE_SIDE_LEN, SQUARE_SIDE_LEN), BORDER_WIDTH)

# returns the x,y index of the clicked board position
def get_square_pos(clicked_mouse_pos):
    x_pos, y_pos = None, None
    for i in range(len(SQUARES_ON_X_AXIS)):
        square_x_pos = SQUARES_ON_X_AXIS[i]
        if clicked_mouse_pos[0] > square_x_pos and clicked_mouse_pos[0] <= square_x_pos + SQUARE_SIDE_LEN:
            x_pos = i
            break
    for i in range(len(SQUARES_ON_Y_AXIS)):
        square_y_pos = SQUARES_ON_Y_AXIS[i]
        if clicked_mouse_pos[1] > square_y_pos and clicked_mouse_pos[1] <= square_y_pos + SQUARE_SIDE_LEN:
            y_pos = i
            break
    return y_pos, x_pos

# render the given board
def render_board(screen, fen_string):
    x_pos = SQUARES_ON_X_AXIS[0]
    y_pos = SQUARES_ON_Y_AXIS[0]
    for piece_letter in fen_string:
        if piece_letter.isdigit():
            cnt = int(piece_letter)
            x_pos += cnt * (SQUARE_SIDE_LEN - PIECE_OFFSET)
        elif piece_letter == "/":
            y_pos += SQUARE_SIDE_LEN - PIECE_OFFSET
            x_pos = SQUARES_ON_X_AXIS[0]
        else:
            piece_texture = piece_textures.get(piece_letter)
            if piece_texture is not None:
                screen.blit(piece_texture, (x_pos, y_pos))
                x_pos += SQUARE_SIDE_LEN - PIECE_OFFSET

# generate renderables for the predictions of the selected model
def gen_render_predictions(selected_model, selected_square, board_fen):
    if None in selected_square:
        return []
    input_tensor = create_input_tensor(board_fen)
    selected_square_ind = selected_square[0] * 8 + selected_square[1]
    selected_piece_value = input_tensor[selected_square_ind]
    rects = []
    for i in range(64, 128):
        input_tensor_copy = np.copy(input_tensor)
        input_tensor_copy[64 + selected_square_ind] = SQUARE_EMPTY
        input_tensor_copy[i] = selected_piece_value
        output_tensor = selected_model.inference(input_tensor_copy.reshape(128, 1))
        transparent_surface = pygame.Surface((SQUARE_SIDE_LEN, SQUARE_SIDE_LEN), pygame.SRCALPHA)
        transparent_surface.fill((int(output_tensor[1][0] * 255), int(output_tensor[0][0] * 255), 0, 128))
        rects.append(transparent_surface)
    return rects

# render the predictions as colored rects
def render_predictions(screen, renderable_predictions):
    x_ind = 0
    y_ind = 0
    for renderable in renderable_predictions:
        screen.blit(renderable, (SQUARES_ON_X_AXIS[x_ind], SQUARES_ON_Y_AXIS[y_ind]))
        x_ind += 1
        if x_ind == 8:
            x_ind = 0
            y_ind += 1

# activation function
def logistic(x):
    x = np.clip(x, -10.0, 10.0)
    return 1.0 / (1.0 + np.exp(-x))

# karpathian validator
class KV:
    def __init__(self, weights_url):
        matrix_dims = [(128, 128), (128, 1), (2, 128), (2, 1)]
        with open(weights_url, "rb") as weights_file:
            self.weights_input_hidden = np.fromfile(weights_file, dtype=np.float32, count=matrix_dims[0][0] * matrix_dims[0][1]).reshape(matrix_dims[0])
            self.weights_hidden_output = np.fromfile(weights_file, dtype=np.float32, count=matrix_dims[2][0] * matrix_dims[2][1]).reshape(matrix_dims[2])
            self.bias_hidden = np.fromfile(weights_file, dtype=np.float32, count=matrix_dims[1][0] * matrix_dims[1][1]).reshape(matrix_dims[1])
            self.bias_output = np.fromfile(weights_file, dtype=np.float32, count=matrix_dims[3][0] * matrix_dims[3][1]).reshape(matrix_dims[3])

    def inference(self, input_tensor):
        output_tensor = np.dot(self.weights_input_hidden, input_tensor)
        output_tensor = self.bias_hidden + output_tensor
        output_tensor = logistic(output_tensor)
        output_tensor = np.dot(self.weights_hidden_output, output_tensor)
        output_tensor = self.bias_output + output_tensor
        output_tensor = logistic(output_tensor)
        return output_tensor

model_alpha     = KV(MODEL_ALPHA_URL)
model_beta      = KV(MODEL_BETA_URL)
model_gamma     = KV(MODEL_GAMMA_URL)

selected_model  = model_alpha
renderable_predictions = []

# render loop
isRunning = True
while isRunning:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning = False
        elif event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
            if is_model_alpha_selected:
                is_model_alpha_selected = False
                is_model_beta_selected = True
                selected_model = model_beta
            elif is_model_beta_selected:
                is_model_beta_selected = False
                is_model_gamma_selected = True
                selected_model = model_gamma
            else:
                is_model_gamma_selected = False
                is_model_alpha_selected = True
                selected_model = model_alpha
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            clicked_mouse_pos = pygame.mouse.get_pos()
            selected_square = get_square_pos(clicked_mouse_pos)
            renderable_predictions = gen_render_predictions(selected_model, selected_square, sys.argv[1])

    screen.fill("black")
    screen.blit(board_texture, EMPTY_BOARD_POS)

    render_text(screen, big_font, "model-α", TEXT_COLOR, BORDER_COLOR, is_model_alpha_selected, 10, 15)
    render_text(screen, big_font, "model-β", TEXT_COLOR, BORDER_COLOR, is_model_beta_selected, 130, 15)
    render_text(screen, big_font, "model-γ", TEXT_COLOR, BORDER_COLOR, is_model_gamma_selected, 250, 15)

    render_text(screen, small_font, "Tip #1: Click on one of the squares to see predictions", TEXT_COLOR, BORDER_COLOR, False, 10, WINDOW_DIMS[1] - 50)
    render_text(screen, small_font, "Tip #2: Press space to switch between models", TEXT_COLOR, BORDER_COLOR, False, 10, WINDOW_DIMS[1] - 30)

    render_board(screen, sys.argv[1])

    mouse_pos = pygame.mouse.get_pos()
    render_border_around_square(screen, BORDER_COLOR, mouse_pos)
    render_predictions(screen, renderable_predictions)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
