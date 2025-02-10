## 🔒 Hades Protocol

This repository contains the source code, scripts, and the datasets used in [E2EE Metaverse](https://hadesprotocol.org/whitepaper.pdf).

## ℹ️ Introduction

The purpose of the Hades protocol is two-fold:

1. Improve the privacy of the users in [Metaverse](https://en.wikipedia.org/wiki/Metaverse),
2. Improve the interoperability between the different implementations of the protocol.

It achieves the above via the methods outlined below:

1. Establish an end-to-end encrypted tunnel between every user, via ***Local Programmable States***,
2. Enable adversarial machines to perform computations on encrypted data, via ***Shared Programmable States***,
3. Introduce a standardized format to store and exchange 3D data, via ***Hadean Transmission Format***.

For an in-depth discussion of the methods see the [whitepaper](https://hadesprotocol.org/whitepaper.pdf).
Below are the models, the datasets, and various source files used/mentioned in the paper:

## 🤖 Models

Each model is a multi-layer perceptron with 128 neurons in the input layer, 128 neurons in the hidden layer, and 2 neurons in the output layer.
All models use the logistic function for activations and MSE as the loss function.

- `models/weights-legal-illegal-10M`: binary classifier for legal/illegal chess moves
- `models/weights-mate-nomate-2M`: binary classifier for checkmate/no-checkmate moves
- `models/weights-white-black-checkmates-2M`: binary classifier for white/black checkmates

### weights-legal-illegal-10M

This is a binary classifier trained to classify legal and illegal chess moves (10M stands for 10 million training samples),
also referred to as the `model-alpha` in the [whitepaper](https://hadesprotocol.org/whitepaper.pdf).
The associated `weights-legal-illegal-log-10M` is a text file containing the training log.
Its accuracy and loss on [lichess databases](https://database.lichess.org/) is show below:

| Database                               | Accuracy    | Loss     |
| -------------------------------------- | ----------- | -------- |
| lichess_db_standard_rated_2013-01.pgn  | 93.813515%  | 0.093202 |
| lichess_db_standard_rated_2013-02.pgn  | 93.820229%  | 0.092993 |
| lichess_db_standard_rated_2013-03.pgn  | 93.843567%  | 0.092535 |
| lichess_db_standard_rated_2013-04.pgn  | 93.875481%  | 0.092033 |
| lichess_db_standard_rated_2013-05.pgn  | 93.874863%  | 0.091946 |
| lichess_db_standard_rated_2013-06.pgn  | 93.907936%  | 0.091325 |

`train_legal_illegal.c` is used for the training, `test_legal_illegal.c` is used for the testing.

Below is the graph of the training run for the above model:

![Training run of model-alpha](/assets/model-alpha.png)

### weights-mate-nomate-2M

This is a binary classifier trained to classify checkmates and non-checkmates (2M stands for 2 million training samples),
also referred to as the `model-beta` in the [whitepaper](https://hadesprotocol.org/whitepaper.pdf).
The associated `weights-mate-nomate-log-2M` is a text file containing the training log.
Its accuracy and loss on [lichess databases](https://database.lichess.org/) is show below:

| Database                               | Accuracy    | Loss     |
| -------------------------------------- | ----------- | -------- |
| lichess_db_standard_rated_2013-01.pgn  | 92.539078%  | 0.126696 |
| lichess_db_standard_rated_2013-02.pgn  | 92.611618%  | 0.125757 |
| lichess_db_standard_rated_2013-03.pgn  | 92.722206%  | 0.122951 |
| lichess_db_standard_rated_2013-04.pgn  | 92.682701%  | 0.123651 |
| lichess_db_standard_rated_2013-05.pgn  | 92.729904%  | 0.122439 |
| lichess_db_standard_rated_2013-06.pgn  | 92.860413%  | 0.119425 |

`train_mate_nomate.c` is used for the training, `test_mate_nomate.c` is used for the testing.

Below is the graph of the training run for the above model:

![Training run of model-alpha](/assets/model-beta.png)

### weights-white-black-checkmates-2M

This is a binary classifier trained to classify checkmates by white and black (2M stands for 2 million training samples),
also referred to as the `model-gamma` in the [whitepaper](https://hadesprotocol.org/whitepaper.pdf).
The associated `weights-white-black-checkmates-log-2M` is a text file containing the training log.
Its accuracy and loss on [lichess databases](https://database.lichess.org/) is show below:

| Database                               | Accuracy    | Loss     |
| -------------------------------------- | ----------- | -------- |
| lichess_db_standard_rated_2013-01.pgn  | 99.309647%  | 0.011806 |
| lichess_db_standard_rated_2013-02.pgn  | 99.340698%  | 0.011336 |
| lichess_db_standard_rated_2013-03.pgn  | 99.334076%  | 0.011727 |
| lichess_db_standard_rated_2013-04.pgn  | 99.258339%  | 0.012468 |
| lichess_db_standard_rated_2013-05.pgn  | 99.259995%  | 0.012195 |
| lichess_db_standard_rated_2013-06.pgn  | 99.246025%  | 0.012457 |

`train_white_black_checkmate.c` is used for the training, `test_white_black_checkmate.c` is used for the testing.

Below is the graph of the training run for the above model:

![Training run of model-alpha](/assets/model-gamma.png)

## 📖 Training Datasets

The training samples are from `lichess_db_standard_rated_2024-11.pgn`. Following python files are used to extract the training samples:

- `gen_legal_and_illegal_moves_lichess.py`: Extracts legal moves from a given database, and modifies some of the extracted legal moves to generate illegal moves
- `gen_mate_and_nomate_moves_lichess.py`: Extracts checkmates and non-checkmate moves from a given database
- `gen_white_and_black_checkmates_lichess.py`: Extracts white and black checkmates from a given database

Their compressed outputs are saved in the `data` directory. You must decompress them before use.

## 🔬 Testing Datasets

The testing datasets are extracted from lichess databases from January to June of 2013. Following python files are used to extract the testing samples:

- `tensorize_lichess_legal_illegal.py`: Extracts legal moves from a given database (does not modify them to generate illegal moves)
- `tensorize_lichess_mate_nomate.py`: Extracts checkmates and non-checkmates from a given database
- `tensorize_lichess_white_black_checkmates.py`: Extracts white and black checkmates from a given database

Their compressed outputs are saved in the `lichess_tensorized` directory. You must decompress them before use.

## 👁 Renderers

These scripts have been used to generate some of the images in the [whitepaper](https://hadesprotocol.org/whitepaper.pdf).

- `render_piece.py`: Renders a chess piece given its symbol
- `render_board.py`: Renders a chess board given its position as a [FEN string](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)
- `graph.py`: Produces a graph from a given training log (e.g. `weights-legal-illegal-log-10M`)

The white pieces generated by `render_piece.py` are:

<img src="/assets/white_pawn.png" style="width: 100px; height: 100px; display: inline-block;"> <img src="/assets/white_rook.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/white_knight.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/white_bishop.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/white_queen.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/white_king.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;">

And the black pieces:

<img src="/assets/black_pawn.png" style="width: 100px; height: 100px; display: inline-block;"> <img src="/assets/black_rook.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/black_knight.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/black_bishop.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/black_queen.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/black_king.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;">

`render_board.py` generates an image of a chess board given its FEN string representation.
E.g. the string 'r1bqkbnr/pp1p1ppp/2n1p3/2p5/3P1B2/5N2/PPP1PPPP/RN1QKB1R' corresponds to the image below:

![Generated chess board](/assets/example_board.png)

## 🔒 Private Inference via CKKS

The source files below are used to test inference on encrypted input:

- `ckks-legal-illegal.cpp`: Runs private inference for `model-alpha`
- `ckks-mate-nomate.cpp`: Runs private inference for `model-beta`
- `ckks-white-black-checkmates.cpp`: Runs private inference for `model-gamma`

They are all modified versions of the same [source file](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/simple-real-numbers.cpp)
in the examples directory of the [OpenFHE](https://github.com/openfheorg/openfhe-development) library and carry the original license.

## 🔨 Building

After cloning, `cd` in the git directory and create a virtual environment:

```
python -m venv .
```

Then install the dependencies:

```
pip install -r requirements.txt
```

The C files (`train_*.c` and `test_*.c`) can be compiled with `gcc -std=c99 -Wall -Wextra -O3 -march=native -s $FILE.c -o $FILE -lm`,
they have no dependencies, other than the standard library.

The C++ files (`ckks-*.cpp`) can be compiled using [CMakeLists.User.txt](https://github.com/openfheorg/openfhe-development/blob/main/CMakeLists.User.txt)
provided by the [OpenFHE](https://github.com/openfheorg/openfhe-development) library.

## ©️ License

All source files are [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) except the ckks-*.cpp files,
which are licensed under the [2-Clause BSD License](https://opensource.org/license/bsd-2-clause).

Copyright 2025 Omar Huseynov
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
