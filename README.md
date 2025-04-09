## 🔒 HADES protocol: E2EE Metaverse

This repository contains the source code, the datasets, and other miscellaneous files used in the
[whitepaper](https://hadesprotocol.org/Hades_E2EE_Metaverse_Protocol_v1_Docs.pdf) and the [preprint](https://hadesprotocol.org/HADES_E2EE_Metaverse_Protocol_Paper.pdf).
This README assumes that you've read at least one of them.

The differences between the whitepaper and the preprint are below:

| Whitepaper <img width="441" height="1">  | Preprint <img width="441" height="1"> |
| -------------------------------------- | ----------- |
| Uses the names of historical/mythological ancient Greek figures | Uses the more standard Alice/Bob/Eve to refer to the participants of the protocol  |
| Has philosophical sections  | Computer science only |
| Cites scientific and non-scientific sources  | Cites only peer-reviewed scientific sources  |
| More detailed descriptions of the parts of the protocol, e.g. HTF, LPSs, SPSs, etc.  | Succinct descriptions (in general you may think of the preprint as the abridged version of the whitepaper)  |
| Discusses E2EE ancestor simulations  | Doesn't mention [Bostrom's work](https://simulation-argument.com)  |

The datasets are sourced from [Lichess](https://database.lichess.org).

## 🔨 Building

First, clone the repository:

```
git clone https://github.com/saccharineboi/Hades.git
```

After cloning, `cd` in the git directory and create a virtual environment:

```
python -m venv .
```

Then install the dependencies:

```
pip install -r requirements.txt
```

Before you continue, you must uncompress the files in the `data/` and the `lichess_tensorized/` directories:

```
unxz data/*.xz
unxz lichess_tensorized/*.xz
```

To compile the C files you may use GCC:

```
gcc -std=c99 -Wall -Wextra -O3 -march=native -s $C_FILE -lm
```

Replace `$C_FILE` with one of the C source files, e.g. `train_legal_illegal.c`.

The C++ files (`ckks-*.cpp`) can be compiled using [CMakeLists.User.txt](https://github.com/openfheorg/openfhe-development/blob/main/CMakeLists.User.txt)
provided by the [OpenFHE](https://github.com/openfheorg/openfhe-development) library.

Alternatively, you may simply replace the contents of [simple-real-numbers.cpp](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/simple-real-numbers.cpp)
with one of `ckks-*.cpp`. In either case, you will need to compile the OpenFHE library with flags `-DWITH_NATIVEOPT=ON` and `-DWITH_TCM=ON`.

## 🤖 Models

Each model is a multi-layer perceptron (MLP) with 128 neurons in the input layer, 128 neurons in the hidden layer, and 2 neurons in the output layer.
All models use the logistic function for activations and MSE as the loss function.

- `models/weights-legal-illegal-10M`: binary classifier for legal/illegal chess moves
- `models/weights-mate-nomate-2M`: binary classifier for checkmate/no-checkmate moves
- `models/weights-white-black-checkmates-2M`: binary classifier for white/black checkmates

### weights-legal-illegal-10M

This is a binary classifier trained to classify legal and illegal chess moves (10M stands for 10 million training samples),
also referred to as the `model-alpha` in both the whitepaper and the preprint.
The associated `weights-legal-illegal-log-10M` is a text file containing the training log.
Its accuracy and loss on [lichess databases](https://database.lichess.org/) is show below:


| Database <img width="441" height="1">  | Accuracy <img width="441" height="1"> | Loss <img width="441" height="1"> |
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
also referred to as the `model-beta` in both the whitepaper and the preprint.
The associated `weights-mate-nomate-log-2M` is a text file containing the training log.
Its accuracy and loss on [lichess databases](https://database.lichess.org/) is show below:

| Database <img width="441" height="1"> | Accuracy <img width="441" height="1"> | Loss <img width="441" height="1"> |
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
also referred to as the `model-gamma` in both the whitepaper and the preprint.
The associated `weights-white-black-checkmates-log-2M` is a text file containing the training log.
Its accuracy and loss on [lichess databases](https://database.lichess.org/) is show below:

| Database <img width="441" height="1"> | Accuracy <img width="441" height="1"> | Loss <img width="441" height="1"> |
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

The training samples are generated from `lichess_db_standard_rated_2024-11.pgn`. Following python scripts are used to extract the training samples:

- `gen_legal_and_illegal_moves_lichess.py`: Extracts legal moves from a given database, and modifies some of the extracted legal moves to generate illegal moves
- `gen_mate_and_nomate_moves_lichess.py`: Extracts checkmates and non-checkmate moves from a given database
- `gen_white_and_black_checkmates_lichess.py`: Extracts white and black checkmates from a given database

Their compressed outputs are saved in the `data` directory. You must uncompress them before use.

## 🔬 Testing Datasets

The testing datasets are extracted from lichess databases from January to June of 2013. Following python scripts are used to extract the testing samples:

- `tensorize_lichess_legal_illegal.py`: Extracts legal moves from a given database (does not modify them to generate illegal moves)
- `tensorize_lichess_mate_nomate.py`: Extracts checkmates and non-checkmates from a given database
- `tensorize_lichess_white_black_checkmates.py`: Extracts white and black checkmates from a given database

Their compressed outputs are saved in the `lichess_tensorized` directory. You must decompress them before use.

## 👁 Renderers

These scripts have been used to generate some of the figures in the papers.

- `render_piece.py`: Renders a chess piece given its symbol
- `render_board.py`: Renders a chess board given its position as a [FEN string](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)
- `graph.py`: Produces a graph from a given training log (e.g. `weights-legal-illegal-log-10M`)
- `KV.py`: Visualizes the predictions made by the models

The white pieces generated by `render_piece.py` are:

<img src="/assets/white_pawn.png" style="width: 100px; height: 100px; display: inline-block;"> <img src="/assets/white_rook.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/white_knight.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/white_bishop.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/white_queen.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/white_king.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;">

And the black pieces:

<img src="/assets/black_pawn.png" style="width: 100px; height: 100px; display: inline-block;"> <img src="/assets/black_rook.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/black_knight.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/black_bishop.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/black_queen.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;"> <img src="/assets/black_king.png" style="width: 100px; height: 100px; display: inline-block; margin-left: 10px;">

`render_board.py` generates an image of a chess board given its FEN string representation.
E.g. the string 'r1bqkbnr/pp1p1ppp/2n1p3/2p5/3P1B2/5N2/PPP1PPPP/RN1QKB1R' corresponds to the image below:

![Generated chess board](/assets/example_board.png)

`KV.py` visualizes the predictions made by the models, e.g. the below
figure shows `model-alpha`'s predictions for the white knight on G1,
correctly predicting the legal squares that the knight can move to.

![Visualization of one of the predictions made by KV.py](/assets/example_pred_vis.png)

The meanings of the colors depend on the model:

- `model-alpha`: Greener values indicate higher confidence in move's legality, redder values indicate higher confidence in move's illegality
- `model-beta`: Greener values indicate higher confidence in move resulting in a checkmate, redder values indicate higher confidence in move not resulting in a checkmate (note that `model-beta` assumes that the move is legal)
- `model-gamma`: Greener values indicate higher confidence in move being a checkmate by white, redder values indicate higher confidence in move being a checkmate by black (note that `model-gamma` assumes that the move is legal and results in a checkmate)

## 🔒 Private Inference via CKKS

The source files below are used to test inference on encrypted input via FHE (namely, the [RNS version of the CKKS scheme](https://link.springer.com/chapter/10.1007/978-3-030-10970-7_16)):

- `ckks-legal-illegal.cpp`: Runs private inference for `model-alpha`
- `ckks-mate-nomate.cpp`: Runs private inference for `model-beta`
- `ckks-white-black-checkmates.cpp`: Runs private inference for `model-gamma`

They are all modified versions of the same [source file](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/simple-real-numbers.cpp)
in the examples directory of the [OpenFHE](https://github.com/openfheorg/openfhe-development) library and carry the original license.

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
