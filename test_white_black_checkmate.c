// Copyright 2025 Omar Huseynov
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the “Software”), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

////////////////////////////////////////
////////////// Constants ///////////////
////////////////////////////////////////

#define PI                              3.1415926f
#define LOG_LINE_LEN                    1000

#define LICHESS_2013_01_MOVES           "./lichess_tensorized/lichess-2013-01-moves-idx3-float-white-black-checkmates"
#define LICHESS_2013_02_MOVES           "./lichess_tensorized/lichess-2013-02-moves-idx3-float-white-black-checkmates"
#define LICHESS_2013_03_MOVES           "./lichess_tensorized/lichess-2013-03-moves-idx3-float-white-black-checkmates"
#define LICHESS_2013_04_MOVES           "./lichess_tensorized/lichess-2013-04-moves-idx3-float-white-black-checkmates"
#define LICHESS_2013_05_MOVES           "./lichess_tensorized/lichess-2013-05-moves-idx3-float-white-black-checkmates"
#define LICHESS_2013_06_MOVES           "./lichess_tensorized/lichess-2013-06-moves-idx3-float-white-black-checkmates"

#define LICHESS_2013_01_LABELS          "./lichess_tensorized/lichess-2013-01-labels-idx2-float-white-black-checkmates"
#define LICHESS_2013_02_LABELS          "./lichess_tensorized/lichess-2013-02-labels-idx2-float-white-black-checkmates"
#define LICHESS_2013_03_LABELS          "./lichess_tensorized/lichess-2013-03-labels-idx2-float-white-black-checkmates"
#define LICHESS_2013_04_LABELS          "./lichess_tensorized/lichess-2013-04-labels-idx2-float-white-black-checkmates"
#define LICHESS_2013_05_LABELS          "./lichess_tensorized/lichess-2013-05-labels-idx2-float-white-black-checkmates"
#define LICHESS_2013_06_LABELS          "./lichess_tensorized/lichess-2013-06-labels-idx2-float-white-black-checkmates"

#define LICHESS_2013_01_SIZE            34475U
#define LICHESS_2013_02_SIZE            35492U
#define LICHESS_2013_03_SIZE            44149U
#define LICHESS_2013_04_SIZE            43551U
#define LICHESS_2013_05_SIZE            49594U
#define LICHESS_2013_06_SIZE            59551U

#define TEST_DATA_FILE_URL              LICHESS_2013_06_MOVES
#define TEST_LABEL_FILE_URL             LICHESS_2013_06_LABELS
#define TEST_SAMPLE_COUNT               LICHESS_2013_06_SIZE

#define MODEL_FILE_URL                  "./models/weights-white-black-checkmates-2M"

#define INPUT_NEURON_COUNT              128
#define HIDDEN_NEURON_COUNT             128
#define OUTPUT_NEURON_COUNT             2

#define WEIGHTS_INPUT_HIDDEN_COUNT      ((HIDDEN_NEURON_COUNT) * (INPUT_NEURON_COUNT))
#define WEIGHTS_HIDDEN_OUTPUT_COUNT     ((OUTPUT_NEURON_COUNT) * (HIDDEN_NEURON_COUNT))

#define SQUARE_EMPTY                    0.0f
#define SQUARE_WHITE_PAWN               3.0f / 8.0f
#define SQUARE_WHITE_ROOK               4.0f / 8.0f
#define SQUARE_WHITE_KNIGHT             5.0f / 8.0f
#define SQUARE_WHITE_BISHOP             6.0f / 8.0f
#define SQUARE_WHITE_QUEEN              7.0f / 8.0f
#define SQUARE_WHITE_KING               8.0f / 8.0f
#define SQUARE_BLACK_PAWN              -3.0f / 8.0f
#define SQUARE_BLACK_ROOK              -4.0f / 8.0f
#define SQUARE_BLACK_KNIGHT            -5.0f / 8.0f
#define SQUARE_BLACK_BISHOP            -6.0f / 8.0f
#define SQUARE_BLACK_QUEEN             -7.0f / 8.0f
#define SQUARE_BLACK_KING              -8.0f / 8.0f

#define WHITE_CHECKMATE_LABEL           0
#define BLACK_CHECKMATE_LABEL           1

////////////////////////////////////////
////////////// Dataset /////////////////
////////////////////////////////////////

struct dataset {
    uint32_t test_count;
    float* test_data;
    float* test_label;
};
typedef struct dataset dataset_t;

static FILE* open_file(const char* restrict path,
                       const char* restrict mode)
{
    FILE* file = fopen(path, mode);
    if (!file) {
        fprintf(stderr, "open_file: failed to open '%s' (mode: '%s')\n", path, mode);
        exit(EXIT_FAILURE);
    }
    return file;
}

static void* allocate_mem(size_t size)
{
    void* addr = malloc(size);
    if (!addr) {
        perror("allocate_mem");
        exit(EXIT_FAILURE);
    }
    return addr;
}

static dataset_t load_dataset()
{
    FILE* test_data_file    = open_file(TEST_DATA_FILE_URL, "rb");
    FILE* test_label_file   = open_file(TEST_LABEL_FILE_URL, "rb");

    fseek(test_data_file, 16, SEEK_SET);
    fseek(test_label_file, 16, SEEK_SET);

    dataset_t data = {};
    data.test_data     = allocate_mem(TEST_SAMPLE_COUNT * INPUT_NEURON_COUNT * sizeof(float));
    data.test_label    = allocate_mem(TEST_SAMPLE_COUNT * OUTPUT_NEURON_COUNT * sizeof(float));

    size_t test_data_items_read = fread(data.test_data,
                                        sizeof(float),
                                        TEST_SAMPLE_COUNT * INPUT_NEURON_COUNT,
                                        test_data_file);
    printf("read %lu items (%lu smaples) from %s\n", test_data_items_read,
           test_data_items_read / INPUT_NEURON_COUNT,
           TEST_DATA_FILE_URL);

    size_t test_label_items_read = fread(data.test_label,
                                         sizeof(float),
                                         TEST_SAMPLE_COUNT * OUTPUT_NEURON_COUNT,
                                         test_label_file);
    printf("read %lu items (%lu samples) from %s\n", test_label_items_read,
           test_label_items_read / OUTPUT_NEURON_COUNT,
           TEST_LABEL_FILE_URL);

    fclose(test_data_file);
    fclose(test_label_file);
    return data;
}

static void free_dataset(dataset_t* restrict data)
{
    free(data->test_data);
    free(data->test_label);
}

////////////////////////////////////////
//////////// Tensor Math ///////////////
////////////////////////////////////////

static void tensor_dot(float* tensor_out,
                       const float* restrict tensor_a,
                       uint32_t tensor_a_rows,
                       uint32_t tensor_a_cols,
                       const float* restrict tensor_b,
                       [[maybe_unused]] uint32_t tensor_b_rows,
                       uint32_t tensor_b_cols)
{
    assert(tensor_a_cols == tensor_b_rows && "tensor_dot: dims don't match");
    for (uint32_t i = 0; i < tensor_a_rows; ++i) {
        for (uint32_t j = 0; j < tensor_b_cols; ++j) {
            float dot = 0.0f;
            for (uint32_t k = 0; k < tensor_a_cols; ++k) {
                dot += tensor_a[i * tensor_a_cols + k] * tensor_b[k * tensor_b_cols + j];
            }
            tensor_out[i * tensor_b_cols + j] = dot;
        }
    }
}

static void tensor_add(float* tensor_out,
                       const float* restrict tensor_a,
                       uint32_t tensor_a_rows,
                       uint32_t tensor_a_cols,
                       const float* restrict tensor_b,
                       [[maybe_unused]] uint32_t tensor_b_rows,
                       [[maybe_unused]] uint32_t tensor_b_cols)
{
    assert(tensor_a_rows == tensor_b_rows && tensor_a_cols == tensor_b_cols && "tensor_add: dims don't match");
    const uint32_t total_count = tensor_a_rows * tensor_a_cols;
    for (uint32_t i = 0; i < total_count; ++i) {
        tensor_out[i] = tensor_a[i] + tensor_b[i];
    }
}

////////////////////////////////////////
/////////// Neural Network /////////////
////////////////////////////////////////

struct mlp {
    float weights_input_hidden[WEIGHTS_INPUT_HIDDEN_COUNT];
    float weights_hidden_output[WEIGHTS_HIDDEN_OUTPUT_COUNT];
    float bias_hidden[HIDDEN_NEURON_COUNT];
    float bias_output[OUTPUT_NEURON_COUNT];
};
typedef struct mlp mlp_t;

void sigmoid_activation(float* restrict x,
                        uint32_t length)
{
    for (uint32_t i = 0; i < length; ++i) {
        x[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

static float compute_percentage(uint32_t numerator,
                                uint32_t denominator)
{
    return 100.0f * (float)numerator / (float)denominator;
}

static float compute_loss(const float* restrict predicted,
                          const float* restrict truth,
                          uint32_t length)
{
    float sum = 0.0f;
    for (uint32_t i = 0; i < length; ++i) {
        sum += (predicted[i] - truth[i]) * (predicted[i] - truth[i]);
    }
    return sum;
}

static uint32_t argmax(const float* restrict tensor,
                       uint32_t length)
{
    uint32_t max_id = 0;
    float max = tensor[max_id];
    for (uint32_t i = 1; i < length; ++i) {
        if (tensor[i] > max) {
            max = tensor[i];
            max_id = i;
        }
    }
    return max_id;
}

static void test(const mlp_t* restrict network,
                 const dataset_t* restrict data)
{
    float total_loss = 0.0f;
    uint32_t correct_w = 0, correct_b = 0;
    uint32_t total_w = 0, total_b = 0;
    for (uint32_t i = 0; i < TEST_SAMPLE_COUNT; ++i) {
        static float input_layer[INPUT_NEURON_COUNT];
        static float hidden_layer[HIDDEN_NEURON_COUNT];
        static float output_layer[OUTPUT_NEURON_COUNT];

        memcpy(input_layer,
               data->test_data + i * INPUT_NEURON_COUNT,
               sizeof input_layer);

        tensor_dot(hidden_layer,
                   network->weights_input_hidden,
                   HIDDEN_NEURON_COUNT,
                   INPUT_NEURON_COUNT,
                   input_layer,
                   INPUT_NEURON_COUNT,
                   1);
        tensor_add(hidden_layer,
                   hidden_layer,
                   HIDDEN_NEURON_COUNT,
                   1,
                   network->bias_hidden,
                   HIDDEN_NEURON_COUNT,
                   1);
        sigmoid_activation(hidden_layer,
                           HIDDEN_NEURON_COUNT);

        tensor_dot(output_layer,
                   network->weights_hidden_output,
                   OUTPUT_NEURON_COUNT,
                   HIDDEN_NEURON_COUNT,
                   hidden_layer,
                   HIDDEN_NEURON_COUNT,
                   1);
        tensor_add(output_layer,
                   output_layer,
                   OUTPUT_NEURON_COUNT,
                   1,
                   network->bias_output,
                   OUTPUT_NEURON_COUNT,
                   1);
        sigmoid_activation(output_layer,
                           OUTPUT_NEURON_COUNT);

        uint32_t truth = argmax(data->test_label + i * OUTPUT_NEURON_COUNT,
                                OUTPUT_NEURON_COUNT);
        uint32_t predicted = argmax(output_layer,
                                    OUTPUT_NEURON_COUNT);

        total_w += (truth == WHITE_CHECKMATE_LABEL);
        total_b += (truth == BLACK_CHECKMATE_LABEL);

        correct_w += (truth == WHITE_CHECKMATE_LABEL && predicted == truth);
        correct_b += (truth == BLACK_CHECKMATE_LABEL && predicted == truth);

        total_loss += compute_loss(output_layer,
                                   data->test_label + i * OUTPUT_NEURON_COUNT,
                                   OUTPUT_NEURON_COUNT);

        printf("testing %u/%u, (%.2f%% done)\r",
               i + 1,
               TEST_SAMPLE_COUNT,
               compute_percentage(i + 1, TEST_SAMPLE_COUNT));
    }
    float accuracy = compute_percentage(correct_w + correct_b, total_w + total_b);
    float w_accuracy = compute_percentage(correct_w, total_w);
    float b_accuracy = compute_percentage(correct_b, total_b);

    static char log_line[LOG_LINE_LEN];
    snprintf(log_line, LOG_LINE_LEN, "accuracy: %.6f%%, w: %.6f%%, b: %.6f%%, total_loss: %.6f, total_w: %u, correct_w: %u, total_b: %u, correct_b: %u\n",
             accuracy,
             w_accuracy,
             b_accuracy,
             total_loss / TEST_SAMPLE_COUNT,
             total_w,
             correct_w,
             total_b,
             correct_b);
    printf("%s", log_line);
}

static mlp_t* load_network()
{
    static mlp_t network = {};
    FILE* fp = open_file(MODEL_FILE_URL, "rb");
    fread(network.weights_input_hidden,
          sizeof(float),
          WEIGHTS_INPUT_HIDDEN_COUNT,
          fp);
    fread(network.weights_hidden_output,
          sizeof(float),
          WEIGHTS_HIDDEN_OUTPUT_COUNT,
          fp);
    fread(network.bias_hidden,
          sizeof(float),
          HIDDEN_NEURON_COUNT,
          fp);
    fread(network.bias_output,
          sizeof(float),
          OUTPUT_NEURON_COUNT,
          fp);
    fclose(fp);
    return &network;
}

////////////////////////////////////////
/////////////// Entry //////////////////
////////////////////////////////////////

int main()
{
    dataset_t data = load_dataset();
    mlp_t* network = load_network();
    test(network, &data);
    free_dataset(&data);
    return 0;
}
