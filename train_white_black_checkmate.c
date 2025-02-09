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
#include <ctype.h>
#include <assert.h>

////////////////////////////////////////
////////////// Constants ///////////////
////////////////////////////////////////

#define PI                              3.1415926f

#define MT_SEED                         0xdeadbeef
#define MT_STATE_VECTOR_LENGTH          624
#define MT_STATE_VECTOR_M               387
#define MT_UPPER_MASK                   0x80000000
#define MT_LOWER_MASK                   0x7fffffff
#define MT_TEMPERING_MASK_B             0x9d2c5680
#define MT_TEMPERING_MASK_C             0xefc60000

#define TRAIN_DATA_FILE_URL             "./data/train-white-black-checkmates-idx3-float-2M-lichess"
#define TRAIN_LABEL_FILE_URL            "./data/train-white-black-labels-idx2-float-2M-lichess"
#define TEST_DATA_FILE_URL              "./data/test-white-black-checkmates-idx3-float-2M-lichess"
#define TEST_LABEL_FILE_URL             "./data/test-white-black-labels-idx2-float-2M-lichess"

#define MODEL_FILE_URL                  "./models/weights-white-black-checkmates-2M"
#define LOG_FILE_URL                    "./models/weights-white-black-checkmates-log-2M"
#define LOG_LINE_LEN                    1000

#define TRAIN_SAMPLE_COUNT              1900000U
#define TEST_SAMPLE_COUNT                100000U

#define INPUT_NEURON_COUNT              128
#define HIDDEN_NEURON_COUNT             128
#define OUTPUT_NEURON_COUNT             2

#define WEIGHTS_INPUT_HIDDEN_COUNT      ((HIDDEN_NEURON_COUNT) * (INPUT_NEURON_COUNT))
#define WEIGHTS_HIDDEN_OUTPUT_COUNT     ((OUTPUT_NEURON_COUNT) * (HIDDEN_NEURON_COUNT))

#define LEARNING_RATE                   0.01f
#define EPOCH_COUNT                     100

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

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define CLAMP(X, _MIN, _MAX) (MIN(_MAX, MAX(X, _MIN)))

////////////////////////////////////////
/////////// Mersenne-Twister ///////////
////////////////////////////////////////

// source: https://github.com/ESultanik/mtwister
struct mt_state {
    uint32_t mt[MT_STATE_VECTOR_LENGTH];
    int32_t index;
};
typedef struct mt_state mt_state_t;

static mt_state_t gen_rand_state(uint32_t seed)
{
    mt_state_t rand = {};
    rand.mt[0] = seed & 0xffffffff;
    for(rand.index = 1; rand.index < MT_STATE_VECTOR_LENGTH; ++rand.index) {
        rand.mt[rand.index] = (6069 * rand.mt[rand.index - 1]) & 0xffffffff;
    }
    return rand;
}

static uint32_t gen_rand_uint32(mt_state_t* restrict rand)
{
    uint32_t y;
    uint32_t mag[2] = { 0x0, 0x9908b0df };
    if (rand->index >= MT_STATE_VECTOR_LENGTH || rand->index < 0) {
        if( rand->index >= MT_STATE_VECTOR_LENGTH + 1 || rand->index < 0) {
            *rand = gen_rand_state(4357);
        }
        int32_t kk;
        for(kk = 0; kk < MT_STATE_VECTOR_LENGTH - MT_STATE_VECTOR_M; ++kk) {
            y = (rand->mt[kk] & MT_UPPER_MASK) | (rand->mt[kk + 1] & MT_LOWER_MASK);
            rand->mt[kk] = rand->mt[kk + MT_STATE_VECTOR_M] ^ (y >> 1) ^ mag[y & 0x1];
        }
        for(; kk < MT_STATE_VECTOR_LENGTH - 1; ++kk) {
            y = (rand->mt[kk] & MT_UPPER_MASK) | (rand->mt[kk + 1] & MT_LOWER_MASK);
            rand->mt[kk] = rand->mt[kk + (MT_STATE_VECTOR_M - MT_STATE_VECTOR_LENGTH)] ^ (y >> 1) ^ mag[y & 0x1];
        }
        y = (rand->mt[MT_STATE_VECTOR_LENGTH - 1] & MT_UPPER_MASK) | (rand->mt[0] & MT_LOWER_MASK);
        rand->mt[MT_STATE_VECTOR_LENGTH - 1] = rand->mt[MT_STATE_VECTOR_M - 1] ^ (y >> 1) ^ mag[y & 0x1];
        rand->index = 0;
    }
    y = rand->mt[rand->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & MT_TEMPERING_MASK_B;
    y ^= (y << 15) & MT_TEMPERING_MASK_C;
    y ^= (y >> 18);
    return y;
}

static float gen_rand_float(mt_state_t* restrict rand)
{
    return (float)gen_rand_uint32(rand) / (float)((uint32_t)0xffffffff);
}

static float gen_rand_float_norm(mt_state_t* restrict rand,
                                 float mean,
                                 float standard_deviation)
{
    float u1 = gen_rand_float(rand);
    float u2 = gen_rand_float(rand);
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * u2);
    return z0 * standard_deviation + mean;
}

////////////////////////////////////////
////////////// Dataset /////////////////
////////////////////////////////////////

struct dataset {
    uint32_t train_count;
    uint32_t test_count;

    float* train_data;
    float* train_label;

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
    FILE* train_data_file   = open_file(TRAIN_DATA_FILE_URL, "rb");
    FILE* train_label_file  = open_file(TRAIN_LABEL_FILE_URL, "rb");
    FILE* test_data_file    = open_file(TEST_DATA_FILE_URL, "rb");
    FILE* test_label_file   = open_file(TEST_LABEL_FILE_URL, "rb");

    fseek(train_data_file, 16, SEEK_SET);
    fseek(train_label_file, 16, SEEK_SET);
    fseek(test_data_file, 16, SEEK_SET);
    fseek(test_label_file, 16, SEEK_SET);

    dataset_t data = {};
    data.train_data    = allocate_mem(TRAIN_SAMPLE_COUNT * INPUT_NEURON_COUNT * sizeof(float));
    data.train_label   = allocate_mem(TRAIN_SAMPLE_COUNT * OUTPUT_NEURON_COUNT * sizeof(float));
    data.test_data     = allocate_mem(TEST_SAMPLE_COUNT * INPUT_NEURON_COUNT * sizeof(float));
    data.test_label    = allocate_mem(TEST_SAMPLE_COUNT * OUTPUT_NEURON_COUNT * sizeof(float));

    size_t train_data_items_read = fread(data.train_data,
                                         sizeof(float),
                                         TRAIN_SAMPLE_COUNT * INPUT_NEURON_COUNT,
                                         train_data_file);
    printf("read %lu items (%lu samples) from %s\n", train_data_items_read,
           train_data_items_read / INPUT_NEURON_COUNT,
           TRAIN_DATA_FILE_URL);

    size_t train_label_items_read = fread(data.train_label,
                                          sizeof(float),
                                          TRAIN_SAMPLE_COUNT * OUTPUT_NEURON_COUNT,
                                          train_label_file);
    printf("read %lu items (%lu samples) from %s\n", train_label_items_read,
           train_label_items_read / OUTPUT_NEURON_COUNT,
           TRAIN_LABEL_FILE_URL);

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

    fclose(train_data_file);
    fclose(train_label_file);
    fclose(test_data_file);
    fclose(test_label_file);
    return data;
}

static void free_dataset(dataset_t* restrict data)
{
    free(data->train_data);
    free(data->train_label);
    free(data->test_data);
    free(data->test_label);
}

////////////////////////////////////////
//////////// Tensor Math ///////////////
////////////////////////////////////////

static void tensor_zero(float* restrict tensor,
                        uint32_t length)
{
    memset(tensor, 0, length * sizeof(float));
}

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

static void tensor_mul(float* tensor_out,
                       const float* restrict tensor_a,
                       uint32_t tensor_a_rows,
                       uint32_t tensor_a_cols,
                       const float* restrict tensor_b,
                       [[maybe_unused]] uint32_t tensor_b_rows,
                       [[maybe_unused]] uint32_t tensor_b_cols)
{
    assert(tensor_a_rows == tensor_b_rows && tensor_a_cols == tensor_b_cols && "tensor_mul: dims don't match");
    const uint32_t total_count = tensor_a_rows * tensor_a_cols;
    for (uint32_t i = 0; i < total_count; ++i) {
        tensor_out[i] = tensor_a[i] * tensor_b[i];
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

static void tensor_sub(float* tensor_out,
                       const float* restrict tensor_a,
                       uint32_t tensor_a_rows,
                       uint32_t tensor_a_cols,
                       const float* restrict tensor_b,
                       [[maybe_unused]] uint32_t tensor_b_rows,
                       [[maybe_unused]] uint32_t tensor_b_cols)
{
    assert(tensor_a_rows == tensor_b_rows && tensor_a_cols == tensor_b_cols && "tensor_sub: dims don't match");
    const uint32_t total_count = tensor_a_rows * tensor_a_cols;
    for (uint32_t i = 0; i < total_count; ++i) {
        tensor_out[i] = tensor_a[i] - tensor_b[i];
    }
}

static void tensor_mul_scalar(float* tensor_out,
                              float k,
                              const float* restrict tensor,
                              uint32_t tensor_rows,
                              uint32_t tensor_cols)
{
    const uint32_t total_count = tensor_rows * tensor_cols;
    for (uint32_t i = 0; i < total_count; ++i) {
        tensor_out[i] = k * tensor[i];
    }
}

static void tensor_sub_from_scalar(float* tensor_out,
                                   float k,
                                   const float* restrict tensor,
                                   uint32_t tensor_rows,
                                   uint32_t tensor_cols)
{
    const uint32_t total_count = tensor_rows * tensor_cols;
    for (uint32_t i = 0; i < total_count; ++i) {
        tensor_out[i] = k - tensor[i];
    }
}

static void tensor_transpose(float* tensor_out,
                             const float* restrict tensor,
                             uint32_t tensor_rows,
                             uint32_t tensor_cols)
{
    for (uint32_t i = 0; i < tensor_rows; ++i) {
        for (uint32_t j = 0; j < tensor_cols; ++j) {
            tensor_out[j * tensor_rows + i] = tensor[i * tensor_cols + j];
        }
    }
}

static void tensor_print(const float* restrict tensor,
                         uint32_t tensor_rows,
                         uint32_t tensor_cols)
{
    for (uint32_t i = 0; i < tensor_rows; ++i) {
        for (uint32_t j = 0; j < tensor_cols; ++j) {
            printf("%.6f\t", tensor[i * tensor_cols + j]);
        }
        putchar('\n');
    }
}

////////////////////////////////////////
/////////// Neural Network /////////////
////////////////////////////////////////

static void init_weights(float* restrict weights,
                         uint32_t input_neuron_count,
                         uint32_t output_neuron_count,
                         mt_state_t* restrict rand)
{
    const uint32_t length = input_neuron_count * output_neuron_count;
    const float standard_deviation = sqrtf(6.0f / (input_neuron_count + output_neuron_count));
    for (uint32_t i = 0; i < length; ++i) {
        weights[i] = gen_rand_float_norm(rand, 0.0f, standard_deviation);
    }
}

struct mlp {
    float weights_input_hidden[WEIGHTS_INPUT_HIDDEN_COUNT];
    float weights_hidden_output[WEIGHTS_HIDDEN_OUTPUT_COUNT];
    float bias_hidden[HIDDEN_NEURON_COUNT];
    float bias_output[OUTPUT_NEURON_COUNT];
};
typedef struct mlp mlp_t;

static mlp_t* gen_mlp(mt_state_t* restrict rand)
{
    static mlp_t network = {};
    init_weights(network.weights_input_hidden,
                 INPUT_NEURON_COUNT,
                 HIDDEN_NEURON_COUNT,
                 rand);
    init_weights(network.weights_hidden_output,
                 HIDDEN_NEURON_COUNT,
                 OUTPUT_NEURON_COUNT,
                 rand);
    tensor_zero(network.bias_hidden,
                HIDDEN_NEURON_COUNT);
    tensor_zero(network.bias_output,
                OUTPUT_NEURON_COUNT);
    return &network;
}

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

static void compute_errors(float* restrict errors_out,
                           const float* restrict predicted,
                           const float* restrict truth,
                           uint32_t length)
{
    for (uint32_t i = 0; i < length; ++i) {
        errors_out[i] = predicted[i] - truth[i];
    }
}

static float compute_loss(const float* restrict predicted,
                          const float* restrict truth,
                          uint32_t length)
{
    float loss = 0.0f;
    for (uint32_t i = 0; i < length; ++i) {
        loss += (predicted[i] - truth[i]) * (predicted[i] - truth[i]);
    }
    return loss;
}

static float compute_diff(const float* restrict gradient,
                          uint32_t length)
{
    float sum = 0.0f;
    for (uint32_t i = 0; i < length; ++i) {
        sum += gradient[i] * gradient[i];
    }
    return sqrtf(sum);
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

static void query(const mlp_t* restrict network,
                  const float* restrict input_tensor,
                  float* restrict output_tensor)
{
    static float input_layer[INPUT_NEURON_COUNT];
    static float hidden_layer[HIDDEN_NEURON_COUNT];
    static float output_layer[OUTPUT_NEURON_COUNT];

    memcpy(input_layer,
           input_tensor,
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

    memcpy(output_tensor,
           output_layer,
           sizeof output_layer);
}

static void test(FILE* log_file,
                 const mlp_t* restrict network,
                 const dataset_t* restrict data,
                 float d_a,
                 float d_b)
{
    float total_loss = 0.0f;
    uint32_t correct_white = 0, correct_black = 0;
    uint32_t total_white = 0, total_black = 0;
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

        total_white += (truth == WHITE_CHECKMATE_LABEL);
        total_black += (truth == BLACK_CHECKMATE_LABEL);

        correct_white += (truth == WHITE_CHECKMATE_LABEL && predicted == truth);
        correct_black += (truth == BLACK_CHECKMATE_LABEL && predicted == truth);

        total_loss += compute_loss(output_layer,
                                   data->test_label + i * OUTPUT_NEURON_COUNT,
                                   OUTPUT_NEURON_COUNT);

        printf("testing %u/%u, (%.2f%% done)\r",
               i + 1,
               TEST_SAMPLE_COUNT,
               compute_percentage(i + 1, TEST_SAMPLE_COUNT));
    }
    float accuracy = compute_percentage(correct_white + correct_black, total_white + total_black);
    float white_accuracy = compute_percentage(correct_white, total_white);
    float black_accuracy = compute_percentage(correct_black, total_black);

    static char log_line[LOG_LINE_LEN];
    snprintf(log_line, LOG_LINE_LEN, "lr: %.6f, loss: %.6f, d_a: %.6f, d_b: %.6f, accuracy: %.6f%%, white: %.6f%%, black: %.6f%%\n",
             LEARNING_RATE,
             total_loss / TEST_SAMPLE_COUNT,
             d_a, d_b,
             accuracy,
             white_accuracy,
             black_accuracy);
    printf("%s", log_line);
    fputs(log_line, log_file);
}

static void train(FILE* log_file,
                  mlp_t* restrict network,
                  const dataset_t* restrict data)
{
    for (uint32_t epoch = 1; epoch <= EPOCH_COUNT; ++epoch) {
        float total_d_a = 0.0f, total_d_b = 0.0f;
        for (uint32_t i = 0; i < TRAIN_SAMPLE_COUNT; ++i) {
            static float weights_hidden_output_transpose[WEIGHTS_HIDDEN_OUTPUT_COUNT];
            tensor_transpose(weights_hidden_output_transpose,
                             network->weights_hidden_output,
                             OUTPUT_NEURON_COUNT,
                             HIDDEN_NEURON_COUNT);

            static float input_layer[INPUT_NEURON_COUNT];
            static float hidden_layer[HIDDEN_NEURON_COUNT];
            static float output_layer[OUTPUT_NEURON_COUNT];
            memcpy(input_layer,
                   data->train_data + i * INPUT_NEURON_COUNT,
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

            static float output_layer_errors[OUTPUT_NEURON_COUNT];
            compute_errors(output_layer_errors,
                           output_layer,
                           data->train_label + i * OUTPUT_NEURON_COUNT,
                           OUTPUT_NEURON_COUNT);

            static float hidden_layer_errors[HIDDEN_NEURON_COUNT];
            tensor_dot(hidden_layer_errors,
                       weights_hidden_output_transpose,
                       HIDDEN_NEURON_COUNT,
                       OUTPUT_NEURON_COUNT,
                       output_layer_errors,
                       OUTPUT_NEURON_COUNT,
                       1);

            static float output_intermediate[OUTPUT_NEURON_COUNT];
            tensor_sub_from_scalar(output_intermediate,
                                   1.0f,
                                   output_layer,
                                   1,
                                   OUTPUT_NEURON_COUNT);
            tensor_mul(output_intermediate,
                       output_intermediate,
                       1,
                       OUTPUT_NEURON_COUNT,
                       output_layer,
                       1,
                       OUTPUT_NEURON_COUNT);
            tensor_mul(output_intermediate,
                       output_intermediate,
                       1,
                       OUTPUT_NEURON_COUNT,
                       output_layer_errors,
                       1,
                       OUTPUT_NEURON_COUNT);

            static float gradient_hidden_output[WEIGHTS_HIDDEN_OUTPUT_COUNT];
            tensor_dot(gradient_hidden_output,
                       output_intermediate,
                       OUTPUT_NEURON_COUNT,
                       1,
                       hidden_layer,
                       1,
                       HIDDEN_NEURON_COUNT);
            tensor_mul_scalar(gradient_hidden_output,
                              LEARNING_RATE,
                              gradient_hidden_output,
                              OUTPUT_NEURON_COUNT,
                              HIDDEN_NEURON_COUNT);
            total_d_a += compute_diff(gradient_hidden_output,
                                      WEIGHTS_HIDDEN_OUTPUT_COUNT);

            tensor_sub(network->weights_hidden_output,
                       network->weights_hidden_output,
                       OUTPUT_NEURON_COUNT,
                       HIDDEN_NEURON_COUNT,
                       gradient_hidden_output,
                       OUTPUT_NEURON_COUNT,
                       HIDDEN_NEURON_COUNT);

            tensor_mul_scalar(output_intermediate,
                              LEARNING_RATE,
                              output_intermediate,
                              OUTPUT_NEURON_COUNT,
                              1);
            tensor_sub(network->bias_output,
                       network->bias_output,
                       OUTPUT_NEURON_COUNT,
                       1,
                       output_intermediate,
                       OUTPUT_NEURON_COUNT,
                       1);

            static float hidden_intermediate[HIDDEN_NEURON_COUNT];
            tensor_sub_from_scalar(hidden_intermediate,
                                   1.0f,
                                   hidden_layer,
                                   1,
                                   HIDDEN_NEURON_COUNT);
            tensor_mul(hidden_intermediate,
                       hidden_intermediate,
                       1,
                       HIDDEN_NEURON_COUNT,
                       hidden_layer,
                       1,
                       HIDDEN_NEURON_COUNT);
            tensor_mul(hidden_intermediate,
                       hidden_intermediate,
                       1,
                       HIDDEN_NEURON_COUNT,
                       hidden_layer_errors,
                       1,
                       HIDDEN_NEURON_COUNT);

            static float gradient_input_hidden[WEIGHTS_INPUT_HIDDEN_COUNT];
            tensor_dot(gradient_input_hidden,
                       hidden_intermediate,
                       HIDDEN_NEURON_COUNT,
                       1,
                       input_layer,
                       1,
                       INPUT_NEURON_COUNT);
            tensor_mul_scalar(gradient_input_hidden,
                              LEARNING_RATE,
                              gradient_input_hidden,
                              HIDDEN_NEURON_COUNT,
                              INPUT_NEURON_COUNT);
            total_d_b += compute_diff(gradient_input_hidden,
                                      WEIGHTS_INPUT_HIDDEN_COUNT);

            tensor_sub(network->weights_input_hidden,
                       network->weights_input_hidden,
                       HIDDEN_NEURON_COUNT,
                       INPUT_NEURON_COUNT,
                       gradient_input_hidden,
                       HIDDEN_NEURON_COUNT,
                       INPUT_NEURON_COUNT);

            tensor_mul_scalar(hidden_intermediate,
                              LEARNING_RATE,
                              hidden_intermediate,
                              HIDDEN_NEURON_COUNT,
                              1);
            tensor_sub(network->bias_hidden,
                       network->bias_hidden,
                       HIDDEN_NEURON_COUNT,
                       1,
                       hidden_intermediate,
                       HIDDEN_NEURON_COUNT,
                       1);

            printf("epoch: %u, training %u/%u, (%.2f%% done)\r",
                   epoch,
                   i + 1,
                   TRAIN_SAMPLE_COUNT,
                   compute_percentage(i + 1, TRAIN_SAMPLE_COUNT));
        }
        putchar('\n');
        test(log_file,
             network,
             data,
             total_d_a / WEIGHTS_HIDDEN_OUTPUT_COUNT,
             total_d_b / WEIGHTS_INPUT_HIDDEN_COUNT);
    }
}

static void save(const mlp_t* restrict network)
{
    FILE* fp = open_file(MODEL_FILE_URL, "wb");
    fwrite(network->weights_input_hidden,
           sizeof(float),
           WEIGHTS_INPUT_HIDDEN_COUNT,
           fp);
    fwrite(network->weights_hidden_output,
           sizeof(float),
           WEIGHTS_HIDDEN_OUTPUT_COUNT,
           fp);
    fwrite(network->bias_hidden,
           sizeof(float),
           HIDDEN_NEURON_COUNT,
           fp);
    fwrite(network->bias_output,
           sizeof(float),
           OUTPUT_NEURON_COUNT,
           fp);
    fclose(fp);
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

static void tensorize_board_fen(float* restrict move_tensor,
                                const char* restrict board_fen,
                                int offset)
{
    for (const char* c = board_fen; *c; ++c) {
        if (isdigit(*c)) {
            int k = *c - '0';
            while (--k >= 0) {
                move_tensor[offset++] = SQUARE_EMPTY;
            }
        }
        else {
            switch (*c) {
                case 'P':
                    move_tensor[offset++] = SQUARE_WHITE_PAWN;
                    break;
                case 'p':
                    move_tensor[offset++] = SQUARE_BLACK_PAWN;
                    break;
                case 'N':
                    move_tensor[offset++] = SQUARE_WHITE_KNIGHT;
                    break;
                case 'n':
                    move_tensor[offset++] = SQUARE_BLACK_KNIGHT;
                    break;
                case 'B':
                    move_tensor[offset++] = SQUARE_WHITE_BISHOP;
                    break;
                case 'b':
                    move_tensor[offset++] = SQUARE_BLACK_BISHOP;
                    break;
                case 'R':
                    move_tensor[offset++] = SQUARE_WHITE_ROOK;
                    break;
                case 'r':
                    move_tensor[offset++] = SQUARE_BLACK_ROOK;
                    break;
                case 'Q':
                    move_tensor[offset++] = SQUARE_WHITE_QUEEN;
                    break;
                case 'q':
                    move_tensor[offset++] = SQUARE_BLACK_QUEEN;
                    break;
                case 'K':
                    move_tensor[offset++] = SQUARE_WHITE_KING;
                    break;
                case 'k':
                    move_tensor[offset++] = SQUARE_BLACK_KING;
                    break;
            }
        }
    }
}

static void tensorize_move(float* restrict tensor,
                           const char* restrict prev_board_fen,
                           const char* restrict current_board_fen)
{
    tensorize_board_fen(tensor,
                        prev_board_fen,
                        0);
    tensorize_board_fen(tensor,
                        current_board_fen,
                        64);
}

////////////////////////////////////////
/////////////// Entry //////////////////
////////////////////////////////////////

int main(int argc, char** argv)
{
    if (argc > 2) {
        mlp_t* network = load_network();
        float input_tensor[INPUT_NEURON_COUNT];
        tensorize_move(input_tensor,
                       argv[1],
                       argv[2]);
        printf("input_tensor:\n");
        tensor_print(input_tensor,
                     16,
                     8);
        float output_tensor[OUTPUT_NEURON_COUNT];
        query(network,
              input_tensor,
              output_tensor);
        uint32_t predicted = argmax(output_tensor,
                                    OUTPUT_NEURON_COUNT);
        printf("output_tensor:\n");
        tensor_print(output_tensor,
                     1,
                     OUTPUT_NEURON_COUNT);
        printf("label: %s\n", predicted == WHITE_CHECKMATE_LABEL ? "white" : "black");
    }
    else {
        dataset_t data = load_dataset();
        mt_state_t rand = gen_rand_state(MT_SEED);
        mlp_t* network = gen_mlp(&rand);
        FILE* log_file = open_file(LOG_FILE_URL, "w");
        test(log_file, network, &data, 0.0f, 0.0f);
        train(log_file, network, &data);
        save(network);
        free_dataset(&data);
        fclose(log_file);
    }
    return 0;
}
