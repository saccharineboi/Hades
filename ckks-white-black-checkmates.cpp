//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

#define PROFILE
#include "openfhe.h"

// header files needed for serialization
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

#include <vector>
#include <fstream>
#include <chrono>
#include <utility>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <cmath>

#define SERIALIZE_CONTEXT 0

////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// SERIALIZATION /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
const char* HADES_CRYPTOCONTEXT_FILE        { "hades_cryptocontext" };
const char* HADES_PUBLICKEY_FILE            { "hades_public_key" };
const char* HADES_RELINEARIZATION_KEY_FILE  { "hades_relinearization_key" };
const char* HADES_ROTATION_KEY_FILE         { "hades_rotation_key" };
const char* HADES_INPUT_CIPHERTEXT_FILE     { "hades_input_ciphertext" };
const char* HADES_OUTPUT_CIPHERTEXT_FILE    { "hades_output_ciphertext" };

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// DATASETS ////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr const char* TEST_DATA_FILE_URL    { "test-white-black-checkmates-idx3-float-2M-lichess" };
constexpr const char* TEST_LABEL_FILE_URL   { "test-white-black-labels-idx2-float-2M-lichess" };
constexpr uint32_t TEST_SAMPLE_COUNT        { 100 };
constexpr uint32_t ENC_TEST_SAMPLE_COUNT    { 100 };

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// MODEL ///////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr const char* MODEL_FILE_URL        { "weights-white-black-checkmates-2M" };

constexpr uint32_t INPUT_NEURON_COUNT       { 128 };
constexpr uint32_t HIDDEN_NEURON_COUNT      { 128 };
constexpr uint32_t OUTPUT_NEURON_COUNT      { 2 };

constexpr uint32_t LEGAL_MOVE_LABEL         { 0 };
constexpr uint32_t ILLEGAL_MOVE_LABEL       { 1 };

constexpr uint32_t LOG_LINE_LEN             { 500 };

constexpr uint32_t WEIGHTS_INPUT_HIDDEN_COUNT  { HIDDEN_NEURON_COUNT * INPUT_NEURON_COUNT };
constexpr uint32_t WEIGHTS_HIDDEN_OUTPUT_COUNT { OUTPUT_NEURON_COUNT * HIDDEN_NEURON_COUNT };

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// FHE PARAMS //////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr double LOGISTIC_INTERVAL_BEGIN        { -100.0f };
constexpr double LOGISTIC_INTERVAL_END          {  100.0f };
constexpr uint32_t LOGISTIC_POLY_DEG            {  100 };
constexpr uint32_t BATCH_SIZE                   {  128 };
constexpr uint32_t MULT_DEPTH                   {  18 };

using namespace lbcrypto;

////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// UTILS //////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

static std::FILE* open_file(const char* path, const char* mode)
{
    std::FILE* file{ std::fopen(path, mode) };
    if (!file) {
        std::fprintf(stderr, "open_file: Failed to open '%s' (mode: '%s')\n", path, mode);
        std::exit(EXIT_FAILURE);
    }
    return file;
}

static void* allocate_mem(size_t size)
{
    void* addr{ std::malloc(size) };
    if (!addr) {
        std::perror("allocate_mem");
        std::exit(EXIT_FAILURE);
    }
    return addr;
}

static double compute_percentage(uint64_t numerator,
                                 uint64_t denominator)
{
    return 100.0 * (double)numerator / (double)denominator;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// DATASET /////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

struct Dataset
{
private:
    std::vector<double> test_data;
    std::vector<double> test_label;

public:
    Dataset();
    ~Dataset() {};

    Ciphertext<DCRTPoly> RetrieveInputCiphertext(size_t index,
                                                 CryptoContext<DCRTPoly>& cc,
                                                 KeyPair<DCRTPoly>& keyPair) const;
    std::vector<double> RetrieveInput(size_t index) const;
    std::vector<double> RetrieveOutput(size_t index) const;
};

Ciphertext<DCRTPoly> Dataset::RetrieveInputCiphertext(size_t index,
                                                      CryptoContext<DCRTPoly>& cc,
                                                      KeyPair<DCRTPoly>& keyPair) const
{
    assert((index * INPUT_NEURON_COUNT < test_data.size()) && "RetrieveInputCiphertext");
    std::vector<double> input(test_data.begin() + index * INPUT_NEURON_COUNT, test_data.begin() + (index + 1) * INPUT_NEURON_COUNT);
    Plaintext ptxt{ cc->MakeCKKSPackedPlaintext(input) };
    Ciphertext<DCRTPoly> ctxt{ cc->Encrypt(keyPair.publicKey, ptxt) };
    return ctxt;
}

std::vector<double> Dataset::RetrieveInput(size_t index) const
{
    assert((index * INPUT_NEURON_COUNT < test_data.size()) && "RetrieveInput");
    std::vector<double> input(test_data.begin() + index * INPUT_NEURON_COUNT, test_data.begin() + (index + 1) * INPUT_NEURON_COUNT);
    return input;
}

std::vector<double> Dataset::RetrieveOutput(size_t index) const
{
    assert((index * OUTPUT_NEURON_COUNT < test_label.size()) && "RetrieveOutput");
    std::vector<double> output(test_label.begin() + index * OUTPUT_NEURON_COUNT, test_label.begin() + (index + 1) * OUTPUT_NEURON_COUNT);
    return output;
}

Dataset::Dataset()
{
    std::FILE* test_data_file{ open_file(TEST_DATA_FILE_URL, "rb") };
    std::FILE* test_label_file{ open_file(TEST_LABEL_FILE_URL, "rb") };

    std::fseek(test_data_file, 16, SEEK_SET);
    std::fseek(test_label_file, 16, SEEK_SET);

    float* float_data{ static_cast<float*>(allocate_mem(TEST_SAMPLE_COUNT * INPUT_NEURON_COUNT * sizeof(float))) };
    float* float_label{ static_cast<float*>(allocate_mem(TEST_SAMPLE_COUNT * OUTPUT_NEURON_COUNT * sizeof(float))) };

    size_t test_data_items_read{ std::fread(float_data,
                                            sizeof(float),
                                            TEST_SAMPLE_COUNT * INPUT_NEURON_COUNT,
                                            test_data_file) };
    std::printf("Read %lu items (%lu smaples) from %s\n", test_data_items_read,
                test_data_items_read / INPUT_NEURON_COUNT,
                TEST_DATA_FILE_URL);

    size_t test_label_items_read{ std::fread(float_label,
                                             sizeof(float),
                                             TEST_SAMPLE_COUNT * OUTPUT_NEURON_COUNT,
                                             test_label_file) };
    std::printf("Read %lu items (%lu samples) from %s\n", test_label_items_read,
                test_label_items_read / OUTPUT_NEURON_COUNT,
                TEST_LABEL_FILE_URL);

    test_data.reserve(INPUT_NEURON_COUNT * TEST_SAMPLE_COUNT);
    test_label.reserve(OUTPUT_NEURON_COUNT * TEST_SAMPLE_COUNT);

    for (size_t i{}; i < test_data_items_read; ++i) {
        test_data.push_back(float_data[i]);
    }

    for (size_t i{}; i < test_label_items_read; ++i) {
        test_label.push_back(float_label[i]);
    }

    std::printf("test_data.size() = %lu, test_label.size() = %lu\n", test_data.size(), test_label.size());

    std::free(float_data);
    std::free(float_label);

    std::fclose(test_data_file);
    std::fclose(test_label_file);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// TENSORS /////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

static std::pair<double, double> tensor_min_max(const std::vector<double>& tensor)
{
    double min{ tensor[0] }, max{ tensor[0] };
    for (double x : tensor) {
        if (x < min) {
            min = x;
        }
        if (x > max) {
            max = x;
        }
    }
    return { min, max };
}

static void tensor_print(const std::vector<double>& tensor,
                         uint32_t tensor_rows,
                         uint32_t tensor_cols)
{
    for (uint32_t i{}; i < tensor_rows; ++i) {
        for (uint32_t j{}; j < tensor_cols; ++j) {
            std::printf("%.6f\t", tensor[i * tensor_cols + j]);
        }
        std::putchar('\n');
    }
}

static void tensor_dot(std::vector<double>& tensor_out,
                       const std::vector<double>& tensor_a,
                       uint32_t tensor_a_rows,
                       uint32_t tensor_a_cols,
                       const std::vector<double>& tensor_b,
                       [[maybe_unused]] uint32_t tensor_b_rows,
                       uint32_t tensor_b_cols)
{
    assert(tensor_a_cols == tensor_b_rows && "tensor_dot: dims don't match");
    for (uint32_t i{}; i < tensor_a_rows; ++i) {
        for (uint32_t j{}; j < tensor_b_cols; ++j) {
            double dot{};
            for (uint32_t k{}; k < tensor_a_cols; ++k) {
                dot += tensor_a[i * tensor_a_cols + k] * tensor_b[k * tensor_b_cols + j];
            }
            tensor_out[i * tensor_b_cols + j] = dot;
        }
    }
}

static Ciphertext<DCRTPoly> tensor_dot_encrypted(CryptoContext<DCRTPoly>& cc,
                                                 const std::vector<Ciphertext<DCRTPoly>>& tensor_a,
                                                 uint32_t tensor_a_rows,
                                                 uint32_t tensor_a_cols,
                                                 Ciphertext<DCRTPoly>& tensor_b)
{
    std::vector<Ciphertext<DCRTPoly>> rows;
    for (uint32_t i{}; i < tensor_a_rows; ++i) {
        if (i) {
            Ciphertext<DCRTPoly> rotated{ cc->EvalRotate(tensor_b, ((int32_t)i)) };
            rows.push_back(cc->EvalMult(tensor_a[i], rotated));
        }
        else {
            rows.push_back(cc->EvalMult(tensor_a[i], tensor_b));
        }
    }
    return cc->EvalAddManyInPlace(rows);
}

static Ciphertext<DCRTPoly> tensor_dot_encrypted_hybrid(CryptoContext<DCRTPoly>& cc,
                                                        const std::vector<Ciphertext<DCRTPoly>>& tensor_a,
                                                        uint32_t tensor_a_rows,
                                                        uint32_t tensor_a_cols,
                                                        Ciphertext<DCRTPoly>& tensor_b,
                                                        [[maybe_unused]] uint32_t tensor_b_rows,
                                                        [[maybe_unused]] uint32_t tensor_b_cols)
{
    Ciphertext<DCRTPoly> partial_sums = tensor_dot_encrypted(cc,
                                                             tensor_a,
                                                             tensor_a_rows,
                                                             tensor_a_cols,
                                                             tensor_b);
    uint32_t shift_count{ static_cast<uint32_t>(std::log2(tensor_a_cols) - std::log2(tensor_a_rows)) };
    uint32_t shift{ tensor_a_cols / 2 };
    for (uint32_t i{}; i < shift_count; ++i) {
        Ciphertext<DCRTPoly> rotated{ cc->EvalRotate(partial_sums, shift) };
        cc->EvalAddInPlace(partial_sums, rotated);
        shift /= 2;
    }
    return partial_sums;
}

static void tensor_add(std::vector<double>& tensor_out,
                       const std::vector<double>& tensor_a,
                       uint32_t tensor_a_rows,
                       uint32_t tensor_a_cols,
                       const std::vector<double>& tensor_b,
                       [[maybe_unused]] uint32_t tensor_b_rows,
                       [[maybe_unused]] uint32_t tensor_b_cols)
{
    assert(tensor_a_rows == tensor_b_rows && tensor_a_cols == tensor_b_cols && "tensor_add: dims don't match");
    const uint32_t total_count{ tensor_a_rows * tensor_a_cols };
    for (uint32_t i{}; i < total_count; ++i) {
        tensor_out[i] = tensor_a[i] + tensor_b[i];
    }
}

static void sigmoid_activation(std::vector<double>& x)
{
    for (double& xi : x) {
        xi = 1.0f / (1.0f + expf(-xi));
    }
}

static double compute_loss(const std::vector<double>& predicted,
                           const std::vector<double>& truth)
{
    double sum{};
    for (size_t i{}; i < predicted.size(); ++i) {
        sum += (predicted[i] - truth[i]) * (predicted[i] - truth[i]);
    }
    return sum;
}

static uint32_t argmax(const std::vector<double>& tensor)
{
    uint32_t max_id{};
    double max{ tensor[max_id] };
    for (uint32_t i{ 1 }; i < static_cast<uint32_t>(tensor.size()); ++i) {
        if (tensor[i] > max) {
            max = tensor[i];
            max_id = i;
        }
    }
    return max_id;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// NEURAL NETWORK ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

struct MLP
{
private:
    std::vector<double> weights_input_hidden;
    std::vector<double> weights_hidden_output;
    std::vector<double> bias_hidden;
    std::vector<double> bias_output;

    std::vector<Ciphertext<DCRTPoly>> encrypted_weights_input_hidden;
    std::vector<Ciphertext<DCRTPoly>> encrypted_weights_hidden_output;
    Ciphertext<DCRTPoly> encrypted_bias_hidden;
    Ciphertext<DCRTPoly> encrypted_bias_output;

public:
    MLP(CryptoContext<DCRTPoly>& cc, KeyPair<DCRTPoly>& keyPair);
    ~MLP() {}

    void TestClear(const Dataset& dataset);
    void TestCiphertext(const Dataset& dataset,
                        CryptoContext<DCRTPoly>& cc,
                        KeyPair<DCRTPoly>& keyPair);
};

void MLP::TestClear(const Dataset& dataset)
{
    double total_time_passed{};
    double total_loss{};
    uint32_t correct_l{}, correct_i{};
    uint32_t total_l{}, total_i{};

    double lowest_hidden{}, highest_hidden{};
    double lowest_output{}, highest_output{};

    for (uint32_t i{}; i < TEST_SAMPLE_COUNT; ++i) {
        std::vector<double> input_layer{ dataset.RetrieveInput(i) };
        std::vector<double> hidden_layer(HIDDEN_NEURON_COUNT);
        std::vector<double> output_layer(OUTPUT_NEURON_COUNT);

        auto t0 = std::chrono::high_resolution_clock::now();

        tensor_dot(hidden_layer,
                   weights_input_hidden,
                   HIDDEN_NEURON_COUNT,
                   INPUT_NEURON_COUNT,
                   input_layer,
                   INPUT_NEURON_COUNT,
                   1);

        tensor_add(hidden_layer,
                   hidden_layer,
                   HIDDEN_NEURON_COUNT,
                   1,
                   bias_hidden,
                   HIDDEN_NEURON_COUNT,
                   1);

        const auto [ lowest_hidden_cand, highest_hidden_cand ] = tensor_min_max(hidden_layer);
        if (lowest_hidden_cand < lowest_hidden) {
            lowest_hidden = lowest_hidden_cand;
        }
        if (highest_hidden_cand > highest_hidden) {
            highest_hidden = highest_hidden_cand;
        }

        sigmoid_activation(hidden_layer);

        tensor_dot(output_layer,
                   weights_hidden_output,
                   OUTPUT_NEURON_COUNT,
                   HIDDEN_NEURON_COUNT,
                   hidden_layer,
                   HIDDEN_NEURON_COUNT,
                   1);

        tensor_add(output_layer,
                   output_layer,
                   OUTPUT_NEURON_COUNT,
                   1,
                   bias_output,
                   OUTPUT_NEURON_COUNT,
                   1);

        const auto [ lowest_output_cand, highest_output_cand ] = tensor_min_max(output_layer);
        if (lowest_output_cand < lowest_output) {
            lowest_output = lowest_output_cand;
        }
        if (highest_output_cand > highest_output) {
            highest_output = highest_output_cand;
        }

        sigmoid_activation(output_layer);

        auto t1 = std::chrono::high_resolution_clock::now();
        auto time_passed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
        total_time_passed += static_cast<double>(time_passed.count());

        uint32_t truth{ argmax(dataset.RetrieveOutput(i)) };
        uint32_t predicted{ argmax(output_layer) };

        total_l += (truth == LEGAL_MOVE_LABEL);
        total_i += (truth == ILLEGAL_MOVE_LABEL);

        correct_l += (truth == LEGAL_MOVE_LABEL && predicted == truth);
        correct_i += (truth == ILLEGAL_MOVE_LABEL && predicted == truth);

        total_loss += compute_loss(output_layer, dataset.RetrieveOutput(i));
    }
    double accuracy{ compute_percentage(correct_l + correct_i, total_l + total_i) };
    double legal_accuracy{ compute_percentage(correct_l, total_l) };
    double illegal_accuracy{ compute_percentage(correct_i, total_i) };
    double avg_time_passed{ total_time_passed / TEST_SAMPLE_COUNT };

    static char log_line[LOG_LINE_LEN];
    std::snprintf(log_line, LOG_LINE_LEN, "accuracy: %.6f%%, legal: %.6f%%, illegal: %.6f%%, total_loss: %.6f, total: %u, correct: %u, avg_time_passed: %.6f microseconds\n",
                  accuracy,
                  legal_accuracy,
                  illegal_accuracy,
                  total_loss / TEST_SAMPLE_COUNT,
                  total_l + total_i,
                  correct_l + correct_i,
                  avg_time_passed);
    std::printf("%s", log_line);
    std::printf("lowest_hidden = %.2f, highest_hidden = %.2f\n", lowest_hidden, highest_hidden);
    std::printf("lowest_output = %.2f, highest_output = %.2f\n", lowest_output, highest_output);
}

void MLP::TestCiphertext(const Dataset& dataset,
                         CryptoContext<DCRTPoly>& cc,
                         KeyPair<DCRTPoly>& keyPair)
{
    double total_time_passed{};
    double total_loss{};
    uint32_t correct_l{}, correct_i{};
    uint32_t total_l{}, total_i{};
    for (uint32_t i{}; i < ENC_TEST_SAMPLE_COUNT; ++i) {
        Ciphertext<DCRTPoly> encrypted_input_layer{ dataset.RetrieveInputCiphertext(i, cc, keyPair) };
        Ciphertext<DCRTPoly> encrypted_hidden_layer;
        Ciphertext<DCRTPoly> encrypted_output_layer;

#if SERIALIZE_CONTEXT == 1
        if (!Serial::SerializeToFile(HADES_INPUT_CIPHERTEXT_FILE, encrypted_input_layer, SerType::BINARY)) {
            std::fprintf(stderr, "Couldn't serialize the input ciphertext to '%s'\n", HADES_INPUT_CIPHERTEXT_FILE);
        }
#endif

        auto t0 = std::chrono::high_resolution_clock::now();

        encrypted_hidden_layer = tensor_dot_encrypted(cc,
                                                      encrypted_weights_input_hidden,
                                                      HIDDEN_NEURON_COUNT,
                                                      INPUT_NEURON_COUNT,
                                                      encrypted_input_layer);

        cc->EvalAddInPlace(encrypted_hidden_layer,
                           encrypted_bias_hidden);

        encrypted_hidden_layer = cc->EvalLogistic(encrypted_hidden_layer,
                                                  LOGISTIC_INTERVAL_BEGIN,
                                                  LOGISTIC_INTERVAL_END,
                                                  LOGISTIC_POLY_DEG);

        encrypted_output_layer = tensor_dot_encrypted_hybrid(cc,
                                                             encrypted_weights_hidden_output,
                                                             OUTPUT_NEURON_COUNT,
                                                             HIDDEN_NEURON_COUNT,
                                                             encrypted_hidden_layer,
                                                             HIDDEN_NEURON_COUNT,
                                                             1);

        cc->EvalAddInPlace(encrypted_output_layer,
                           encrypted_bias_output);

        encrypted_output_layer = cc->EvalLogistic(encrypted_output_layer,
                                                  LOGISTIC_INTERVAL_BEGIN,
                                                  LOGISTIC_INTERVAL_END,
                                                  LOGISTIC_POLY_DEG);

        auto t1 = std::chrono::high_resolution_clock::now();
        auto time_passed = std::chrono::duration_cast<std::chrono::seconds>(t1 - t0);
        total_time_passed += static_cast<double>(time_passed.count());

#if SERIALIZE_CONTEXT == 1
        if (!Serial::SerializeToFile(HADES_OUTPUT_CIPHERTEXT_FILE, encrypted_output_layer, SerType::BINARY)) {
            std::fprintf(stderr, "Couldn't serialize the output ciphertext to '%s'\n", HADES_INPUT_CIPHERTEXT_FILE);
        }
#endif

        uint32_t truth = argmax(dataset.RetrieveOutput(i));

        Plaintext decrypted_output_layer;
        cc->Decrypt(keyPair.secretKey, encrypted_output_layer, &decrypted_output_layer);
        decrypted_output_layer->SetLength(OUTPUT_NEURON_COUNT);
        std::vector<double> decrypted_values = decrypted_output_layer->GetRealPackedValue();
        tensor_print(decrypted_values, 1, OUTPUT_NEURON_COUNT);
        uint32_t predicted = argmax(decrypted_values);

        total_l += (truth == LEGAL_MOVE_LABEL);
        total_i += (truth == ILLEGAL_MOVE_LABEL);

        correct_l += (truth == LEGAL_MOVE_LABEL && predicted == truth);
        correct_i += (truth == ILLEGAL_MOVE_LABEL && predicted == truth);

        total_loss += compute_loss(decrypted_values, dataset.RetrieveOutput(i));
    }
    double accuracy = compute_percentage(correct_l + correct_i, total_l + total_i);
    double legal_accuracy = compute_percentage(correct_l, total_l);
    double illegal_accuracy = compute_percentage(correct_i, total_i);
    double avg_time_passed = total_time_passed / ENC_TEST_SAMPLE_COUNT;

    static char log_line[LOG_LINE_LEN];
    std::snprintf(log_line, LOG_LINE_LEN, "accuracy: %.6f%%, legal: %.6f%%, illegal: %.6f%%, total_loss: %.6f, total: %u, correct: %u, avg_time_passed: %.6f seconds\n",
                  accuracy,
                  legal_accuracy,
                  illegal_accuracy,
                  total_loss / ENC_TEST_SAMPLE_COUNT,
                  total_l + total_i,
                  correct_l + correct_i,
                  avg_time_passed);
    std::printf("%s", log_line);
}

MLP::MLP(CryptoContext<DCRTPoly>& cc, KeyPair<DCRTPoly>& keyPair)
{
    static float tmp_weights_input_hidden[WEIGHTS_INPUT_HIDDEN_COUNT];
    static float tmp_weights_hidden_output[WEIGHTS_HIDDEN_OUTPUT_COUNT];
    static float tmp_bias_hidden[HIDDEN_NEURON_COUNT];
    static float tmp_bias_output[OUTPUT_NEURON_COUNT];

    std::FILE* fp{ open_file(MODEL_FILE_URL, "rb") };
    std::fread(tmp_weights_input_hidden,
               sizeof(float),
               WEIGHTS_INPUT_HIDDEN_COUNT,
               fp);
    std::fread(tmp_weights_hidden_output,
               sizeof(float),
               WEIGHTS_HIDDEN_OUTPUT_COUNT,
               fp);
    std::fread(tmp_bias_hidden,
               sizeof(float),
               HIDDEN_NEURON_COUNT,
               fp);
    std::fread(tmp_bias_output,
               sizeof(float),
               OUTPUT_NEURON_COUNT,
               fp);
    std::fclose(fp);

    weights_input_hidden.reserve(WEIGHTS_INPUT_HIDDEN_COUNT);
    for (uint32_t i{}; i < WEIGHTS_INPUT_HIDDEN_COUNT; ++i) {
        weights_input_hidden.push_back(tmp_weights_input_hidden[i]);
    }

    weights_hidden_output.reserve(WEIGHTS_HIDDEN_OUTPUT_COUNT);
    for (uint32_t i{}; i < WEIGHTS_HIDDEN_OUTPUT_COUNT; ++i) {
        weights_hidden_output.push_back(tmp_weights_hidden_output[i]);
    }

    bias_hidden.reserve(HIDDEN_NEURON_COUNT);
    for (uint32_t i{}; i < HIDDEN_NEURON_COUNT; ++i) {
        bias_hidden.push_back(tmp_bias_hidden[i]);
    }

    bias_output.reserve(OUTPUT_NEURON_COUNT);
    for (uint32_t i{}; i < OUTPUT_NEURON_COUNT; ++i) {
        bias_output.push_back(tmp_bias_output[i]);
    }

    encrypted_weights_input_hidden.reserve(HIDDEN_NEURON_COUNT);
    for (uint32_t i{}; i < HIDDEN_NEURON_COUNT; ++i) {
        std::vector<double> diag(INPUT_NEURON_COUNT);
        for (uint32_t j{}; j < INPUT_NEURON_COUNT; ++j) {
            size_t row{ j };
            size_t col{ (i + j) % HIDDEN_NEURON_COUNT };
            size_t ind{ row * INPUT_NEURON_COUNT + col };
            diag[j] = weights_input_hidden[ind];
        }
        encrypted_weights_input_hidden.push_back(cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(diag)));
    }

    encrypted_weights_hidden_output.reserve(OUTPUT_NEURON_COUNT);
    for (uint32_t i{}; i < OUTPUT_NEURON_COUNT; ++i) {
        std::vector<double> diag(HIDDEN_NEURON_COUNT);
        for (uint32_t j{}; j < HIDDEN_NEURON_COUNT; ++j) {
            size_t row{ j % OUTPUT_NEURON_COUNT };
            size_t col{ (i + j) % HIDDEN_NEURON_COUNT };
            size_t ind{ row * HIDDEN_NEURON_COUNT + col };
            diag[j] = weights_hidden_output[ind];
        }
        encrypted_weights_hidden_output.push_back(cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(diag)));
    }

    encrypted_bias_hidden = cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(bias_hidden));

    std::vector<double> bias_output_cpy(HIDDEN_NEURON_COUNT);
    bias_output_cpy[0] = bias_output[0]; bias_output_cpy[1] = bias_output[1];
    encrypted_bias_output = cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(bias_output_cpy));
}

static std::vector<int32_t> gen_index_list()
{
    std::vector<int32_t> buffer;
    for (int32_t i{ 1 }; i < 128; ++i) {
        buffer.push_back(i);
    }
    return buffer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// SERIALIZATION ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

static void SerializeCryptocontext(const CryptoContext<DCRTPoly>& cc, const char* url)
{
    if (!Serial::SerializeToFile(url, cc, SerType::BINARY)) {
        std::fprintf(stderr, "Couldn't serialize the cryptocontext to '%s'\n", url);
    }
}

static void SerializePublicKey(const KeyPair<DCRTPoly>& keyPair, const char* url)
{
    if (!Serial::SerializeToFile(url, keyPair.publicKey, SerType::BINARY)) {
        std::fprintf(stderr, "Couldn't serialize the public key to '%s'\n", url);
    }
}

static void SerializeRelinearizationKey(const CryptoContext<DCRTPoly>& cc, const char* url)
{
    std::ofstream fp(url, std::ios::out | std::ios::binary);
    if (fp.is_open()) {
        if (!cc->SerializeEvalMultKey(fp, SerType::BINARY)) {
            std::fprintf(stderr, "Couldn't serialize the relinearization key to '%s'\n", url);
        }
    }
    else {
        std::fprintf(stderr, "Couldn't create '%s' to store the relinearization key\n", url);
    }
}

static void SerializeRotationKey(const CryptoContext<DCRTPoly>& cc, const char* url)
{
    std::ofstream fp(url, std::ios::out | std::ios::binary);
    if (fp.is_open()) {
        if (!cc->SerializeEvalAutomorphismKey(fp, SerType::BINARY)) {
            std::fprintf(stderr, "Couldn't serialize the rotation key to '%s'\n", url);
        }
    }
    else {
        std::fprintf(stderr, "Couldn't create '%s' to store the rotation key\n", url);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// MAIN ////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

int main()
{
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(MULT_DEPTH);
    parameters.SetBatchSize(BATCH_SIZE);
    parameters.SetSecurityLevel(SecurityLevel::HEStd_128_quantum);

    CryptoContext<DCRTPoly> cc{ GenCryptoContext(parameters) };

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    KeyPair<DCRTPoly> keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    std::vector<int32_t> indexList = gen_index_list();
    cc->EvalRotateKeyGen(keyPair.secretKey, indexList);

#if SERIALIZE_CONTEXT == 1
    SerializeCryptocontext(cc, HADES_CRYPTOCONTEXT_FILE);
    SerializePublicKey(keyPair, HADES_PUBLICKEY_FILE);
    SerializeRelinearizationKey(cc, HADES_RELINEARIZATION_KEY_FILE);
    SerializeRotationKey(cc, HADES_ROTATION_KEY_FILE);
#endif

    Dataset dataset;
    MLP mlp(cc, keyPair);
    mlp.TestClear(dataset);
    mlp.TestCiphertext(dataset, cc, keyPair);

    return 0;
}
