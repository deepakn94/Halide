#include <cstdio>
#include <chrono>
#include <iostream>

#include "conv_layer.h"
#include "conv_layer_auto_schedule.h"

#include "halide_benchmark.h"
#include "HalideBuffer.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    Buffer<float> input(4, 32, 67, 67);
    Buffer<float> filter(32, 32, 3, 3);
    Buffer<float> bias(32);

    for (int c = 0; c < input.dim(1).extent(); c++) {
        for (int n = 0; n < input.dim(0).extent(); n++) {
            for (int y = 0; y < input.dim(3).extent(); y++) {
                for (int x = 0; x < input.dim(2).extent(); x++) {
                    input(n, c, x, y) = 1.0;
                }
            }
        }
    }

    for (int w = 0; w < filter.dim(0).extent(); w++) {
        for (int x = 0; x < filter.dim(1).extent(); x++) {
            for (int y = 0; y < filter.dim(2).extent(); y++) {
                for (int z = 0; z < filter.dim(3).extent(); z++) {
                    filter(w, x, y, z) = 1.0;
                }
            }
        }
    }

    for (int x = 0; x < bias.width(); x++) {
        bias(x) = 1.0;
    }

    Buffer<float> output(4, 32, 64, 64);

    conv_layer(input, filter, bias, output);

    printf("Verifying that output matches...\n");
    for (int c = 0; c < output.dim(1).extent(); c++) {
        for (int n = 0; n < output.dim(0).extent(); n++) {
            for (int y = 0; y < output.dim(3).extent(); y++) {
                for (int x = 0; x < output.dim(2).extent(); x++) {
                    assert(output(n, c, x, y) == 289.0);  // (32 * 9) + 1.0.
                }
            }
        }
    }
    printf("Verified!\n\n");

    // Timing code

    // Auto-scheduled version
    double min_t_auto = benchmark(10, 10, [&]() {
        conv_layer_auto_schedule(input, filter, bias, output);
    });
    printf("Auto-scheduled time: %gms\n", min_t_auto * 1e3);

    return 0;
}
