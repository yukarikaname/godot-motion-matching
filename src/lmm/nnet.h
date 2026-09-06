#pragma once

#include <cstdio>
#include <string>
#include <vector>

// Minimal feed-forward MLP (orangeduck-style) used by Learned Motion Matching. Reads the
// .bin layout produced by train_lmm.py:
//   u32 in_mean_n, f32[in], u32 in_std_n, f32[in],
//   u32 out_mean_n, f32[out], u32 out_std_n, f32[out],
//   u32 n_layers,
//   per layer: u32 Wrows, u32 Wcols, f32[Wrows*Wcols], u32 bcount, f32[bcount]
// weights are stored row-major as [out][in] (train_lmm writes weight.T). Activation is
// ReLU on every layer except the last.

class MMNN {
public:
    std::vector<float> input_mean, input_std;
    std::vector<float> output_mean, output_std;
    std::vector<std::vector<float>> weights;  // each [out][in] row-major
    std::vector<std::vector<float>> biases;   // each [out]
    int in_dim = 0, out_dim = 0;

    bool load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) { fprintf(stderr, "[MMNN] cannot open %s\n", path.c_str()); return false; }
        uint32_t n;
        auto read_vec = [&](std::vector<float>& v) -> bool {
            if (fread(&n, sizeof(uint32_t), 1, f) != 1) return false;
            v.resize(n);
            if (n && fread(v.data(), sizeof(float), n, f) != n) return false;
            return true;
        };
        if (!read_vec(input_mean) || !read_vec(input_std) ||
            !read_vec(output_mean) || !read_vec(output_std)) {
            fclose(f); return false;
        }
        if (fread(&n, sizeof(uint32_t), 1, f) != 1) { fclose(f); return false; }
        in_dim = (int)input_mean.size();
        out_dim = (int)output_mean.size();
        weights.resize(n);
        biases.resize(n);
        for (uint32_t l = 0; l < n; l++) {
            uint32_t rows, cols;
            if (fread(&rows, sizeof(uint32_t), 1, f) != 1 ||
                fread(&cols, sizeof(uint32_t), 1, f) != 1) { fclose(f); return false; }
            weights[l].resize((size_t)rows * cols);
            size_t gotW = weights[l].size() ?
                fread(weights[l].data(), sizeof(float), weights[l].size(), f) : 0;
            if (gotW != weights[l].size()) { fclose(f); return false; }
            uint32_t bcount;
            if (fread(&bcount, sizeof(uint32_t), 1, f) != 1) { fclose(f); return false; }
            biases[l].resize(bcount);
            size_t gotB = bcount ? fread(biases[l].data(), sizeof(float), bcount, f) : 0;
            if (gotB != bcount) { fclose(f); return false; }
        }
        fclose(f);
        return true;
    }

    // x: [in_dim]; writes [out_dim]
    void evaluate(const float* x, float* out) const {
        std::vector<float> cur(x, x + in_dim);
        std::vector<float> next;
        for (size_t l = 0; l < weights.size(); l++) {
            const int out_n = (int)biases[l].size();
            next.assign(out_n, 0.0f);
            const int in_n = (int)cur.size();
            // bias
            for (int j = 0; j < out_n; j++) next[j] = biases[l][j];
            // weights stored [out][in]
            for (int i = 0; i < in_n; i++) {
                if (cur[i] != 0.0f) {
                    const float* wrow = &weights[l][(size_t)i * out_n];
                    for (int j = 0; j < out_n; j++) next[j] += cur[i] * wrow[j];
                }
            }
            if (l != weights.size() - 1) {
                for (int j = 0; j < out_n; j++) next[j] = next[j] > 0.0f ? next[j] : 0.0f;
            }
            cur.swap(next);
        }
        // denormalize
        for (int j = 0; j < out_dim; j++) {
            out[j] = cur[j] * output_std[j] + output_mean[j];
        }
    }
};
