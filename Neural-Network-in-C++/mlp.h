#pragma once
#include "matrix.h"
#include <random>
#include <utility>
#include <cassert>

#define FOR(i, a, b) for(size_t i = a; i <= b; i++)
#define FOR_(i, a, b) for(size_t i = a; i < b; i++)
#define pb push_back

namespace mlp {

using namespace std;
using namespace Mino;

inline auto sigmoid(double x) {
    return 1.0f / (1 + exp(-x));
}
inline auto d_sigmoid(double x) {
    return x / (1 - x);
}

template<typename T>
class MLP {

public:
    vector<size_t> units_per_layer;
    vector<Matrix<T>> bias_vectors;
    vector<Matrix<T>> weight_matrices;
    vector<Matrix<T>> activations;

    double step;

    explicit MLP(vector<size_t> units_per_layer, double step = .001f):
        units_per_layer(units_per_layer),
        weight_matrices(),
        bias_vectors(),
        activations(),
        step(step) {
            FOR_(i, 0, units_per_layer.size() - 1) {
                size_t in_channels{units_per_layer[i]}, out_channels{units_per_layer[i+1]};
                Matrix<T> W  = Mino::mtx<T> :: randn(out_channels, in_channels);
                weight_matrices.push_back(W);

                Matrix<T> b  = Mino::mtx<T>::randn(out_channels, 1);
                bias_vectors.push_back(b);

                activations.resize(units_per_layer.size());
            }

        }

    Matrix<T> forward(Matrix<T> x) {
        assert(get<0>(x.shape) == units_per_layer[0] && get<1>(x.shape));

        activations[0] = x;
        Matrix<T> prev = x;

        for(int i = 0; i < units_per_layer.size() - 1; ++i) {
//            prev.print_shape(); weight_matrices[i].print_shape(); cout << endl;
            Matrix<T> y = weight_matrices[i].matmul(prev);
            y = y + bias_vectors[i];
            y = y.apply_function(mlp::sigmoid);
            activations[i+1] = y;
            prev = y;

        }
        return prev;
    }

    Matrix<T> operator()(Matrix<T> x) {
        return forward(x);
    }

    void backprop(Matrix<T> target) {
        assert(get<0>(target.shape) == units_per_layer.back());

        auto y = target;
        auto y_hat = activations.back();
        auto error = target - y_hat;

        for(int i = weight_matrices.size() - 1; i >= 0; i--) {
            auto Wt = weight_matrices[i].T();
            auto prev_errors = Wt.matmul(error);

            auto d_outputs = activations[i+1].apply_function(d_sigmoid);
            auto gradients = error.multiply_elementwise(d_outputs);
            gradients = gradients * step;

            auto a_trans = activations[i].T();
            auto weight_gradients = gradients.matmul(a_trans);

            bias_vectors[i] = bias_vectors[i] + gradients;
            weight_matrices[i] = weight_matrices[i] + weight_gradients;
            error = prev_errors;
        }
    }
};

}
