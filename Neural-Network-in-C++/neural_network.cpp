#pragma once
#include "matrix.h"
#include "mlp.h"
#include<bits/stdc++.h>
#define pb push_back

using namespace std;
using namespace Mino;
using namespace mlp;

void log(const auto iter, const auto &x, const auto &y, const auto &y_hat){
    auto mse = (y.data[0] - y_hat.data[0]);
    mse = mse*mse;
    cout << "Epoch " << iter << ": ";
    cout << "Error:" << mse << " x:"
       << x.data[0] << " y:"
       << y.data[0] << " predict:"
       << y_hat.data[0] << " \n";
}

auto make_model(size_t in_channels, size_t out_channels, size_t hidden_units, size_t hidden_layers, double step) {

    vector<size_t> units_per_layer;
    units_per_layer.pb(in_channels);
    for(int i = 1; i <= hidden_layers; i++)
        units_per_layer.pb(hidden_units);
    units_per_layer.pb(out_channels);

    MLP<double> model(units_per_layer, 0.01f);
    return model;
}

double create_data(double x) {
    return sin(x) * sin(x);
}

int main() {

    int in_channels{1}, out_channels{1}, hidden_units_per_layer{8}, hidden_layers{3};
    double step{.5f};
    auto model = make_model(
      in_channels=1,
      out_channels=1,
      hidden_units_per_layer=8,
      hidden_layers=3,
      step=.5f);

    int max_iter = 1000;
    const double Pi = 3.14159;

    for(int i = 1; i <= max_iter; i++) {
        auto x = mtx<double> :: rand(in_channels, 1) * Pi;
        auto y = x.apply_function(create_data);

        auto y_hat = model.forward(x);
        model.backprop(y);

        /// mse - x - y - x^
        log(i, x, y, y_hat);
    }



}
