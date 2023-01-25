#pragma once
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <tuple>
#include <functional>
#include <random>

using namespace std;

namespace Mino {

template<typename Type>

class Matrix {
    size_t cols;
    size_t rows;

    public:
        vector<Type> data;
        tuple<size_t, size_t> shape;
        int numel = rows * cols;

        Matrix(size_t rows, size_t cols) : cols(cols), rows(rows), data({}) {
            data.resize(cols * rows, Type());
            shape = {rows, cols};
        }

        Matrix() : cols(0), rows(0), data({}) { shape = {rows, cols}; }

        Type &operator()(size_t row, size_t col) {
            return data[row * cols + col];
        }

        void print_shape() {
            cout << "Matrix shape = (" << rows << ", " << cols << ")\n";
        }

        void show_matrix() {
            for(size_t r = 0; r < rows; r++) {
                for(size_t c = 0; c < cols; c++) {
                    cout << (*this)(r, c) << " ";
                } cout << endl;
            } cout << endl;
        }

        Matrix matmul(Matrix &target) {
            assert(cols == target.rows);
            Matrix output(rows, target.cols);
            for(size_t r = 0; r < rows; r++)
                for(size_t c = 0; c < target.cols; c++)
                    for(size_t k = 0; k < cols; k++)
                        output(r, c) += (*this)(r, k) * target(k, c);

            return output;
        }

        Matrix multiply_elementwise(Matrix &target) {
            assert(shape == target.shape);
            Matrix output(rows, cols);
            for(size_t r = 0; r < rows; r++)
                for(size_t c = 0; c < cols; c++)
                    output(r, c) = (*this)(r, c) * target(r, c);

            return output;
        }

        Matrix square() {
            Matrix output(rows, cols);
            output = multiply_elementwise(output);
            return output;
        }

        Matrix multiply_scalar(double &scalar) {
            Matrix output(rows, cols);
            for(size_t r = 0; r < rows; r++)
                for(size_t c = 0; c < cols; c++)
                    output(r, c) = scalar * (*this)(r, c);

            return output;
        }

        Matrix operator*(double scalar) {
            return multiply_scalar(scalar);
        }

        Matrix add(Matrix &target) {
            assert(shape == target.shape);
            Matrix output(rows, cols);
            for(size_t r = 0; r < rows; r++)
                for(size_t c = 0; c < cols; c++)
                    output(r, c) = (*this)(r, c) + target(r, c);

            return output;
        }

        Matrix operator + (Matrix &target) {
            return add(target);
        }

        Matrix operator - (Matrix &target) {
            Matrix sub = target*(-1);
            return add(sub);
        }

        Matrix transpose() {
            Matrix transposed(cols, rows);
            for(size_t r = 0; r < cols; r++)
                for(size_t c = 0; c < rows; c++)
                    transposed(r, c) = (*this)(c, r);

            return transposed;
        }

        Matrix T() {
            return transpose();
        }

    Matrix apply_function(const function<Type(const Type &)> &function) {
        Matrix output(rows, cols);
        for(size_t r = 0; r < rows; r++)
            for(size_t c = 0; c < cols; c++)
                output(r, c) = function((*this)(r, c));

        return output;
    }
};



/// Matrix initialization
template <typename T>

struct mtx {
    static Matrix<T> randn(size_t rows, size_t cols) {
        Matrix<T> M(rows, cols);

        random_device rd{};
        mt19937 gen{rd()};

        // init Gaussian Distribution N(mean = 0, std = 1/sqrt(numel))
        T n(M.numel);
        T std{1 / sqrt(n)};

        normal_distribution <T> d{0, std};

        for(size_t r = 0; r < rows; r++)
            for(size_t c = 0; c < cols; c++)
                M(r, c) = d(gen);

        return M;
    }

    static Matrix<T> rand(size_t rows, size_t cols) {
        Matrix<T> M{rows, cols};

        random_device rd{};
        mt19937 gen{rd()};
        uniform_real_distribution<T> d{0, 1};

        for(size_t r = 0; r < rows; ++r)
            for(int c = 0; c < cols; ++c)
                M(r, c) = d(gen);

        return M;
    }
};

}
