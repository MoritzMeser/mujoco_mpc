//
// Created by Moritz Meser on 26.04.24.
//

#include "reward_computation.h"
#include <cmath>
#include <stdexcept>
#include "mujoco/mujoco.h"

double sigmoid(double x, double value_at_1, std::string sigmoid_type) {
    if (sigmoid_type == "cosine" || sigmoid_type == "linear" || sigmoid_type == "quadratic") {
        if (!(0 <= value_at_1 && value_at_1 <= 1)) {
            throw std::invalid_argument("Value at 1 must be in [0, 1].");
        }
    } else {
        if (!(0 < value_at_1 && value_at_1 < 1)) {
            throw std::invalid_argument("Value at 1 must be in (0, 1).");
        }
    }
    if (sigmoid_type == "gaussian") {
        double scale = std::sqrt(-2 * std::log(value_at_1));
        return std::exp(-0.5 * std::pow(x * scale, 2));
    } else if (sigmoid_type == "linear") {
        double scale = 1.0 - value_at_1;
        double scaled_x = x * scale;
        return std::abs(scaled_x) < 1.0 ? 1.0 - scaled_x : 0.0;
    } else if (sigmoid_type == "quadratic") {
        double scale = std::sqrt(1 - value_at_1);
        double scaled_x = x * scale;
        return std::abs(scaled_x) < 1.0 ? 1.0 - scaled_x * scaled_x : 0.0;
        // TODO: in the python implementation there are some more sigmoid types, but they are currently not used
    } else {
        throw std::invalid_argument("Unknown sigmoid type.");
    }
}

double tolerance(double x, std::pair<double, double> bounds, double margin,
                 std::string sigmoid_type, double value_at_margin) {
    double lower = bounds.first;
    double upper = bounds.second;

    if (lower > upper) {
        throw std::invalid_argument("Lower bound must be <= than upper bound.");
    }
    if (margin < 0) {
        throw std::invalid_argument("Margin must be >= 0.");
    }
    bool in_bounds = lower <= x && x <= upper;
    double value;
    if (margin == 0) {
        value = in_bounds ? 1.0 : 0.0;
    } else {
        double distance = std::min(std::abs(x - lower), std::abs(x - upper)) / margin;
        if (in_bounds) {
            value = 1.0;
        } else {
            value = sigmoid(distance, value_at_margin, sigmoid_type);
        }
    }
    return value;
}


void rotateVector(mjtNum *vector, const mjtNum *rotationMatrix, const mjtNum *inputVector) {
    for (int i = 0; i < 3; i++) {
        vector[i] = 0;
        for (int j = 0; j < 3; j++) {
            vector[i] += rotationMatrix[i + 3 * j] * inputVector[j];
        }
    }
}