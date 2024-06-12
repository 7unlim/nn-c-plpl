#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib> 

int getLength() {
    std::ifstream ifs("encoded_data.csv");

    if (!ifs.is_open()) {
        throw std::runtime_error("Error opening file");
    }

    std::string line;

    std::getline(ifs, line);
    ifs.close();

    std::stringstream headerStream(line);
    int header_count = 0;
    while (std::getline(headerStream, line, ',')) {
        header_count++;
    }
    
    return header_count;
}

std::vector<double> initializeWeights(size_t num_weights) {
    std::vector<double> weights(num_weights);

    srand(static_cast<unsigned>(time(0)));

    double lower = -(1.0 / sqrt(num_weights));
    double upper = (1.0 / sqrt(num_weights));

    for (auto& weight : weights) {
        double random = static_cast<double>(rand()) / RAND_MAX;
        weight = lower + random * (upper - lower);
    }

    return weights;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

std::pair<std::vector<std::vector<double>>, std::vector<double>> getInputAndOutput(const std::string& file) {
    std::vector<std::vector<double>> input_layer;
    std::vector<double> output_layer;

    std::ifstream ifs(file);

    if (!ifs.is_open()) {
        throw std::runtime_error("Error opening file");
    }

    std::string line;

    std::getline(ifs, line);

    while (std::getline(ifs, line)) {
        std::vector<double> data_holder;
        std::istringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            try {
                double converted_cell = std::stod(cell);
                data_holder.push_back(converted_cell);
            } catch (const std::invalid_argument &e) {
                std::cerr << "Error converting string to double, setting default value 0.0" << std::endl;
                data_holder.push_back(0);
            }
        }
        input_layer.push_back(data_holder);
        output_layer.push_back(data_holder.back());
        data_holder.pop_back(); 
    }
    ifs.close();

    return std::make_pair(input_layer, output_layer);
}

std::vector<double> feedForwardHidden(const std::vector<double>& input_vector, const std::vector<double>& weights) {
    std::vector<double> hidden_layer;
    size_t input_size = input_vector.size();
    size_t num_hidden_neurons = weights.size() / input_size;

    for (size_t j = 0; j < num_hidden_neurons; ++j) {
        double weighted_sum = 0.0;
        for (size_t i = 0; i < input_size; ++i) {
            weighted_sum += input_vector[i] * weights[j * input_size + i];
        }
        // weighted_sum += 0.1; // Bias
        hidden_layer.push_back(sigmoid(weighted_sum));
    }

    return hidden_layer;
}

double feedForwardOutput(const std::vector<double>& hidden_layer, const std::vector<double>& weights) {
    double output = 0.0;
    size_t hidden_size = hidden_layer.size();

    for (size_t i = 0; i < hidden_size; ++i) {
        output += hidden_layer[i] * weights[i];
    }
    // output += 0.1; // Bias

    return sigmoid(output);
}

std::vector<std::vector<double>> feedForward(const std::string& file) {
    std::pair<std::vector<std::vector<double>>, std::vector<double>> input_output_pair = getInputAndOutput(file);
    std::vector<std::vector<double>> input_layer = input_output_pair.first;
    std::vector<double> output_layer = input_output_pair.second;
    int input_neurons = getLength() - 1; 
    int output_neurons = 1;
    int hidden_neurons = (input_neurons + output_neurons) / 2;

    std::vector<double> weights_input_hidden = initializeWeights(input_neurons * hidden_neurons);
    std::vector<double> weights_hidden_output = initializeWeights(hidden_neurons * output_neurons);

    std::vector<std::vector<double>> outputs;

    for (size_t i = 0; i < input_layer.size(); ++i) {
        std::vector<double> hidden_layer_output = feedForwardHidden(input_layer[i], weights_input_hidden);
        double final_output = feedForwardOutput(hidden_layer_output, weights_hidden_output);
        outputs.push_back({final_output});
    }

    for (const auto& output : outputs) {
        for (const auto& o : output) {
            std::cout << "{" << o << "}" << std::endl;
        }
    }

    return outputs;
}

double calculate_loss(double prediction, double label) {
    // Binary Cross Entropy Loss Function
    return -1 * (label * log(prediction) + (1 - label) * log(1 - prediction));
}

void backward_propagation(const std::vector<double>& hidden_layer_output, double final_output, double label,
                          std::vector<double>& weights_hidden_output, std::vector<double>& weights_input_hidden,
                          const std::vector<double>& input_vector, double learningRate) {
    double output_error = final_output - label;
    double output_gradient = output_error * final_output * (1 - final_output);

    for (size_t i = 0; i < hidden_layer_output.size(); ++i) {
        double delta_weight = -learningRate * output_gradient * hidden_layer_output[i];
        weights_hidden_output[i] += delta_weight;
    }


    std::vector<double> hidden_gradients;
    for (size_t i = 0; i < hidden_layer_output.size(); ++i) {
        double hidden_gradient = hidden_layer_output[i] * (1 - hidden_layer_output[i]);
        double sum = 0.0;
        for (size_t j = 0; j < weights_hidden_output.size(); ++j) {
            sum += output_gradient * weights_hidden_output[j];
        }
        hidden_gradients.push_back(hidden_gradient * sum);
    }

    for (size_t i = 0; i < input_vector.size(); ++i) {
        for (size_t j = 0; j < hidden_gradients.size(); ++j) {
            double delta_weight = -learningRate * hidden_gradients[j] * input_vector[i];
            weights_input_hidden[j * input_vector.size() + i] += delta_weight;
        }
    }
}

// void update_weights(const std::vector<double>& weights_hidden_output, const std::vector<double>& weights_input_hidden,
//                     std::vector<double>& updated_weights_hidden_output, std::vector<double>& updated_weights_input_hidden) {
//     updated_weights_hidden_output = weights_hidden_output;
//     updated_weights_input_hidden = weights_input_hidden;
// }

void train() {
    std::pair<std::vector<std::vector<double>>, std::vector<double>> input_output_pair = getInputAndOutput("encoded_data.csv");
    std::vector<std::vector<double>> input_layer = input_output_pair.first;
    std::vector<double> output_layer = input_output_pair.second;

    int input_neurons = getLength() - 1; // Subtract 1 for the output column
    int output_neurons = 1;
    int hidden_neurons = (input_neurons + output_neurons) / 2;

    std::vector<double> weights_input_hidden = initializeWeights(input_neurons * hidden_neurons);
    std::vector<double> weights_hidden_output = initializeWeights(hidden_neurons * output_neurons);

    int epochs = 100;
    double learningRate = 0.01; 

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;
        for (size_t i = 0; i < input_layer.size(); ++i) {
            std::vector<double> hidden_layer_output = feedForwardHidden(input_layer[i], weights_input_hidden);
            double final_output = feedForwardOutput(hidden_layer_output, weights_hidden_output);

            if (output_layer[i] > 0.5) {
                output_layer[i] = 1;
            } else {
                output_layer[i] = 0;
            }

            double loss = calculate_loss(final_output, output_layer[i]);
            totalLoss += loss;

            backward_propagation(hidden_layer_output, final_output, output_layer[i],
                                 weights_hidden_output, weights_input_hidden, input_layer[i], learningRate);
        }

        double averageLoss = totalLoss / input_layer.size();
        std::cout << "Epoch " << epoch + 1 << ", Average Loss: " << averageLoss << std::endl;
    }
} 