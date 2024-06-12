#ifndef NN
#define NN

#include <vector>

int getLength();
std::vector<double> initializeWeights(size_t num_weights);
double sigmoid(double x);
std::vector<double> sigmoidDerivative(const std::vector<double>& z);
std::vector<std::vector<double>> getInputLayer();
std::vector<double> getHiddenLayer();
std::vector<double> feedForwardHidden(std::vector<double>& input_vector, std::vector<double>& weights);
double feedForwardOutput(std::vector<double> hidden_layer, std::vector<double> weights);
std::vector<double> feedForward(const std::string& file);
// void train(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& labels, int epochs, double learningRate);
void train();

#endif