#include <numpp/matrix.hpp>
#include <vector>
#include <random>

struct TrainingSample {
    numpp::matrixf input;
    numpp::matrixf output;
};

struct TestSample {
    numpp::matrixf input;
    size_t label;
};

struct Gradients {
    std::vector<numpp::matrixf> biases;
    std::vector<numpp::matrixf> weights;
};

using Dataset = std::vector<TrainingSample>;
using TestDataset = std::vector<TestSample>;

template<numpp::numpp_matrix EXPR>
auto sigmoid(const EXPR& z) {
    return 1.0 / (1.0 + (-z).exp());
}

template<numpp::numpp_matrix EXPR>
auto sigmoid_prime(const EXPR& z) {
    return numpp::elementwise_mul(sigmoid(z), 1- sigmoid(z));
}

class Network {
    private:
        size_t num_layers;
        std::vector<size_t> sizes;
        
        std::vector<numpp::matrixf> biases;
        std::vector<numpp::matrixf> weights;
    public:
        Network(const std::vector<size_t>& sizes__)
        : num_layers(sizes__.size()),sizes(sizes__) {
            weights.resize(num_layers);
            biases.resize(num_layers);
            for (size_t l = 1; l < num_layers; ++l) {
                weights[l] = numpp::matrixf::randn(sizes[l], sizes[l-1]);
                biases[l]  = numpp::matrixf::randn(sizes[l],          1);
            }
        }

        numpp::matrixf feedforward(numpp::matrixf a) const {
            for (size_t l = 1; l < num_layers; ++l)
                a = sigmoid(weights[l] * a + biases[l]);
            return a;
        }

        void train(Dataset training_data, size_t epochs, size_t mini_batch_size, float eta, const TestDataset* test_data = nullptr) {
            const size_t n = training_data.size();
            const size_t n_test = test_data ? test_data->size() : 0;

            std::mt19937 rng(std::random_device{}());
            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(training_data.begin(), training_data.end(), rng);
                for (size_t k = 0; k < n; k += mini_batch_size) {
                    const size_t end = std::min(k + mini_batch_size, n);
                    Dataset mini_batch(
                        training_data.begin() + k,
                        training_data.begin() + end
                    );
                    update_mini_batch(mini_batch, eta);
                }

                if (test_data) {
                    std::cout
                        << "Epoch " << epoch
                        << ": "
                        << evaluate(*test_data)
                        << " / "
                        << n_test
                        << '\n';
                } else {
                    std::cout
                        << "Epoch "
                        << epoch
                        << " complete\n";
                }
            }
        }

        void update_mini_batch(const Dataset& mini_batch, float eta) {
            std::vector<numpp::matrixf> nabla_b(num_layers);
            std::vector<numpp::matrixf> nabla_w(num_layers);
            
            bool first = true;
            for (const auto& sample : mini_batch) {
                auto gradients = backprop(sample.input, sample.output);
                if (first) {
                    nabla_b = gradients.biases;
                    nabla_w = gradients.weights;
                    first = false;
                    continue;
                }

                for (size_t l = 1; l < num_layers; ++l) {
                    nabla_b[l] += gradients.biases[l];
                    nabla_w[l] += gradients.weights[l];
                }
            }

            const float scale = eta / mini_batch.size();
            for (size_t l = 1; l < num_layers; ++l) {
                weights[l] -= scale * nabla_w[l];
                biases[l]  -= scale * nabla_b[l];
            }
        }

        Gradients backprop(const numpp::matrixf& x, const numpp::matrixf& y) {
            std::vector<numpp::matrixf> nabla_b(num_layers);
            std::vector<numpp::matrixf> nabla_w(num_layers);

            auto activation = x;
            std::vector<numpp::matrixf> activations;
            std::vector<numpp::matrixf> zs;

            activations.push_back(x);
            for (size_t l = 1; l < num_layers; ++l) {
                auto z = weights[l] * activation + biases[l];
                zs.push_back(z);
                activation = sigmoid(z);
                activations.push_back(activation);
            }

            auto delta = numpp::elementwise_mul(cost_derivative(activations.back(), y), sigmoid_prime(zs.back()));

            nabla_b.back() = delta;
            nabla_w.back() = delta * activations[activations.size() - 2].transpose();

            for (size_t l = 2; l < num_layers; ++l) {
                auto z  = zs[zs.size() - l];
                auto sp = sigmoid_prime(z);
                delta = numpp::elementwise_mul(weights[num_layers - l + 1].transpose() * delta, sp);
                nabla_b[num_layers - l] = delta;
                nabla_w[num_layers - l] = delta * activations[activations.size() - l - 1].transpose();
            }

            return {
                std::move(nabla_b),
                std::move(nabla_w)
            };
        }

        size_t evaluate(const TestDataset& test_data) const {
            size_t correct = 0;
            for (const auto& [input, label] : test_data) {
                const auto prediction = argmax(feedforward(input));
                if (prediction == label)
                    ++correct;
            }
            return correct;
        }

        numpp::matrixf cost_derivative(const numpp::matrixf& output_activations, const numpp::matrixf& y) const {
            return output_activations - y;
        }
};

int main() {
    Network nn({2,2,1});

    Dataset dataset = {
        { {{0}, {0}}, {{0}} },
        { {{0}, {1}}, {{1}} },
        { {{1}, {0}}, {{1}} },
        { {{1}, {1}}, {{0}} }
    };

    nn.train(dataset, 10000, 4, 3.0);

    for (const auto& sample : dataset) {
        const auto& input = sample.input;
        const auto output = nn.feedforward(input);
        auto predict = numpp::round(output(0,0));
        std::cout
        << "\npredict: "
        << predict
        << " y: "
        << sample.output(0,0);
    }
}