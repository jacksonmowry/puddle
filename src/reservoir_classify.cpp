#include "framework.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <stddef.h>
#include <string>
#include <unistd.h>
#include <utility>

using namespace std;
using namespace neuro;
using nlohmann::json;

struct Observation {
    vector<double> x;
    int y;
};

void softmax(vector<double>& x) {
    double exp_sum = 0;
    for (double a : x) {
        exp_sum += exp(a);
    }

    for (double& a : x) {
        a = exp(a) / exp_sum;
    }
}

int max_element(const vector<double>& x) {
    double max_elem = -1;
    int max_idx = 0;

    for (size_t i = 0; i < x.size(); i++) {
        if (x[i] > max_elem) {
            max_elem = x[i];
            max_idx = i;
        }
    }

    return max_idx;
}

vector<Observation> processed_data;
vector<Observation> dataset;
atomic_size_t idx = 0;
json d_min;
json d_max;
size_t num_bins;

void* worker(void* arg) {
    Network* n = (Network*)arg;

    const int weight_idx = n->get_edge_property("Weight")->index;

    Processor* p = nullptr;

    json proc_params = n->get_data("proc_params");
    string proc_name = n->get_data("other")["proc_name"];
    p = Processor::make(proc_name, proc_params);
    p->load_network(n);

    if (!n) {
        fprintf(stderr, "%s: main: Unable to load network.\n", __FILE__);
    }
    n->make_sorted_node_vector();
    const size_t num_outputs = n->num_outputs();

    while (true) {
        size_t work_idx = idx++;

        if (work_idx >= dataset.size()) {
            break;
        }

        Observation o = dataset[work_idx];

        p->clear_activity();

        for (size_t i = 0; i < o.x.size(); i++) {
            const double encoder_range = (double)d_max.at(i) - (double)d_min.at(i);
            const double bin_width = encoder_range / num_bins;
            const double bin = floor((o.x[i]-(double)d_min.at(i)) / bin_width);
            const int idx = (num_bins * i) + bin;

            p->apply_spike({idx, 0, 255}, false);
        }

        p->run(100);

        // 1 for bias
        processed_data[work_idx].x.push_back(1);
        vector<int> output_counts = p->output_counts();
        for (int a : output_counts) {
            processed_data[work_idx].x.push_back(a / (double)100);
        }

        processed_data[work_idx].y = o.y;
    }

    delete p;

    return nullptr;
}

int main(int argc, char* argv[]) {
    if (argc != 12) {
        fprintf(stderr,
                "usage: %s starting_resevoir.json data.csv labels.csv "
                "learning_rate num_threads epochs lambda [d_min] [d_max] "
                "num_bins num_classes\n",
                argv[0]);
        exit(1);
    }

    json network_json;
    vector<string> json_source = {argv[1]};

    ifstream fin(argv[1]);
    fin >> network_json;

    fstream data(argv[2]);
    fstream labels(argv[3]);

    while (!data.eof()) {
        string line;
        getline(data, line);

        int data;
        int idx = 0;
        if (line.length() == 0) {
            break;
        }
        stringstream ss(line);
        dataset.push_back({});
        while (ss >> data) {
            dataset.back().x.push_back(data);
        }

        labels >> dataset.back().y;
        dataset.back().y -= 1;
    }

    double learning_rate;
    sscanf(argv[4], "%lf", &learning_rate);

    size_t num_threads;
    sscanf(argv[5], "%zu", &num_threads);

    size_t total_epochs;
    sscanf(argv[6], "%zu", &total_epochs);

    double lambda;
    sscanf(argv[7], "%lf", &lambda);

    stringstream dmin(argv[8]);
    dmin >> d_min;
    stringstream dmax(argv[9]);
    dmax >> d_max;
    d_max.at(0);
    d_max.at(1);

    sscanf(argv[10], "%zu", &num_bins);

    size_t num_classes;
    sscanf(argv[11], "%zu", &num_classes);

    Network* n = new Network();
    n->from_json(network_json);
    const int weight_idx = n->get_edge_property("Weight")->index;

    Processor* p = nullptr;

    json proc_params = n->get_data("proc_params");
    string proc_name = n->get_data("other")["proc_name"];
    p = Processor::make(proc_name, proc_params);
    p->load_network(n);

    if (!n) {
        fprintf(stderr, "%s: main: Unable to load network.\n", __FILE__);
    }
    n->make_sorted_node_vector();
    const size_t num_outputs = n->num_outputs();

    processed_data.resize(dataset.size());

    fprintf(stderr, "Preprocessing dataset\n");

    pthread_t* threads = (pthread_t*)calloc(num_threads, sizeof(*threads));

    for (size_t i = 0; i < num_threads; i++) {
        pthread_create(threads + i, nullptr, worker, n);
    }

    for (size_t i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    vector<vector<int>> conf(num_classes, vector<int>(num_classes, 0));
    vector<vector<pair<double, int>>> desired_edge_updates(
        num_classes, vector<pair<double, int>>(num_outputs + 1));

    MOA m;
    m.Seed(m.Seed_From_Time(), "rand");
    vector<vector<double>> w(num_classes, vector<double>(num_outputs + 1));
    for (size_t i = 0; i < w.size(); i++) {
        for (size_t j = 0; j < w[i].size(); j++) {
            w[i][j] = m.Random_Normal(0, 10);
        }
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    for (size_t epochs = 0; epochs < total_epochs; epochs++) {
        printf("Epoch %zu:\n", epochs);
        shuffle(processed_data.begin(), processed_data.end(),
                std::default_random_engine(seed));

        double loss = 0;
        size_t correct = 0;
        size_t total = 0;

        const size_t batch_size = 10;

        for (size_t batch = 0; batch < processed_data.size() / batch_size;
             batch++) {
            printf("\0331k\rBatch: %zu/%zu", batch + 1,
                   processed_data.size() / batch_size);

            for (size_t idx = 0; idx < batch_size; idx++) {
                size_t work_idx = (batch * batch_size) + idx;
                Observation o = processed_data[work_idx];

                // Wx + b = y
                vector<double> y(num_classes);

                for (size_t i = 0; i < num_classes; i++) {
                    for (size_t j = 0; j < num_outputs + 1; j++) {
                        y[i] += w[i][j] * o.x[j];
                    }
                }

                // Now we softmax y
                softmax(y);

                loss += -log(y[o.y]);
                vector<double> target(num_classes);
                target[o.y] = 1;

                if (max_element(y) == o.y) {
                    correct++;
                }
                total++;

                conf[o.y][max_element(y)]++;

                // Calculate weight updates
                for (size_t i = 0; i < num_classes; i++) {
                    for (size_t j = 0; j < num_outputs + 1; j++) {
                        double gradient = (y[i] - target[i]) * o.x[j];
                        desired_edge_updates[i][j].first -=
                            learning_rate * gradient + (lambda * w[i][j]);
                        desired_edge_updates[i][j].second++;
                    }
                }
            }

            // Perform batchwise weight updates
            for (size_t i = 0; i < desired_edge_updates.size(); i++) {
                for (size_t j = 0; j < desired_edge_updates[i].size(); j++) {
                    double update = desired_edge_updates[i][j].first /
                                    desired_edge_updates[i][j].second;

                    // Prevent adding nan or infinity
                    if (isnormal(update)) {
                        w[i][j] += update;
                    }
                }
            }
        }

        if (epochs == total_epochs - 1) {
            printf("CONFUSION MATRIX:\n");
            for (size_t i = 0; i < conf.size(); i++) {
                for (size_t j = 0; j < conf[i].size(); j++) {
                    printf("%4d ", conf[i][j]);
                    conf[i][j] = 0;
                }
                puts("");
            }
            puts("");
        }

        for (size_t i = 0; i < conf.size(); i++) {
            for (size_t j = 0; j < conf[i].size(); j++) {
                conf[i][j] = 0;
            }
        }

        printf("Accuracy: %.2f, Loss: %.2f\n", correct / (double)total,
               loss / (double)total);
    }

    printf("Final weight matrix:\n");
    for (size_t i = 0; i < w.size(); i++) {
        for (size_t j = 0; j < w[i].size(); j++) {
            printf("%6.3f ", w[i][j]);
        }
        printf("\n");
    }
}
