#include "encoder.hpp"
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

int max_idx(const vector<double>& x) {
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
vector<Encoder*> encoders;
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
            vector<Spike> spikes = encoders[i]->encode(o.x[i]);

            p->apply_spikes(spikes, false);
        }

        p->run(200);

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
        replace(line.begin(), line.end(), ',', ' ');

        double data;
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

    // Create the actual encoders we're going to use
    for (size_t i = 0; i < d_min.size(); i++) {
        double min = d_min.at(i);
        double max = d_max.at(i);

        encoders.emplace_back(new BalancedEncoder((i * 2), 100, min, max));
        // encoders.emplace_back(
        //     new BinEncoder((i * num_bins), num_bins, min, max));
    }

    pthread_t* threads = (pthread_t*)calloc(num_threads, sizeof(*threads));

    for (size_t i = 0; i < num_threads; i++) {
        pthread_create(threads + i, nullptr, worker, n);
    }

    for (size_t i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    for (Encoder* e : encoders) {
        delete e;
    }

    delete n;

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

    vector<Observation> train_set;
    vector<Observation> test_set;

    for (size_t i = 0; i < processed_data.size(); i++) {
        double chance = rand() / (double)RAND_MAX;

        if (chance < 0.75) {
            train_set.push_back(processed_data[i]);
        } else {
            test_set.push_back(processed_data[i]);
        }
    }

    shuffle(train_set.begin(), train_set.end(),
            std::default_random_engine(seed));
    size_t last_correct = 0;
    for (size_t epochs = 0; epochs < total_epochs; epochs++) {
        double loss = 0;
        size_t correct = 0;
        size_t total = 0;

        const size_t batch_size = 10;

        for (size_t batch = 0; batch < train_set.size() / batch_size; batch++) {
            printf("\0331\rEpoch: %zu, Batch: %zu/%zu", epochs, batch + 1,
                   train_set.size() / batch_size);

            for (size_t idx = 0; idx < batch_size; idx++) {
                size_t work_idx = (batch * batch_size) + idx;
                Observation o = train_set[work_idx];

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

                if (max_idx(y) == o.y) {
                    correct++;
                }
                total++;

                conf[o.y][max_idx(y)]++;

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
                    desired_edge_updates[i][j].first = 0;
                    desired_edge_updates[i][j].second = 0;

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

        printf(" Accuracy: %.2f, Loss: %.2f", correct / (double)total,
               loss / (double)total);

        if (epochs == total_epochs - 1) {
            last_correct = correct;
        }
    }

    printf("\n\nTraining accuracy: %f\n",
           (double)last_correct / train_set.size());

    // printf("Final weight matrix:\n");
    // for (size_t i = 0; i < w.size(); i++) {
    //     for (size_t j = 0; j < w[i].size(); j++) {
    //         printf("%6.3f ", w[i][j]);
    //     }
    //     printf("\n");
    // }

    vector<vector<int>> test_conf(num_classes, vector<int>(num_classes, 0));
    double correct = 0;
    double test_loss = 0.0;
    for (size_t i = 0; i < test_set.size(); i++) {
        Observation o = test_set[i];

        vector<double> y_hat(num_classes);

        for (size_t i = 0; i < num_classes; i++) {
            for (size_t j = 0; j < num_outputs + 1; j++) {
                y_hat[i] += w[i][j] * o.x[j];
            }
        }

        int prediction = max_idx(y_hat);
        if (prediction == o.y) {
            correct++;
        }
        softmax(y_hat);

        test_loss += -log(y_hat[o.y]);
        test_conf[o.y][prediction]++;
    }

    printf("TEST CONFUSION MATRIX:\n");
    for (size_t i = 0; i < test_conf.size(); i++) {
        for (size_t j = 0; j < test_conf[i].size(); j++) {
            printf("%4d ", test_conf[i][j]);
            test_conf[i][j] = 0;
        }
        puts("");
    }
    puts("");

    printf("Test Accuracy: %.2f\n", correct / test_set.size());
    printf("Test Loss: %.2f\n", test_loss / test_set.size());
}
