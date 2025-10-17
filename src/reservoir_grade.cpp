#include "framework.hpp"
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <pthread.h>
#include <stddef.h>
#include <string>
#include <unistd.h>

using namespace std;
using namespace neuro;
using nlohmann::json;

struct atom {
    int label;
    vector<int> v;
};

struct obs {
    double total;
    double count;
};

struct observation {
    vector<double> features;
    int label;
};

vector<observation> dataset;
atomic_size_t dataset_idx = 0;

pthread_mutex_t in_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t out_mutex = PTHREAD_MUTEX_INITIALIZER;
vector<atom> outputs;

fstream data_file;
fstream labels;

void* worker(void* arg) {
    Network* n = (Network*)arg;
    Processor* p = nullptr;

    json proc_params = n->get_data("proc_params");
    string proc_name = n->get_data("other")["proc_name"];

    p = Processor::make(proc_name, proc_params);
    p->load_network(n);

    while (true) {
        p->clear_activity();

        size_t idx = dataset_idx++;
        if (idx > dataset.size() - 1) {
            break;
        }

        observation o = dataset[idx];
        for (size_t i = 0; i < o.features.size(); i++) {
            // int idx = 2 * i + ((int)o.features[i] / 50);
            int idx = 10 * i + ((int)o.features[i] / 10);
            p->apply_spike({idx, 0, 255}, false);

            idx++;
        }

        p->run(200);

        vector<int> output_counts = p->neuron_counts();

        const int num_outputs = n->num_outputs();
        Node* node =
            n->sorted_node_vector[n->sorted_node_vector.size() - num_outputs];
        atom a;
        a.label = o.label - 1;
        for (size_t i = 0; i < node->incoming.size(); i++) {
            uint32_t id = node->incoming[i]->from->id;
            a.v.push_back(output_counts[id]);
        }

        pthread_mutex_lock(&out_mutex);
        outputs.push_back(a);
        pthread_mutex_unlock(&out_mutex);
    }

    return nullptr;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        fprintf(
            stderr,
            "usage: %s starting_resevoir.json data.csv labels.csv threads\n",
            argv[0]);
        exit(1);
    }

    json network_json;
    vector<string> json_source = {argv[1]};

    ifstream fin(argv[1]);
    fin >> network_json;

    Network* n = new Network();
    n->from_json(network_json);

    fstream data(argv[2]);
    fstream labels(argv[3]);

    string line;

    while (!data.eof() && !labels.eof()) {
        getline(data, line);
        if (line.length() == 0) {
            break;
        }

        dataset.push_back({});

        stringstream ss(line);
        double data;
        while (ss >> data) {
            dataset.back().features.push_back(data);
        }

        int label;
        labels >> label;

        dataset.back().label = label;
    }

    size_t thread_count;
    sscanf(argv[4], "%zu", &thread_count);

    const size_t num_outputs = n->num_outputs();
    bool done = false;
    n->make_sorted_node_vector();

    // SETUP Thread pool
    pthread_t* threads = (pthread_t*)calloc(thread_count, sizeof(pthread_t));
    for (std::size_t i = 0; i < thread_count; i++) {
        pthread_create(threads + i, nullptr, worker, n);
    }

    for (std::size_t i = 0; i < thread_count; i++) {
        pthread_join(threads[i], nullptr);
    }

    vector<vector<obs>> dunn(4, vector<obs>(4));
    size_t total_zeros = 0;

    for (size_t i = 0; i < outputs.size(); i++) {
        printf("%d: [", outputs[i].label);
        for (size_t j = 0; j < outputs[i].v.size(); j++) {
            printf("%2d", outputs[i].v[j]);

            if (j != outputs[i].v.size() - 1) {
                printf(", ");
            }
        }
        printf("]\n");
        size_t zeros = 0;

        for (auto a : outputs[i].v) {
            zeros += a == 0;
        }

        if (zeros == outputs[i].v.size()) {
            total_zeros++;
            continue;
        }

        for (size_t j = 0; j < outputs.size(); j++) {
            if (i == j) {
                continue;
            }

            atom a = outputs[i];
            atom b = outputs[j];
            double dot = 0;
            double a_norm = 0;
            double b_norm = 0;
            for (size_t k = 0; k < a.v.size(); k++) {
                dot += a.v[k] * b.v[k];
                a_norm += pow(a.v[k], 2);
                b_norm += pow(b.v[k], 2);
            }

            dunn[a.label][b.label].total += dot / (sqrt(a_norm) * sqrt(b_norm));
            dunn[a.label][b.label].count++;
        }
    }

    for (size_t i = 0; i < dunn.size(); i++) {
        for (size_t j = 0; j < dunn[i].size(); j++) {
            printf("%10.3f ", dunn[i][j].total / dunn[i][j].count);
        }
        puts("");
    }
    puts("");
    printf("Total zeros: %zu/%zu\n", total_zeros, outputs.size());
}
