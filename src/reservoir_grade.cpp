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

json d_min;
json d_max;
size_t num_bins;
size_t num_classes;

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
            const double encoder_range =
                (double)d_max.at(i) - (double)d_min.at(i);
            const double bin_width = encoder_range / num_bins;
            const double bin = encoder_range == 0 ? 0 :
                min(floor((o.features[i] - (double)d_min.at(i)) / bin_width),
                    (double)num_bins - 1);
            const int idx = (num_bins * i) + bin;

            p->apply_spike({idx, 0, 255}, false);
        }

        p->run(100);

        atom a;
        a.label = o.label;
        a.v = p->output_counts();

        pthread_mutex_lock(&out_mutex);
        outputs.push_back(std::move(a));
        pthread_mutex_unlock(&out_mutex);
    }

    return nullptr;
}

int main(int argc, char* argv[]) {
    if (argc != 9) {
        fprintf(stderr,
                "usage: %s starting_resevoir.json data.csv labels.csv threads "
                "[d_min] [d_max] num_bins num_classes\n",
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

        std::replace(line.begin(), line.end(), ',', ' ');
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

    stringstream dmin(argv[5]);
    dmin >> d_min;
    stringstream dmax(argv[6]);
    dmax >> d_max;

    sscanf(argv[7], "%zu", &num_bins);

    sscanf(argv[8], "%zu", &num_classes);

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

    vector<vector<obs>> dunn(num_classes, vector<obs>(num_classes));
    size_t total_zeros = 0;

    for (size_t i = 0; i < outputs.size(); i++) {
        // printf("%d: [", outputs[i].label);
        // for (size_t j = 0; j < outputs[i].v.size(); j++) {
        //     printf("%2d", outputs[i].v[j]);

        //     if (j != outputs[i].v.size() - 1) {
        //         printf(", ");
        //     }
        // }
        // printf("]\n");
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

    vector<vector<double>> vals(num_classes, vector<double>(num_classes));

    for (size_t i = 0; i < dunn.size(); i++) {
        for (size_t j = 0; j < dunn[i].size(); j++) {
            vals[i][j] = dunn[i][j].total / dunn[i][j].count;
            printf("%10.3f ", dunn[i][j].total / dunn[i][j].count);
        }
        puts("");
    }
    puts("");
    printf("Total zeros: %zu/%zu\n", total_zeros, outputs.size());

    bool valid = true;

    // Calculate the smallest delta between each class and the others
    for (size_t i = 0; i < vals.size(); i++) {
        double smallest = 1;
        double target = vals[i][i];

        for (size_t j = 0; j < vals.size(); j++) {
            if (i == j) {
                continue;
            }

            if (target - vals[i][j] < smallest) {
                smallest = target - vals[i][j];
            }
        }

        if (smallest < 0) {
            valid = false;
        }

        printf("Class %d smallest delta %f\n", (int)i+1, smallest);
    }

    if (!valid) {
        printf("INVALID RESERVOIR\n");
    }
}
