#include "framework.hpp"
#include <algorithm>
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
#include <utility>

using namespace std;
using namespace neuro;
using nlohmann::json;

struct observation {
    vector<double> features;
    int label;
};

void softmax(vector<double>& x) {
    double sum = 0;

    for (double& d : x) {
        d /= 200;
    }

    for (double d : x) {
        sum += exp(d);
    }

    for (double& d : x) {
        d = exp(d) / sum;
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

struct worker_args {
    Network* n;
    vector<observation>& dataset;
    double learning_rate;
    vector<vector<int>>& conf;
    vector<vector<pair<double, int>>>& desired_edge_updates;
    double& loss;
    size_t& correct;
    size_t& total;
};

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t work = PTHREAD_COND_INITIALIZER;
pthread_cond_t done = PTHREAD_COND_INITIALIZER;
bool start_work = false;
atomic_size_t outstanding_work = 0;
const size_t batch_size = 70;
atomic_size_t work_done = batch_size + 1;
atomic_size_t work_idx;
size_t batch_idx;

void* worker(void* arg) {
    worker_args* args = (worker_args*)arg;
    Network* n = args->n;
    json proc_params = n->get_data("proc_params");
    string proc_name = n->get_data("other")["proc_name"];
    const size_t num_outputs = n->num_outputs();
    const int weight_idx = n->get_edge_property("Weight")->index;

    vector<vector<pair<double, int>>> desired_edge_updates(n->num_outputs());
    for (size_t i = 0; i < desired_edge_updates.size(); i++) {
        Node* node = n->get_output(i);

        desired_edge_updates[i].resize(node->incoming.size());
    }
    vector<vector<int>> conf(num_outputs, vector<int>(num_outputs, 0));

    while (true) {
        pthread_mutex_lock(&mutex);
        while (!start_work || work_done >= batch_size ||
               outstanding_work == 0) {
            pthread_cond_wait(&work, &mutex);
        }
        pthread_mutex_unlock(&mutex);

        Processor* p = nullptr;

        double loss = 0;
        size_t correct = 0;
        size_t total = 0;

        while (true) {
            pthread_mutex_lock(&mutex);
            size_t idx = work_idx++;
            pthread_mutex_unlock(&mutex);
            if (idx >= batch_size) {
                break;
            }

            pthread_mutex_lock(&mutex);
            observation work_item =
                args->dataset[(batch_idx * batch_size) + idx];

            outstanding_work--;
            pthread_mutex_unlock(&mutex);

            if (p == nullptr) {
                p = Processor::make(proc_name, proc_params);
                p->load_network(n);
            }

            p->clear_activity();

            for (size_t i = 0; i < work_item.features.size(); i++) {
                // int idx = 2 * i + ((int)work_item.features[i] / 50);
                int idx = 10 * i + ((int)work_item.features[i] / 10);
                p->apply_spike({idx, 0, 255}, false);

                idx++;
            }

            p->run(200);

            vector<double> output_charges = p->neuron_charges();
            vector<double> output;
            for (size_t i = 0; i < n->num_outputs(); i++) {
                output.push_back(output_charges[output_charges.size() -
                                                n->num_outputs() + i]);
            }
            vector<double> target(output.size(), 0);

            const int target_class = work_item.label - 1;
            target[target_class] = 1;

            int predicted_class = max_element(output);

            conf[target_class][predicted_class]++;
            if (predicted_class == target_class) {
                correct++;
            }
            total++;

            // Calculate sample loss

            double sample_loss = -log(output[target_class]);
            loss += sample_loss;

            // Loop through each output neuron and make weight updates
            size_t updates = 0;
            for (size_t i = 0; i < output.size(); i++) {
                Node* node = n->get_output(i);

                for (size_t j = 0; j < node->incoming.size(); j++) {
                    Edge* e = node->incoming[j];
                    double weight = (double)e->get(weight_idx);
                    double neuron_loss = output[i] - target[i];
                    double firing_count = p->neuron_counts()[e->from->id];
                    desired_edge_updates[i][j].first -=
                        args->learning_rate * neuron_loss / firing_count;
                    desired_edge_updates[i][j].second++;
                }
            }
        }

        delete p;

        // Make updates, don't forget to subtract from outstanding work
        pthread_mutex_lock(&mutex);

        // Update confusion matrix
        for (size_t i = 0; i < conf.size(); i++) {
            for (size_t j = 0; j < conf[i].size(); j++) {
                args->conf[i][j] += conf[i][j];
                conf[i][j] = 0;
            }
        }

        // Update weight updates
        for (size_t i = 0; i < desired_edge_updates.size(); i++) {
            for (size_t j = 0; j < desired_edge_updates[i].size(); j++) {
                args->desired_edge_updates[i][j].first +=
                    desired_edge_updates[i][j].first;
                args->desired_edge_updates[i][j].second +=
                    desired_edge_updates[i][j].second;
                desired_edge_updates[i][j].first = 0;
                desired_edge_updates[i][j].second = 0;
            }
        }

        args->loss += loss;
        args->correct += correct;
        args->total += total;

        work_done += total;
        pthread_mutex_unlock(&mutex);
        pthread_cond_signal(&done);
    }
};

int main(int argc, char* argv[]) {
    if (argc != 7) {
        fprintf(stderr,
                "usage: %s starting_resevoir.json data.csv labels.csv lambda "
                "learning_rate num_threads\n",
                argv[0]);
        exit(1);
    }

    json network_json;
    vector<string> json_source = {argv[1]};

    ifstream fin(argv[1]);
    fin >> network_json;

    fstream data(argv[2]);
    fstream labels(argv[3]);

    vector<observation> dataset;
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

    double lambda = 0.000001;
    double learning_rate = 0.001;

    sscanf(argv[4], "%lf", &lambda);
    sscanf(argv[5], "%lf", &learning_rate);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    Network* n = new Network();
    n->from_json(network_json);
    json proc_params = n->get_data("proc_params");
    string proc_name = n->get_data("other")["proc_name"];
    const int weight_idx = n->get_edge_property("Weight")->index;

    const size_t num_outputs = n->num_outputs();
    vector<vector<int>> conf(num_outputs, vector<int>(num_outputs, 0));

    vector<vector<pair<double, int>>> desired_edge_updates(n->num_outputs());
    for (size_t i = 0; i < desired_edge_updates.size(); i++) {
        Node* node = n->get_output(i);

        desired_edge_updates[i].resize(node->incoming.size());
    }

    double loss = 0;
    size_t correct = 0;
    size_t total = 0;

    size_t num_threads = -1;
    sscanf(argv[6], "%zu", &num_threads);

    pthread_t* threads = (pthread_t*)calloc(num_threads, sizeof(*threads));

    worker_args arg = {.n = n,
                       .dataset = dataset,
                       .learning_rate = learning_rate,
                       .conf = conf,
                       .desired_edge_updates = desired_edge_updates,
                       .loss = loss,
                       .correct = correct,
                       .total = total};

    start_work = false;

    for (size_t i = 0; i < num_threads; i++) {
        pthread_create(threads + i, NULL, worker, &arg);
    }

    for (size_t epochs = 0; epochs < 100; epochs++) {
        printf("Epoch %zu:\n", epochs);

        pthread_mutex_lock(&mutex);
        shuffle(dataset.begin(), dataset.end(), default_random_engine(seed));
        pthread_mutex_unlock(&mutex);

        // Batch loop
        for (size_t batch = 0; batch < dataset.size() / batch_size; batch++) {
            fprintf(stderr, "\033[1K\rBatch: %zu/%zu", batch + 1,
                    dataset.size() / batch_size);

            n->make_sorted_node_vector();

            work_idx = 0;
            batch_idx = batch;
            pthread_mutex_lock(&mutex);
            start_work = false;
            work_done = 0;
            outstanding_work = batch_size;
            pthread_mutex_unlock(&mutex);

            // Start work
            pthread_mutex_lock(&mutex);
            start_work = true;
            pthread_mutex_unlock(&mutex);
            pthread_cond_broadcast(&work);

            // Wait for work to be complete
            pthread_mutex_lock(&mutex);
            while (work_done != batch_size) {
                pthread_cond_wait(&done, &mutex);
            }
            pthread_mutex_unlock(&mutex);

            pthread_mutex_lock(&mutex);
            start_work = false;
            pthread_mutex_unlock(&mutex);

            for (size_t i = 0; i < num_outputs; i++) {
                Node* node = n->get_output(i);

                for (size_t j = 0; j < node->incoming.size(); j++) {
                    Edge* e = node->incoming[j];
                    if (desired_edge_updates[i][j].second != 0) {
                        double update = desired_edge_updates[i][j].first /
                                        desired_edge_updates[i][j].second;
                        // fprintf(stderr, "Update: %f\n", update);
                        if (isnormal(update)) {
                            double new_weight =
                                e->get(weight_idx) +
                                desired_edge_updates[i][j].first /
                                    desired_edge_updates[i][j].second;
                            e->set(weight_idx, new_weight);
                        }

                        desired_edge_updates[i][j].first = 0;
                        desired_edge_updates[i][j].second = 0;
                    }
                }
            }
        }

        printf("\nCONFUSION MATRIX:\n");
        for (size_t i = 0; i < num_outputs; i++) {
            for (size_t j = 0; j < num_outputs; j++) {
                printf("%4d ", conf[i][j]);
                conf[i][j] = 0;
            }
            puts("");
        }
        puts("");

        printf("Accuracy: %.2f, Loss: %.2f\n", correct / (double)total,
               loss / (double)total);

        loss = 0;
        correct = 0;
        total = 0;
    }

    for (size_t i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}
