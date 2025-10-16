#include "framework.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <stddef.h>
#include <string>
#include <unistd.h>
#include <utility>

using namespace std;
using namespace neuro;
using nlohmann::json;

void softmax(vector<double>& x) {
    double min = 999999999;
    double max = -999999999;
    for (double d : x) {
        if (d < min) {
            min = d;
        }
        if (d > max) {
            max = d;
        }
    }

    for (double& d : x) {
        // TODO evaluate if we even want negative charges in our output layer
        // if (d < 0) {
        //     d = 0;
        // }
        d /= max;
    }

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

Network* load_network(Processor** pp, const json& network_json) {
    Network* net;
    json proc_params;
    string proc_name;
    Processor* p;

    net = new Network();
    net->from_json(network_json);

    p = *pp;
    if (p == nullptr) {
        proc_params = net->get_data("proc_params");
        proc_name = net->get_data("other")["proc_name"];
        p = Processor::make(proc_name, proc_params);
        *pp = p;
    }

    if (p->get_network_properties().as_json() !=
        net->get_properties().as_json()) {
        fprintf(stderr,
                "%s: load_network: Network and processor properties do not "
                "match.\n",
                __FILE__);
        return nullptr;
    }

    if (!p->load_network(net)) {
        fprintf(stderr, "%s: load_network: Failed to load network.\n",
                __FILE__);
        return nullptr;
    }

    return net;
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        fprintf(stderr,
                "usage: %s starting_resevoir.json data.csv labels.csv lambda "
                "learning_rate\n",
                argv[0]);
        exit(1);
    }

    json network_json;
    vector<string> json_source = {argv[1]};

    ifstream fin(argv[1]);
    fin >> network_json;

    double lambda = 0.000001;
    double learning_rate = 0.001;

    sscanf(argv[4], "%lf", &lambda);
    sscanf(argv[5], "%lf", &learning_rate);

    Network* n = new Network();
    n->from_json(network_json);
    const int weight_idx = n->get_edge_property("Weight")->index;

    for (size_t epochs = 0; epochs < 100; epochs++) {
        printf("Epoch %zu:\n", epochs);
        fstream data(argv[2]);
        fstream labels(argv[3]);

        const size_t num_outputs = n->num_outputs();
        vector<vector<int>> conf(num_outputs, vector<int>(num_outputs, 0));
        double loss = 0;
        size_t correct = 0;
        size_t total = 0;

        const size_t batch_size = 70;

        // Batch loop
        // TODO we should read in all data ahead of time (if ram usage isn't
        // that big of a deal)
        //      then we can avoid this while (true) loop and just use a
        //      calculated number of batches
        bool done = false;
        while (true) {
            if (done) {
                break;
            }
            size_t batch_samples = 0;
            vector<vector<pair<double, int>>> desired_edge_updates(
                n->num_outputs());
            for (size_t i = 0; i < desired_edge_updates.size(); i++) {
                Node* node = n->get_output(i);

                desired_edge_updates[i].resize(node->incoming.size());
            }

            Processor* p = nullptr;

            json proc_params = n->get_data("proc_params");
            string proc_name = n->get_data("other")["proc_name"];
            p = Processor::make(proc_name, proc_params);
            p->load_network(n);
            // track_all_neuron_events(p, n);

            if (!n) {
                fprintf(stderr, "%s: main: Unable to load network.\n",
                        __FILE__);
            }
            n->make_sorted_node_vector();

            while (!data.eof()) {
                batch_samples++;

                if (batch_samples > batch_size) {
                    break;
                }
                p->clear_activity();
                string line;
                getline(data, line);

                int data;
                int idx = 0;
                if (line.length() == 0) {
                    break;
                }
                stringstream ss(line);
                while (ss >> data) {
                    p->apply_spike({10 * idx + data / 10, 0, 255}, false);

                    idx++;
                }

                p->run(100);

                vector<double> output_charges = p->neuron_charges();
                vector<double> output;
                for (size_t i = 0; i < n->num_outputs(); i++) {
                    output.push_back(output_charges[output_charges.size() -
                                                    n->num_outputs() + i]);
                }
                vector<double> softmax_out(output.size());
                vector<double> target(output.size(), 0);

                labels >> data;
                const int target_class = data - 1;
                target[target_class] = 1;
                transform(output.begin(), output.end(), softmax_out.begin(),
                          [](int x) { return (double)x; });

                int predicted_class = max_element(softmax_out);

                conf[target_class][predicted_class]++;
                if (predicted_class == data - 1) {
                    correct++;
                }
                total++;

                // Calculate sample loss

                // double target_loss = -log(softmax_out[target_class]);
                double target_loss = 0;
                for (size_t c = 0; c < target.size(); c++) {
                    target_loss += pow(target[c] - softmax_out[c], 2);
                }
                // Adding norm loss
                for (size_t i = 0; i < num_outputs; i++) {
                    Node* node = n->get_output(i);

                    for (size_t j = 0; j < node->incoming.size(); j++) {
                        Edge* e = node->incoming[j];
                        // fprintf(stderr, "%f %f\n", e->get(weight_idx),
                        //         e->get(weight_idx) * lambda);
                        target_loss += lambda * e->get(weight_idx);
                    }
                }
                // Adding norm loss
                loss += target_loss;

                // Loop through each output neuron and make weight updates
                size_t updates = 0;
                for (size_t i = 0; i < softmax_out.size(); i++) {
                    Node* node = n->get_output(i);

                    for (size_t j = 0; j < node->incoming.size(); j++) {
                        Edge* e = node->incoming[j];
                        double weight = (double)e->get(weight_idx);
                        double neuron_loss = target[i] - softmax_out[i];
                        double firing_count = p->neuron_counts()[e->from->id];
                        // fprintf(stderr, "Loss: %f, Weight: %f, Update: %f\n",
                        //         neuron_loss, weight,
                        //         neuron_loss * weight * 0.01);
                        desired_edge_updates[i][j].first -=
                            learning_rate *
                            ((neuron_loss * weight /
                              (firing_count != 0 ? firing_count : 1)) +
                             (lambda * weight));
                        desired_edge_updates[i][j].second++;
                    }
                }
            }
            if (data.eof()) {
                done = true;
            }
            delete (p);

            for (size_t i = 0; i < num_outputs; i++) {
                Node* node = n->get_output(i);

                for (size_t j = 0; j < node->incoming.size(); j++) {
                    Edge* e = node->incoming[j];
                    if (desired_edge_updates[i][j].second != 0) {
                        double new_weight =
                            e->get(weight_idx) +
                            desired_edge_updates[i][j].first /
                                desired_edge_updates[i][j].second;
                        if (desired_edge_updates[i][j].first /
                                desired_edge_updates[i][j].second >
                            0.000001) {
                            // fprintf(stderr, "%f + %f = %f\n",
                            //         e->get(weight_idx),
                            //         desired_edge_updates[i][j].first /
                            //             desired_edge_updates[i][j].second,
                            //         new_weight);
                        }
                        e->set(weight_idx, new_weight);
                    }
                }
            }
        }

        printf("CONFUSION MATRIX:\n");
        for (size_t i = 0; i < num_outputs; i++) {
            for (size_t j = 0; j < num_outputs; j++) {
                printf("%4d ", conf[i][j]);
            }
            puts("");
        }
        puts("");

        printf("Accuracy: %.2f, Loss: %.2f\n", correct / (double)total,
               loss / (double)total);
    }
}
