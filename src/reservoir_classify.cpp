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
        // We may need this not sure
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
    int max_idx = -1;

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
    if (argc != 4) {
        fprintf(stderr,
                "usage: %s starting_resevoir.json data.csv labels.csv\n",
                argv[0]);
        exit(1);
    }

    json network_json;
    vector<string> json_source = {argv[1]};

    ifstream fin(argv[1]);
    fin >> network_json;

    Network* n = new Network();
    n->from_json(network_json);
    const int weight_idx = n->get_edge_property("Weight")->index;

    for (size_t epochs = 0; epochs < 100; epochs++) {
        printf("Epoch %zu:\n", epochs);
        fstream data(argv[2]);
        fstream labels(argv[3]);

        int conf[4][4] = {0};
        double loss = 0;
        size_t correct = 0;
        size_t total = 0;

        const size_t batch_size = 70;


        // Batch loop
        // TODO we should read in all data ahead of time (if ram usage isn't that big of a deal)
        //      then we can avoid this while (true) loop and just use a calculated number of batches
        bool done = false;
        while (true) {
            if (done) {
                break;
            }
            size_t batch_samples = 0;
            vector<vector<pair<int, int>>> desired_edge_updates(
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
                vector<int> output;
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
                          [](int x) { return (double)x /*/ 100*/; });

                softmax(softmax_out);
                int predicted_class = max_element(softmax_out);
                // printf("predicted class: %d | ", predicted_class);

                // printf("[%2d, %2d, %2d, %2d] |\n", output[0], output[1],
                // output[2],
                //        output[3]);
                // printf("[%.2f, %.2f, %.2f, %.2f] |", softmax_out[0],
                //        softmax_out[1], softmax_out[2], softmax_out[3]);
                // exit(1);

                conf[target_class][predicted_class]++;
                if (predicted_class == data - 1) {
                    correct++;
                }
                total++;

                double target_loss = -log(softmax_out[target_class]);
                loss += target_loss;

                // printf("%9s |",
                //        predicted_class == data - 1 ? "CORRECT" :
                //        "INCORRECT");
                // printf("Target: [");
                // for (size_t i = 0; i < target.size(); i++) {
                //     printf("%.2f", target[i]);
                //     if (i != target.size() - 1) {
                //         printf(", ");
                //     }
                // }
                // printf("], Output: [");
                // for (size_t i = 0; i < softmax_out.size(); i++) {
                //     printf("%.2f", softmax_out[i]);
                //     if (i != softmax_out.size() - 1) {
                //         printf(", ");
                //     }
                // }
                // printf("]\n");

                // Loop through each output neuron and make weight updates
                size_t updates = 0;
                for (size_t i = 0; i < softmax_out.size(); i++) {
                    Node* node = n->get_output(i);

                    for (size_t j = 0; j < node->incoming.size(); j++) {
                        Edge* e = node->incoming[j];
                        size_t firing_count =
                            p->neuron_counts()[e->from->id];
                        int weight = (int)e->get(weight_idx);
                        double neuron_loss = target[i] - softmax_out[i];
                        int weight_delta = 15 * neuron_loss * firing_count;
                        // fprintf(stderr, "Update of %d\n", weight_delta);
                        if (weight_delta != 0) {
                            desired_edge_updates[i][j].first += weight_delta;
                            desired_edge_updates[i][j].second++;
                        }
                    }
                    // fprintf(stderr, "\n");
                }
            }
            if (data.eof()) {
                done = true;
            }
            delete (p);

            for (size_t i = 0; i < 4; i++) {
                Node* node = n->get_output(i);

                for (size_t j = 0; j < node->incoming.size(); j++) {
                    Edge* e = node->incoming[j];
                    if (desired_edge_updates[i][j].second != 0 &&
                        e->get(weight_idx) > -255 && e->get(weight_idx) < 255) {
                        int new_weight = (int)e->get(weight_idx) +
                                         desired_edge_updates[i][j].first /
                                             desired_edge_updates[i][j].second;
                        // fprintf(stderr, "%d ",
                        //         desired_edge_updates[i][j].first /
                        //             desired_edge_updates[i][j].second);
                        if (new_weight < -255) {
                            new_weight = -255;
                        } else if (new_weight > 255) {
                            new_weight = 255;
                        }
                        e->set(weight_idx, new_weight);
                        // fprintf(stderr, "%d ",
                        //         (int)desired_edge_updates[i][j].first /
                        //             desired_edge_updates[i][j].second);
                    }
                }
                // fprintf(stderr, "\n");
            }
        }

        printf("CONFUSION MATRIX:\n");
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                printf("%4d ", conf[i][j]);
            }
            puts("");
        }
        puts("");

        printf("Accuracy: %.2f, Loss: %.2f\n", correct / (double)total,
               loss / (double)total);
        // sleep(2);
    }
}
