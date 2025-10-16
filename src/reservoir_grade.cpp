#include "framework.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
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

    fstream data(argv[2]);
    fstream labels(argv[3]);

    vector<atom> outputs;

    const size_t num_outputs = n->num_outputs();
    bool done = false;

    Processor* p = nullptr;

    json proc_params = n->get_data("proc_params");
    string proc_name = n->get_data("other")["proc_name"];
    p = Processor::make(proc_name, proc_params);
    p->load_network(n);

    if (!n) {
        fprintf(stderr, "%s: main: Unable to load network.\n", __FILE__);
    }
    n->make_sorted_node_vector();

    while (!data.eof()) {
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

        vector<int> output_counts = p->neuron_counts();
        int label;
        labels >> label;
        label -= 1;

        const int num_outputs = n->num_outputs();
        Node* node =
            n->sorted_node_vector[n->sorted_node_vector.size() - num_outputs];
        atom a;
        a.label = label;
        for (size_t i = 0; i < node->incoming.size(); i++) {
            uint32_t id = node->incoming[i]->from->id;
            a.v.push_back(output_counts[id]);
        }

        outputs.push_back(a);
    }

    vector<vector<obs>> dunn(4, vector<obs>(4));

    for (size_t i = 0; i < outputs.size(); i++) {
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
                a_norm += a.v[k] * a.v[k];
                b_norm += b.v[k] * b.v[k];
            }

            fprintf(stderr, "%f\n", acos(dot / (sqrt(a_norm) * sqrt(b_norm))));
            dunn[a.label][b.label].total +=
                acos(dot / (sqrt(a_norm) * sqrt(b_norm)));
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
}
