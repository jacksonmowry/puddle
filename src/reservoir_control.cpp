#include "framework.hpp"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <pthread.h>
#include <stddef.h>
#include <string>
#include <type_traits>
#include <unistd.h>

using namespace std;
using namespace neuro;
using nlohmann::json;

class Encoder {
  public:
    Encoder(size_t _starting_neuron) : starting_neuron(_starting_neuron) {}
    virtual ~Encoder() {};
    virtual size_t num_neurons() = 0;
    virtual vector<Spike> encode(double x) = 0;

    size_t starting_neuron;
};

// Balances spikes between 2 neurons for values within the given range of [dmin,
// dmax]
// At the extremes `timesteps` will be produced for one neuron while the other
// will receive 0, if this is not desired extent dmin and dmax accordingly
class BalancedEncoder : public Encoder {
  public:
    BalancedEncoder(size_t starting_neuron, size_t _timesteps, double _dmin,
                    double _dmax)
        : Encoder(starting_neuron), timesteps(_timesteps), dmin(_dmin),
          dmax(_dmax) {
        range = dmax - dmin;
    }

    virtual size_t num_neurons() { return 2; }
    virtual vector<Spike> encode(double x) {
        vector<Spike> spikes;
        double frac = (x - dmin) / range;

        for (size_t i = 0; i < frac * timesteps; i++) {
            spikes.emplace_back(starting_neuron, i, 255);
        }

        for (size_t i = 0; i < (1 - frac) * timesteps; i++) {
            spikes.emplace_back(starting_neuron + 1, i, 255);
        }

        return std::move(spikes);
    }

    size_t timesteps;
    double dmin;
    double dmax;
    double range;
};

// Produces single spikes into bins
class BinEncoder : public Encoder {
  public:
    BinEncoder(size_t starting_neuron, size_t _num_bins, double _dmin,
               double _dmax)
        : Encoder(starting_neuron), num_bins(_num_bins), dmin(_dmin),
          dmax(_dmax) {
        range = dmax - dmin;
        bin_width = range / num_bins;
    }

    virtual size_t num_neurons() { return num_bins; }
    virtual vector<Spike> encode(double x) {
        vector<Spike> spikes;

        const int bin =
            min(floor((x - dmin) / bin_width), (double)num_bins - 1);
        const int idx = starting_neuron + bin;

        spikes.emplace_back(idx, 0, 255);

        return std::move(spikes);
    }

    size_t num_bins;
    size_t bin_width;
    double dmin;
    double dmax;
    double range;
};

struct ObsReward {
    vector<double> obs;
    double reward;
    bool done;
};

class App {
  public:
    App(size_t num_observations, vector<double> dmin, vector<double> dmax,
        size_t num_actions) {
        this->num_observations = num_observations;
        this->dmin = dmin;
        this->dmax = dmax;

        this->num_actions = num_actions;
    }

    virtual ~App() {};
    virtual ObsReward step(size_t action) = 0;
    virtual ObsReward reset() = 0;
    virtual void print() = 0;

    size_t num_observations;
    vector<double> dmin;
    vector<double> dmax;

    size_t num_actions;
};

// X = 1, O = -1, X always goes first
class TicTacToe : public App {
  public:
    TicTacToe()
        : App(9, {-1, -1, -1, -1, -1, -1, -1, -1, -1},
              {1, 1, 1, 1, 1, 1, 1, 1, 1}, 9) {
        reset();
    }
    virtual ObsReward step(size_t action) {
        bool valid_move = board[action] == 0;

        if (!valid_move) {
            // Currently we're going to make invalid moved heavily unfavored and
            // not let the AI move, hopefully this will discourage invalid moves
            // entirely
            return (ObsReward){.obs = board, .reward = -10000, .done = false};
        }

        board[action] = player_symbol;

        // Check if player wins
        if (board[0] == player_symbol && board[1] == player_symbol &&
                board[2] == player_symbol ||
            board[3] == player_symbol && board[4] == player_symbol &&
                board[5] == player_symbol ||
            board[6] == player_symbol && board[7] == player_symbol &&
                board[8] == player_symbol ||
            board[0] == player_symbol && board[3] == player_symbol &&
                board[6] == player_symbol ||
            board[1] == player_symbol && board[4] == player_symbol &&
                board[7] == player_symbol ||
            board[2] == player_symbol && board[5] == player_symbol &&
                board[8] == player_symbol ||
            board[0] == player_symbol && board[4] == player_symbol &&
                board[8] == player_symbol ||
            board[2] == player_symbol && board[4] == player_symbol &&
                board[6] == player_symbol) {
            return (ObsReward{.obs = board, .reward = 10000000, .done = true});
        }

        // Check for cats game
        if (board[0] != 0 && board[1] != 0 && board[2] != 0 && board[3] != 0 &&
            board[4] != 0 && board[5] != 0 && board[6] != 0 && board[7] != 0 &&
            board[8] != 0) {
            return (ObsReward{.obs = board, .reward = 0, .done = true});
        }

        // Make cpu move (TODO make this not random)
        size_t cpu_move = rand() % 9;
        size_t times = 0;
        while (board[cpu_move] != 0) {
            cpu_move = rand() % 9;
        }

        board[cpu_move] = cpu_symbol;
        // Check if cpu wins
        if (board[0] == cpu_symbol && board[1] == cpu_symbol &&
                board[2] == cpu_symbol ||
            board[3] == cpu_symbol && board[4] == cpu_symbol &&
                board[5] == cpu_symbol ||
            board[6] == cpu_symbol && board[7] == cpu_symbol &&
                board[8] == cpu_symbol ||
            board[0] == cpu_symbol && board[3] == cpu_symbol &&
                board[6] == cpu_symbol ||
            board[1] == cpu_symbol && board[4] == cpu_symbol &&
                board[7] == cpu_symbol ||
            board[2] == cpu_symbol && board[5] == cpu_symbol &&
                board[8] == cpu_symbol ||
            board[0] == cpu_symbol && board[4] == cpu_symbol &&
                board[8] == cpu_symbol ||
            board[2] == cpu_symbol && board[4] == cpu_symbol &&
                board[6] == cpu_symbol) {
            return (ObsReward{.obs = board, .reward = -10000, .done = true});
        }

        // Check for cats game
        if (board[0] != 0 && board[1] != 0 && board[2] != 0 && board[3] != 0 &&
            board[4] != 0 && board[5] != 0 && board[6] != 0 && board[7] != 0 &&
            board[8] != 0) {
            return (ObsReward{.obs = board, .reward = 0, .done = true});
        }

        return (ObsReward{.obs = board, .reward = 100000, .done = false});
    }

    virtual ObsReward reset() {
        bool cpu_first = rand() / (double)RAND_MAX > 0.5;
        board.clear();
        board.resize(9, 0);

        if (cpu_first) {
            cpu_symbol = x_val;
            player_symbol = o_val;

            size_t move = rand() % 9;
            board[move] = cpu_symbol;
        } else {
            cpu_symbol = o_val;
            player_symbol = x_val;
        }

        return (ObsReward){
            .obs = board,
            .reward = 0,
            .done = false,
        };
    }

    virtual void print() {
        char cpu_char = cpu_symbol == x_val ? 'X' : 'O';
        char player_char = player_symbol == x_val ? 'X' : 'O';
        printf(
            "%c %c %c\n%c %c %c\n%c %c %c\ncpu symbol: %c\nplayer symbol: %c\n",
            board[0] == cpu_symbol      ? cpu_char
            : board[0] == player_symbol ? player_char
                                        : '-',
            board[1] == cpu_symbol      ? cpu_char
            : board[1] == player_symbol ? player_char
                                        : '-',
            board[2] == cpu_symbol      ? cpu_char
            : board[2] == player_symbol ? player_char
                                        : '-',
            board[3] == cpu_symbol      ? cpu_char
            : board[3] == player_symbol ? player_char
                                        : '-',
            board[4] == cpu_symbol      ? cpu_char
            : board[4] == player_symbol ? player_char
                                        : '-',
            board[5] == cpu_symbol      ? cpu_char
            : board[5] == player_symbol ? player_char
                                        : '-',
            board[6] == cpu_symbol      ? cpu_char
            : board[6] == player_symbol ? player_char
                                        : '-',
            board[7] == cpu_symbol      ? cpu_char
            : board[7] == player_symbol ? player_char
                                        : '-',
            board[8] == cpu_symbol      ? cpu_char
            : board[8] == player_symbol ? player_char
                                        : '-',
            cpu_char, player_char);
    }

    vector<double> board;
    double player_symbol;
    double cpu_symbol;

    constexpr static double x_val = 1;
    constexpr static double o_val = -1;

    // App::num_actions
};

class TightRope : public App {
  public:
    TightRope() : App(1, {-10}, {10}, 2) { reset(); }

    virtual ObsReward step(size_t action) {
        int prev_delta = abs(pos);
        // 0 == left
        // 1 == right
        if (action == 0 && pos != -10) {
            pos -= 1;
        } else if (action == 1 && pos != 10) {
            pos += 1;
        }

        if (pos == 0) {
            return (
                ObsReward{.obs = {(double)pos}, .reward = 1.0, .done = true});
        }

        int new_delta = abs(pos);

        if (new_delta >= prev_delta) {
            // We moved in the wrong direction or didn't move (we're against the
            // wall)
            return (
                ObsReward{.obs = {(double)pos}, .reward = -1, .done = false});
        } else {
            // We moved in the correct direction
            return (
                ObsReward{.obs = {(double)pos}, .reward = 1, .done = false});
        }
    }

    virtual ObsReward reset() {
        pos = (rand() % 21) - 10;

        return (ObsReward{.obs = {(double)pos}, .reward = 0, .done = false});
    }

    virtual void print() {
        char buf[22] = {0};
        memset(buf, '-', sizeof(buf));

        if (pos == 10) {
            buf[pos] = '!';
        } else {
            buf[10] = '*';
            buf[pos + 10] = 'X';
        }

        puts(buf);
    }

    int pos = 0;
};

class Box : public App {
  public:
    Box() : App(2, {-20, -20}, {20, 20}, 4) { reset(); }

    virtual ObsReward step(size_t action) {
        double prev_delta = delta();
        // 0 == Up
        // 1 == Down
        // 2 == Left
        // 3 == Right
        if (action == 0 && y_pos != -20) {
            y_pos -= 1;
        } else if (action == 1 && y_pos != 20) {
            y_pos += 1;
        } else if (action == 2 && x_pos != -20) {
            x_pos -= 1;
        } else if (action == 3 && x_pos != 20) {
            x_pos += 1;
        }

        if (x_pos == 0 && y_pos == 0) {
            return (ObsReward{.obs = {(double)x_pos, (double)y_pos},
                              .reward = 1.0,
                              .done = true});
        }

        double new_delta = delta();

        if (new_delta >= prev_delta) {
            // We moved in the wrong direction or didn't move (we're against a
            // wall)
            return (ObsReward{.obs = {(double)x_pos, (double)y_pos},
                              .reward = -1.0,
                              .done = false});
        } else {
            // We moved in the correct direction
            return (ObsReward{.obs = {(double)x_pos, (double)y_pos},
                              .reward = 1.0,
                              .done = false});
        }
    }

    virtual ObsReward reset() {
        while (x_pos == 0 || y_pos == 0) {
            x_pos = (rand() % 41) - 20;
            y_pos = (rand() % 41) - 20;
        }

        return (ObsReward{
            .obs = {(double)x_pos, (double)y_pos}, .reward = 0, .done = false});
    }

    virtual void print() {
        char buf[41][42] = {0};
        memset(buf, '-', sizeof(buf));

        for (size_t i = 0; i < 40; i++) {
            buf[i][41] = '\n';
        }

        if (x_pos == 0 && y_pos == 0) {
            buf[20][20] = '!';
        } else {
            buf[20][20] = '*';
            buf[y_pos + 20][x_pos + 20] = 'X';
        }

        puts((char*)buf);
    }

    int x_pos = 0;
    int y_pos = 0;

  private:
    double delta() { return sqrt(pow(abs(x_pos), 2) + pow(abs(y_pos), 2)); }
};

class Agent {
  public:
    Agent(json network_json, App* _a, double _learning_rate,
          double _regularization_lambda, vector<Encoder*> encoders, MOA _m)
        : app(_a), learning_rate(_learning_rate),
          regularization_lambda(_regularization_lambda), encoders(encoders),
          m(_m) {
        n = new Network();
        n->from_json(network_json);

        weight_idx = n->get_edge_property("Weight")->index;
        num_outputs = n->num_outputs();

        p = nullptr;
        json proc_params = n->get_data("proc_params");
        string proc_name = n->get_data("other")["proc_name"];
        p = Processor::make(proc_name, proc_params);
        p->load_network(n);

        if (!n) {
            fprintf(stderr, "%s: main: Unable to load network.\n", __FILE__);
        }

        if (!p) {
            fprintf(stderr, "%s: main: Unable to create processor.\n",
                    __FILE__);
        }

        n->make_sorted_node_vector();

        app->reset();

        w.resize(app->num_actions, vector<double>(num_outputs + 1));
        for (size_t i = 0; i < w.size(); i++) {
            for (size_t j = 0; j < w[i].size(); j++) {
                w[i][j] = m.Random_Normal(0, 1);
            }
        }
    }

    Agent(json network_json, App* a, double learning_rate,
          double regularization_lambda, vector<Encoder*> encoders)
        : Agent(network_json, a, learning_rate, regularization_lambda, encoders,
                {}) {}

    ~Agent() {
        for (Encoder* e : encoders) {
            delete e;
        }

        delete n;
        delete p;

        delete app;
    }

    vector<double> activations(vector<double> observations) {
        p->clear_activity();

        // Encode all observations
        for (size_t ob = 0; ob < observations.size(); ob++) {
            vector<Spike> spikes = encoders[ob]->encode(observations[ob]);

            for (Spike s : spikes) {
                p->apply_spikes(spikes, false);
            }
        }

        p->run(sim_time);

        const vector<int> firing_counts = p->output_counts();
        vector<double> normalized(firing_counts.size() + 1);
        normalized[0] = 1;
        transform(firing_counts.begin(), firing_counts.end(),
                  normalized.begin() + 1,
                  [](int x) { return (double)x / 100; });

        return normalized;
    }

    double grade_reservoir(vector<size_t> tests_per_feature) {
        app->reset();

        vector<double> increment_width(app->num_observations);
        vector<double> counters(app->num_observations, 0);
        for (size_t i = 0; i < app->dmin.size(); i++) {
            increment_width[i] =
                (app->dmax[i] - app->dmin[i]) / tests_per_feature[i];
        }

        vector<vector<double>> outputs;

        bool done = false;
        size_t idx = app->num_observations - 1;
        while (!done) {
            // We have to move this into the agent first
            // Will use the `inputs` vector to be encoded
            // dmin + (increment_width[idx] * counters[idx])
            vector<double> inputs;
            for (size_t i = 0; i < app->num_observations; i++) {
                inputs.push_back(app->dmin[i] +
                                 (increment_width[i] * counters[i]));
            }

            vector<double> normalized_activations = activations(inputs);
            outputs.push_back(normalized_activations);

            while (true) {
                counters[idx]++;

                if (counters[idx] == tests_per_feature[idx]) {
                    if (idx == 0) {
                        done = true;
                        break;
                    }

                    counters[idx] = 0;
                    idx--;
                } else {
                    break;
                }
            }

            idx = app->num_observations - 1;
        }

        double min = DBL_MAX;
        for (size_t i = 0; i < outputs.size(); i++) {
            for (size_t j = i + 1; j < outputs.size(); j++) {
                // Calculate angle between vectors a and b
                double dot_product = 0;
                double norm_a = 0;
                double norm_b = 0;

                const vector<double>& a = outputs[i];
                const vector<double>& b = outputs[j];

                for (size_t k = 0; k < outputs[i].size(); k++) {
                    dot_product += a[k] * b[k];
                    norm_a += pow(a[k], 2);
                    norm_b += pow(b[k], 2);
                }

                norm_a = sqrt(norm_a);
                norm_b = sqrt(norm_b);

                double angle = acos(dot_product / (norm_a * norm_b));
                if (angle < min) {
                    min = angle;
                }
            }
        }

        return min;
    }

    void train(size_t epochs) {
        for (size_t i = 0; i < epochs; i++) {
            fprintf("Epoch: %zu, epsilon: %f ", i, epsilon);
        }
    }

    App* app;
    Network* n;
    Processor* p;
    vector<Encoder*> encoders;
    MOA m;
    double learning_rate;
    double regularization_lambda;
    size_t sim_time = 100;

    vector<vector<double>> w;

    int weight_idx;
    size_t num_outputs;

    const double discount_factor = 0.95;
    const double epislon_decay_factor = 0.999;
    double epsilon = 0.5;
};

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

int max_element(const vector<double>& x) {
    double max_elem = -1;

    for (size_t i = 0; i < x.size(); i++) {
        if (x[i] > max_elem) {
            max_elem = x[i];
        }
    }

    return max_elem;
}

vector<double> matrix_vector_multiply(vector<vector<double>> m,
                                      vector<double> v) {
    assert(m[0].size() == v.size());

    vector<double> result(v.size(), 0);
    for (size_t row = 0; row < m.size(); row++) {
        for (size_t col = 0; col < m[row].size(); col++) {
            result[row] += m[row][col] * v[col];
        }
    }

    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        fprintf(
            stderr,
            "usage: %s resevoir.json learning_rate lambda epochs num_bins\n",
            argv[0]);
        exit(1);
    }

    srand(time(nullptr));

    json network_json;
    vector<string> json_source = {argv[1]};

    ifstream fin(argv[1]);
    fin >> network_json;

    double learning_rate;
    sscanf(argv[2], "%lf", &learning_rate);

    double lambda;
    sscanf(argv[3], "%lf", &lambda);

    size_t total_epochs;
    sscanf(argv[4], "%zu", &total_epochs);

    size_t num_bins;
    sscanf(argv[5], "%zu", &num_bins);

    App* app = new Box();

    MOA m;
    m.Seed(m.Seed_From_Time(), "rand");

    const double discount_factor = 0.95;
    double epsilon = 0.5;
    const double epsilon_decay_factor = 0.999;

    vector<double> training_reward;

    // Create the actual encoders we're going to use
    vector<Encoder*> encoders;
    for (size_t i = 0; i < app->num_observations; i++) {
        double min = app->dmin[i];
        double max = app->dmax[i];

        encoders.emplace_back(
            new BinEncoder((i * num_bins), num_bins, min, max));
    }

    Agent a(network_json, app, learning_rate, lambda, std::move(encoders), m);

    const double largest_min_angle = a.grade_reservoir({10, 10});
    printf("Largest Minimum Angle: %f\n", largest_min_angle);

    // Training Loop
    for (size_t epochs = 0; epochs < total_epochs; epochs++) {
        printf("Epoch %zu, epsilon: %f:\n", epochs, epsilon);

        ObsReward o = app->reset();
        epsilon *= epsilon_decay_factor;
        bool done = false;
        size_t step = 0;
        double epoch_reward = 0;

        // app->print();

        while (!done) {
            // printf("\0331k\rStep: %zu", step++);
            size_t action = -1;
            vector<double> reservoir_activations = a.activations(o.obs);
            vector<double> model_prediction =
                matrix_vector_multiply(a.w, reservoir_activations);

            if (m.Random_Double() < epsilon) {
                // Perform random action
                action = m.Random_32() % app->num_actions;
            } else {
                // Get prediced action
                action = max_idx(model_prediction);
            }

            ObsReward new_o = app->step(action);
            step++;
            if (epochs == total_epochs - 1) {
                app->print();
            }
            done = new_o.done;
            epoch_reward += new_o.reward;

            vector<double> next_activations = a.activations(new_o.obs);
            vector<double> next_prediction =
                matrix_vector_multiply(a.w, next_activations);
            const double target =
                new_o.reward + discount_factor * max_element(next_prediction);

            vector<double> y = model_prediction;

            vector<double> y_hat = y;
            y[action] = target;

            vector<double> partial_gradient(y.size());

            for (size_t i = 0; i < y.size(); i++) {
                partial_gradient[i] = y_hat[i] - y[i];
            }

            // Perform weight updates
            for (size_t row = 0; row < a.w.size(); row++) {
                for (size_t col = 0; col < a.w[row].size(); col++) {
                    const double gradient =
                        partial_gradient[row] * reservoir_activations[col];

                    a.w[row][col] -=
                        (gradient * learning_rate) + (lambda * a.w[row][col]);
                }
            }

            o = new_o;
        }

        if (epochs == total_epochs - 1) {
            app->print();
        }

        training_reward.push_back(epoch_reward / (double)step);
    }

    // Testing loop
    // vector<double> testing_rewards;
    // for (size_t epochs = 0; epochs < 100; epochs++) {
    //     printf("Test%zu\n", epochs);

    //     ObsReward o = app->reset();
    //     bool done = false;
    //     size_t step = 0;
    //     double epoch_reward = 0;

    //     while (!done) {
    //         size_t action = -1;
    //         vector<double> reservoir_activations = a.activations(o.obs);
    //         vector<double> model_prediction =
    //             matrix_vector_multiply(a.w, reservoir_activations);

    //         // Get prediced action
    //         action = max_idx(model_prediction);

    //         ObsReward new_o = app->step(action);
    //         step++;
    //         if (epochs == total_epochs - 1) {
    //             app->print();
    //         }
    //         done = new_o.done;
    //         epoch_reward += new_o.reward;

    //         o = new_o;
    //     }

    //     if (epochs == total_epochs - 1) {
    //         app->print();
    //     }

    //     testing_rewards.push_back(epoch_reward / (double)step);
    // }

    // delete p;
    // delete n;

    // printf("Final weight matrix:\n");
    // for (size_t i = 0; i < a.w.size(); i++) {
    //     for (size_t j = 0; j < a.w[i].size(); j++) {
    //         printf("%6.3f ", a.w[i][j]);
    //     }
    //     printf("\n");
    // }

    // printf("Training rewards: [");
    // for (size_t i = 0; i < training_reward.size(); i++) {
    //     printf("%f", training_reward[i]);

    //     if (i != training_reward.size() - 1) {
    //         printf(", ");
    //     }
    // }
    // printf("]\n");

    // printf("Testing rewards: [");
    // for (size_t i = 0; i < testing_rewards.size(); i++) {
    //     printf("%f", testing_rewards[i]);

    //     if (i != testing_rewards.size() - 1) {
    //         printf(", ");
    //     }
    // }
    // printf("]\n");
}
