#include "encoder.hpp"
#include <cstddef>
#include <framework.hpp>

using namespace std;

Encoder::Encoder(size_t _starting_neuron) : starting_neuron(_starting_neuron) {}
Encoder::~Encoder() {};

BalancedEncoder::BalancedEncoder(size_t starting_neuron, size_t _timesteps,
                                 double _dmin, double _dmax)
    : Encoder(starting_neuron), timesteps(_timesteps), dmin(_dmin),
      dmax(_dmax) {
    range = dmax - dmin;
}

size_t BalancedEncoder::num_neurons() { return 2; }
vector<neuro::Spike> BalancedEncoder::encode(double x) {
    vector<neuro::Spike> spikes;
    double frac = (x - dmin) / range;

    for (size_t i = 0; i < frac * timesteps; i++) {
        spikes.emplace_back(starting_neuron, i, 255);
    }

    for (size_t i = 0; i < (1 - frac) * timesteps; i++) {
        spikes.emplace_back(starting_neuron + 1, i, 255);
    }

    return std::move(spikes);
}

BinEncoder::BinEncoder(size_t starting_neuron, size_t _num_bins, double _dmin,
                       double _dmax)
    : Encoder(starting_neuron), num_bins(_num_bins), dmin(_dmin), dmax(_dmax) {
    range = dmax - dmin;
    bin_width = range / num_bins;
}

size_t BinEncoder::num_neurons() { return num_bins; }
vector<neuro::Spike> BinEncoder::encode(double x) {
    vector<neuro::Spike> spikes;

    const int bin = min(floor((x - dmin) / bin_width), (double)num_bins - 1);
    const int idx = starting_neuron + bin;

    spikes.emplace_back(idx, 0, 255);

    return std::move(spikes);
}
