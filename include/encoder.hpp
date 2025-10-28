#pragma once

#include <cstddef>
#include <framework.hpp>

using namespace std;

class Encoder {
  public:
    Encoder(size_t _starting_neuron);
    virtual ~Encoder();

    virtual size_t num_neurons() = 0;
    virtual vector<neuro::Spike> encode(double x) = 0;

    size_t starting_neuron;
};

// Balances spikes between 2 neurons for values within the given range of [dmin,
// dmax]
// At the extremes `timesteps` will be produced for one neuron while the other
// will receive 0, if this is not desired extent dmin and dmax accordingly
class BalancedEncoder : public Encoder {
  public:
    BalancedEncoder(size_t starting_neuron, size_t _timesteps, double _dmin,
                    double _dmax);

    virtual size_t num_neurons();
    virtual vector<neuro::Spike> encode(double x);

    size_t timesteps;
    double dmin;
    double dmax;
    double range;
};

// Produces single spikes into bins
class BinEncoder : public Encoder {
  public:
    BinEncoder(size_t starting_neuron, size_t _num_bins, double _dmin,
               double _dmax);

    virtual size_t num_neurons();
    virtual vector<neuro::Spike> encode(double x);

    size_t num_bins;
    double bin_width;
    double dmin;
    double dmax;
    double range;
};
