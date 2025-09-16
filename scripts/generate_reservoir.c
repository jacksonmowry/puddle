// Generates a resevoir in 2D space allowing us to consider distance between
// neurons Then the network representation is converted to a TENNLab network

#include <getopt.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define log_fatal(fmt, ...)                                                    \
    do {                                                                       \
        fprintf(stderr, __FILE__ ": " fmt __VA_OPT__(, ) __VA_ARGS__);         \
        exit(1);                                                               \
    } while (false)

typedef struct Neuron {
    double x;
    double y;
    size_t* connections;
    size_t num_connections;
    size_t cap_connections;
} Neuron;

const char* empty_network =
    "{\"Associated_Data\":{\"other\":{\"proc_name\":\"risp\"},\"proc_params\":{"
    "\"discrete\":true,\"leak_mode\":\"configurable\",\"spike_value_factor\":255,\"max_delay\":15,\"max_"
    "threshold\":1073741824,\"max_weight\":255,\"min_potential\":-1073741825,\"min_threshold\":"
    "1,\"min_weight\":-255}},\"Edges\":[],\"Inputs\":[],\"Network_Values\":[],"
    "\"Nodes\":[],\"Outputs\":[],\"Properties\":{\"edge_properties\":[{"
    "\"index\":1,\"max_value\":15,\"min_value\":1,\"name\":\"Delay\",\"size\":"
    "1,\"type\":73},{\"index\":0,\"max_value\":255,\"min_value\":-255,\"name\":"
    "\"Weight\",\"size\":1,\"type\":73}],\"network_properties\":[],\"node_"
    "properties\":[{\"index\":0,\"max_value\":1073741824,\"min_value\":1,\"name\":"
    "\"Threshold\",\"size\":1,\"type\":73},{\"index\":1,\"max_value\":1,\"min_value\":0,\"name\":"
    "\"Leak\",\"size\":1,\"type\":66}]}}";

int main(int argc, char* argv[]) {
    size_t resevoir_size = 100;
    double input_percent = 0.20;
    double output_percent = 0.20;
    double connection_chance = 0.50;
    size_t feature_neurons = 16;
    size_t class_neurons = 3;
    char* filename = nullptr;

    srand(time(NULL));

    int c;
    int digit_optind = 0;

    while (1) {
        auto this_option_optind = optind ? optind : 1;
        auto option_index = 0;
        static struct option long_options[] = {
            {"resevoir_size", required_argument, 0, 's'},
            {"input_percent", required_argument, 0, 'i'},
            {"output_percent", required_argument, 0, 'o'},
            {"connection_probability", required_argument, 0, 'p'},
            {"feature_neurons", required_argument, 0, 'f'},
            {"class_neurons", required_argument, 0, 'c'},
            {0, 0, 0, 0},
        };

        c = getopt_long(argc, argv, "s:i:o:p:f:c:", long_options,
                        &option_index);

        if (c == -1) {
            break;
        }

        switch (c) {
        case 0:
            printf("option %s", long_options[option_index].name);
            if (optarg) {
                printf(" with arg%s", optarg);
            }
            printf("\n");
            break;
        case 's':
            resevoir_size = strtoull(optarg, nullptr, 0);
            break;
        case 'i':
            input_percent = strtod(optarg, nullptr);
            if (input_percent <= 0.0 || input_percent > 1.00) {
                log_fatal("Input percent cannot be less than or equal to 0 or "
                          "greater than 1");
            }
            break;
        case 'o':
            output_percent = 1 - strtod(optarg, nullptr);
            if (output_percent <= 0.0 || output_percent > 1.00) {
                log_fatal("Output percent cannot be less than or equal to 0 or "
                          "greater than 1\n");
            }
            break;
        case 'p':
            connection_chance = strtod(optarg, nullptr);
            if (connection_chance <= 0.0 || connection_chance > 1.00) {
                log_fatal(
                    "Connection chance cannot be less than or equal to 0 or "
                    "greater than 1\n");
            }
            break;
        case 'f':
            feature_neurons = strtoull(optarg, nullptr, 0);
            break;
        case 'c':
            class_neurons = strtoull(optarg, nullptr, 0);
            break;
        case '?':
            break;
        default:
            printf("?? getopt returned character code 0%o ??\n", c);
        }
    }

    if (optind < argc) {
        const auto remaining_arguments = argc - optind - 1;
        if (remaining_arguments > 1) {
            log_fatal("Too many file arguments, expected 1 extra arguments\n");
        }

        filename = argv[optind];
    }

    Neuron* neurons = calloc(resevoir_size, sizeof(Neuron));
    size_t input_neurons = 0;
    size_t output_neurons = 0;

    for (size_t i = 0; i < resevoir_size; i++) {
        neurons[i] = (Neuron){.x = rand() / (double)RAND_MAX,
                              .y = rand() / (double)RAND_MAX,
                              .connections = calloc(20, sizeof(size_t)),
                              .num_connections = 0,
                              .cap_connections = 20};
        if (neurons[i].y < input_percent) {
            input_neurons++;
        }
        if (neurons[i].y > output_percent) {
            output_neurons++;
        }
    }

    for (size_t i = 0; i < resevoir_size; i++) {
        for (size_t j = 0; j < resevoir_size; j++) {
            // Calculate distance to the other neuron
            double distance = sqrt(pow(neurons[i].x - neurons[j].x, 2) +
                                   pow(neurons[i].y - neurons[j].y, 2));

            // We'll use the distance to determine how likely a connection is
            if ((double)rand() / RAND_MAX < connection_chance * distance) {
                if (neurons[i].num_connections >= neurons[i].cap_connections) {
                    // Realloc
                    neurons[i].connections = realloc(
                        neurons[i].connections,
                        (neurons[i].cap_connections * 2) * sizeof(size_t));
                    neurons[i].cap_connections *= 2;
                }

                neurons[i].connections[neurons[i].num_connections++] = j;
            }
        }
    }

    if (filename) {
        printf("FJ %s\n", filename);
    } else {
        printf("FJ\n");
        printf("%s\n", empty_network);
    }

    // Add network inputs
    printf("AN ");
    for (size_t i = 0; i < feature_neurons; i++) {
        printf("%zu ", i);
    }
    printf("\n");

    printf("AI ");
    for (size_t i = 0; i < feature_neurons; i++) {
        printf("%zu ", i);
    }
    printf("\n");

    // Build resevoir
    printf("AN ");
    for (size_t i = feature_neurons; i < feature_neurons + resevoir_size; i++) {
        printf("%zu ", i);
    }
    printf("\n");

    // Add network outputs
    printf("AN ");
    for (size_t i = feature_neurons + resevoir_size;
         i < feature_neurons + resevoir_size + class_neurons; i++) {
        printf("%zu ", i);
    }
    printf("\n");

    printf("AO ");
    for (size_t i = feature_neurons + resevoir_size;
         i < feature_neurons + resevoir_size + class_neurons; i++) {
        printf("%zu ", i);
    }
    printf("\n");

    printf("SNP_ALL Leak 1\n");

    for (size_t i = 0; i < feature_neurons + resevoir_size;
         i++) {
        printf("SNP %zu Threshold %d\n", i, rand() % 255 + 1);
    }
    for (size_t i = feature_neurons + resevoir_size; i < feature_neurons + resevoir_size + class_neurons;
         i++) {
        printf("SNP %zu Threshold %d\n", i, 1073741824);
        printf("SNP %zu Leak %d\n", i, 0);
    }

    // Connect feature neurons to input
    for (size_t i = 0; i < feature_neurons; i++) {
        for (size_t j = 0; j < resevoir_size; j++) {
            if (neurons[j].y >= input_percent) {
                continue;
            }

            printf("AE %zu %zu\n", i, feature_neurons + j);
            printf("SEP %zu %zu Weight %d\n", i, feature_neurons + j,
                   (((rand() % 2) * 2) - 1) * rand() % 256);
            printf("SEP %zu %zu Delay %d\n", i, feature_neurons + j,
                   rand() % 15 + 1);
        }
    }

    // Connect resevoir neurons to each other
    for (size_t i = 0; i < resevoir_size; i++) {
        for (size_t j = 0; j < neurons[i].num_connections; j++) {
            size_t pre = feature_neurons + i;
            size_t post = feature_neurons + neurons[i].connections[j];
            printf("AE %zu %zu\n", pre, post);
            printf("SEP %zu %zu Weight %d\n", pre, post,
                   (((rand() % 2) * 2) - 1) * rand() % 256);
            printf("SEP %zu %zu Delay %d\n", pre, post, rand() % 15 + 1);
        }
    }

    // Connect output resevoir neurons to class neurons
    for (size_t i = feature_neurons + resevoir_size;
         i < feature_neurons + resevoir_size + class_neurons; i++) {
        for (size_t j = 0; j < resevoir_size; j++) {
            if (neurons[j].y <= output_percent) {
                continue;
            }

            printf("AE %zu %zu\n", feature_neurons + j, i);
            printf("SEP %zu %zu Weight %d\n", feature_neurons + j, i,
                   (((rand() % 2) * 2) - 1) * rand() % 256);
            printf("SEP %zu %zu Delay %d\n", feature_neurons + j, i,
                   rand() % 15 + 1);
        }
    }

    printf("TJ\n");
}
