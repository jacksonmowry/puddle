#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    char buf[4096] = {0};

    double mins[128];
    double maxes[128];
    size_t features = 0;

    for (size_t i = 0; i < 128; i++) {
        mins[i] = DBL_MAX;
        maxes[i] = DBL_MIN;
    }

    bool first = true;
    while (fgets(buf, sizeof(buf) - 1, stdin)) {
        size_t feature_idx = 0;
        char* token = strtok(buf, " ");
        do {
            if (first) {
                features++;
            }

            double val;
            sscanf(token, "%lf", &val);

            if (val < mins[feature_idx]) {
                mins[feature_idx] = val;
            }
            if (val > maxes[feature_idx]) {
                maxes[feature_idx] = val;
            }

            feature_idx++;
        } while ((token = strtok(NULL, " ")));

        first = false;
    }

    printf("Min: [");
    for (size_t i = 0; i < features; i++) {
        printf("%f", mins[i]);

        if (i != features-1) {
            printf(",");
        }
    }
    printf("]\n");

    printf("Max: [");
    for (size_t i = 0; i < features; i++) {
        printf("%f", maxes[i]);

        if (i != features-1) {
            printf(",");
        }
    }
    printf("]\n");

    printf("Num: %zu\n", features);
}
