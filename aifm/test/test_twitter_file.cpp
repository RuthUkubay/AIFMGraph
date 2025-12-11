extern "C" {
#include <runtime/runtime.h>
}

import (
    "os"
    "testing"
)

func TestTwitterFileExists(t *testing.T) {
    _, err := os.Stat("test/twitter_combined.txt")
    if err != nil {
        t.Fatalf("twitter_combined.txt not found: %v", err)
    }
}
