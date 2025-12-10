#!/bin/bash

source shared.sh

all_passed=1

# function run_single_test {
#     echo "Running test $1..."
#     rerun_local_iokerneld
#     if [[ $1 == *"tcp"* ]]; then
#     	rerun_mem_server
#     fi
#     if run_program ./bin/$1 2>/dev/null | grep -q "Passed"; then
#         say_passed
#     else
#         say_failed
#     	all_passed=0
#     fi
# }

function run_single_test {
  echo "Running test $1..."
  rerun_local_iokerneld
  [[ $1 == *"tcp"* ]] && rerun_mem_server

  # Show stdout+stderr live (left) AND still grep for "Passed" (right).
  # stdbuf makes line-buffered so you see lines immediately.
  if stdbuf -oL -eL run_program ./bin/$1 2>&1 \
      | tee >(cat >&2) | grep -q "Passed"; then
    say_passed
  else
    say_failed
    all_passed=0
  fi
}

function run_all_tests {
    TESTS=`ls bin | grep test_`
    for test in $TESTS
    do
        run_single_test $test
    done
}

function cleanup {
    kill_local_iokerneld
    kill_mem_server
}

run_all_tests
cleanup

if [[ $all_passed -eq 1 ]]; then
    exit 0
else
    exit -1
fi
