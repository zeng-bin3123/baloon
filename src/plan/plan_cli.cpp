// plan_cli.cpp - command line: plan_cli <context_tokens> <pool_bytes> [key=value ...] (see baloon/plan_cli.hpp).
#include <string>
#include <vector>
#include "baloon/plan_cli.hpp"

#if !BALOON_PLAN_MAIN_DISABLED
int main(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    return baloon::plan::cli::run(args, stdout, stderr);
}
#endif
