// chat_cli: `--request MSG [--model M] [--max-tokens N]` prints the request body; `--parse` reads an SSE stream on stdin
// and prints the assistant text. Exit 2 on a usage error.
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <string>
#include "baloon/chat_cli.hpp"

int main(int argc, char** argv) {
    std::string model = "baloon", request; bool parse = false, have_req = false; int max_tokens = 256;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--parse") parse = true;
        else if (a == "--request" && i + 1 < argc) { request = argv[++i]; have_req = true; }
        else if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--max-tokens" && i + 1 < argc) { max_tokens = std::atoi(argv[++i]); if (max_tokens <= 0) { std::fprintf(stderr, "--max-tokens must be > 0\n"); return 2; } }
        else { std::fprintf(stderr, "usage: chat_cli (--request MSG [--model M] [--max-tokens N] | --parse)\n"); return a == "--help" ? 0 : 2; }
    }
    if (parse == have_req) { std::fprintf(stderr, "exactly one of --request / --parse\n"); return 2; }
    if (have_req) {
        std::printf("%s\n", baloon::chat::build_request(model, {{"user", request}}, max_tokens).c_str());
        return 0;
    }
    const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    baloon::chat::stream(in, [](const std::string& d) { std::fwrite(d.data(), 1, d.size(), stdout); std::fflush(stdout); });
    std::printf("\n");
    return 0;
}
