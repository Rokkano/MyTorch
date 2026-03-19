#include "test.hh"

#include <map>
#include <format>

int main() {
    std::map<std::string, bool> results;
    for (auto& f : REGISTRY) {
        std::cout << f.first << "...";
        bool result = f.second();
        results[f.first] = result;
        std::cout << "\r" << f.first << "..." << (result ? "✅" : "❌") << std::endl;
    }

    int passed = 0;
    int failed = 0;
    for (auto &kv : results)
    {
        passed += kv.second;
        failed += !kv.second;
    }

    std::cout << std::format("==== {} PASSED ==== {} FAILED ====", passed, failed) << std::endl;
}