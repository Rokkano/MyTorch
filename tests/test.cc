#include "test.hh"

#include <argparse.hpp>
#include <format>
#include <map>

bool run_test(Record record)
{
    std::cout << "  " << record.func << "(" << record.args << ")...";
    bool result = record.runner();

    std::cout << "\x1b[2K";
    std::cout << "\r";
    std::cout << (result ? "✅ " : "❌ ") << record.func << "(" << record.args << ")";
    if (assertBuffer().tellp() != 0)
    {
        std::cout << " " << assertBuffer().str();
        assertBuffer().str("");
        assertBuffer().clear();
    }

    std::cout << std::endl;
    return result;
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("Test suite");

    program.add_argument("--functional").help("Run functional tests only.").flag();

    program.add_argument("--unit").help("Run unit tests only.").flag();

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::map<std::string, bool> results;
    if (program["--unit"] == true || program["--functional"] == false)
    {
        for (auto &f : registry())
            results[f.first] = run_test(f.second);
    }
    if (program["--functional"] == true || program["--unit"] == false)
    {
        for (auto &f : functionalRegistry())
            results[f.first] = run_test(f.second);
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