extern "C" const char* __asan_default_options() {
    return "detect_leaks=1:halt_on_error=0";
}