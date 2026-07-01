// Lsan custom configuration to avoid false positive on OpenCV
// See : https://forums.wxwidgets.org/viewtopic.php?t=50402
#ifdef __has_feature
#if __has_feature(address_sanitizer)
extern "C" const char* __lsan_default_suppressions() {
    return  "leak:libfontconfig.so\n"
            "leak:libglib-2.0.so\n"
            "leak:FcFontRenderPrepare\n"
            "leak:FcLangSetCopy\n";
}
#endif
#endif