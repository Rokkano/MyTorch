#pragma once

#include <opencv2/opencv.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wcatch-value"

#define CVPLOT_HEADER_ONLY
#include <CvPlot/cvplot.h>

#pragma GCC diagnostic pop

struct OpenCVWindowOpts
{
    std::string title = "Graph";
    std::string xlabel = "x";
    std::string ylabel = "y";

    std::size_t width = 800;
    std::size_t height = 640;

    bool horizontalGrid = false;
    bool verticalGrid = false;

    OpenCVWindowOpts defaultOpts() { return {}; }
};

inline char OPENCV_WINDOW_ID[] = "main";

void show(CvPlot::Axes &axes, const OpenCVWindowOpts &opts);