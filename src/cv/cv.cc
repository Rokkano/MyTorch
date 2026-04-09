#include "cv.hh"

void show(CvPlot::Axes &axes, OpenCVWindowOpts opts)
{
    axes.title(opts.title);
    axes.xLabel(opts.xlabel);
    axes.yLabel(opts.ylabel);

    axes.enableHorizontalGrid(opts.horizontalGrid);
    axes.enableVerticalGrid(opts.verticalGrid);

    CvPlot::show(OPENCV_WINDOW_ID, axes, opts.height, opts.width);
}
