#include <cstddef>
#include <string>
#include <iostream>
#include <cmath>
#include <chrono>
#include <ctime>

class Progress {

public:
    std::size_t length = 50;
        
    std::string barOpen = "[";
    std::string barClose = "]";
    std::string barFill = "█";
    std::string barLeft = " ";


protected:
    double currentProgress = 0; // /max
    double progressToChar = 1;
    double maxProgress;

    std::size_t lastPrintedProgress = 0; // /max


public:
    Progress(double max = 100) 
    {
        this->maxProgress = max;
        this->progressToChar = length / max;
    };

    virtual void update(double progress = 1) 
    {
        this->currentProgress += progress;
        std::size_t printedProgress = std::floor(this->currentProgress * this->progressToChar);
        if (this->lastPrintedProgress < printedProgress)
        {
            this->lastPrintedProgress = printedProgress;
            printProgress();
        }
    }

    virtual void printProgress() 
    {
        std::cout << "\r" << barOpen;
        for(std::size_t i = 0; i < this->lastPrintedProgress; i++)
            std::cout << barFill;
        for(std::size_t i = this->lastPrintedProgress; i < length; i++)
            std::cout << barLeft;
        std::cout << barClose << std::floor(this->currentProgress * 100 / this->maxProgress) << "%";
        std::cout << std::flush;
    }
};


class ETAProgress : public Progress
{
    typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
    typedef std::chrono::duration<double> Duration;

public:

    TimePoint zeroRef;
    TimePoint lastTimePoint;
    Duration iterationEstimation; // streaming mean

    ETAProgress(double max = 100) : Progress(max) {}

    void update(double progress = 1) override
    {
        ETAProgress::TimePoint now = std::chrono::system_clock::now();
        if (this->lastTimePoint != zeroRef && this->lastTimePoint != now)
            this->iterationEstimation += ((now - this->lastTimePoint) - this->iterationEstimation) / currentProgress;
        this->lastTimePoint = now;
        Progress::update(progress);
    }

    void printProgress() override
    {
        Progress::printProgress();
        std::cout << "  | " << std::chrono::round<std::chrono::seconds>(this->iterationEstimation) << "/it";
        std::cout << "  | ETA " << std::chrono::round<std::chrono::seconds>(this->iterationEstimation) * (maxProgress - currentProgress);
        std::cout << "\033[K" << std::flush;
        if (this->currentProgress >= this->maxProgress)
            std::cout << "\n" << std::flush;
    }
};

