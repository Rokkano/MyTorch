#include <chrono>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <format>
#include <iostream>
#include <string>

class Progress
{

public:
    std::size_t length = 50;

    std::string barLeft = " ";
    std::string barOpen = "[";
    std::string barClose = "]";
    std::string barFill = "█";
    // std::string barOpen = "<";
    // std::string barClose = ">";
    // std::string barFill = "=";

protected:
    double currentProgress = 0;
    double maxProgress;

    double progressToChar = 1;

public:
    Progress(double max = 100)
    {
        this->maxProgress = max;
        this->progressToChar = length / max;
    };

    virtual void update(double progress = 1)
    {
        this->currentProgress += progress;
        printProgress();
    }

    virtual void printProgress()
    {
        std::size_t currentProgressChar = std::round(this->currentProgress * this->progressToChar);
        std::cout << "\r" << barOpen;

        // BAR
        for (std::size_t i = 0; i < currentProgressChar; i++)
            std::cout << barFill;
        for (std::size_t i = currentProgressChar; i < length; i++)
            std::cout << barLeft;

        // PERCENTAGE
        std::cout << barClose << " " << std::floor(this->currentProgress * 100 / this->maxProgress) << "%";

        // N° ITERATIONS
        std::cout << " (" << this->currentProgress << "/" << this->maxProgress << ")";
        std::cout << std::flush;
    }
};

class ETAProgress : public Progress
{
    typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
    typedef std::chrono::duration<double> Duration;

public:
    TimePoint zeroRef;
    TimePoint firstRef;
    TimePoint lastTimePoint;

    // https://www.geeksforgeeks.org/web-tech/expression-for-mean-and-variance-in-a-running-stream/
    Duration iterationEstimation; // streaming mean

    ETAProgress(double max = 100) : Progress(max) {}

    void update(double progress = 1) override
    {
        ETAProgress::TimePoint now = std::chrono::system_clock::now();
        if (this->lastTimePoint != zeroRef)
            this->iterationEstimation +=
                ((now - this->lastTimePoint) - this->iterationEstimation) / this->currentProgress;
        else
            this->firstRef = std::chrono::system_clock::now();
        this->lastTimePoint = now;
        Progress::update(progress);
    }

    void printProgress() override
    {
        Progress::printProgress();

        auto format = [](double count, bool subCount = false)
        {
            if (count > 3600)
                return std::format("{}h{}m", std::floor(count / 3600), std::floor((int)std::floor(count) % 3600 / 60));
            if (count > 60)
                return std::format("{}m{}s", std::floor(count / 60), (int)std::floor(count) % 60);
            if (count > 1)
                return std::format("{}s", std::floor(count * 1000) / 1000);
            if (count > 0.001)
                return std::format("{}ms", std::floor(count * 1000));
            else
                return subCount ? std::format("<1ms") : std::format("0ms");
        };

        std::cout << "  |  " << format(this->iterationEstimation.count(), true) << "/it";
        std::cout << "  |  ETA "
                  << format(((this->iterationEstimation) * (maxProgress - currentProgress)).count(), false);

        std::cout << "\033[K" << std::flush;
        if (this->currentProgress >= this->maxProgress)
        {
            Duration elapsedTime = this->lastTimePoint - this->firstRef;
            std::cout << "  |  Completed in " << format((elapsedTime).count(), false) << "\n" << std::flush;
        }
    }
};
