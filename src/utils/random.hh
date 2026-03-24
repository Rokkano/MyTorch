#pragma once

#include "pattern.hh"

#include <random>

class UniformDistribution
{
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dis;

public:
    UniformDistribution(const std::default_random_engine &gen, const std::uniform_real_distribution<float> dis)
        : gen(gen), dis(dis)
    {
    }

    float operator()() { return this->dis(this->gen); }
};

class NormalDistribution
{
    std::default_random_engine gen;
    std::normal_distribution<float> dis;

public:
    NormalDistribution(const std::default_random_engine &gen, const std::normal_distribution<float> dis)
        : gen(gen), dis(dis)
    {
    }

    float operator()() { return this->dis(this->gen); }
};

class Random : public Singleton<Random>
{
    std::default_random_engine gen = std::default_random_engine();

public:
    UniformDistribution uniformDistribution(float min, float max)
    {
        return UniformDistribution(this->gen, std::uniform_real_distribution<float>(min, max));
    }

    NormalDistribution normalDistribution(float mean, float std)
    {
        return NormalDistribution(this->gen, std::normal_distribution<float>(mean, std));
    }
};