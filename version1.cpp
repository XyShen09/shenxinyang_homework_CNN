#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <iostream>
using namespace std;

class Perceptron
{
    vector<double> m_vecWeights;
    double m_dBias;
    double m_dLearningRate;

public:
    Perceptron(int iInputSize, double dLearningRate) :/*m_vecWeights(iInputSize, 0.0),*/ m_dBias(0.0), m_dLearningRate(dLearningRate)
    {
        //nothing.
        std::random_device rd;
        std::mt19937 gen(rd());

        // Xavier 初始化的标准差：sqrt(2.0 / (输入规模 + 输出规模))
        // 对于单层感知机，输出规模通常为 1
        //若取固定值0.01就是"基础高斯分布初始化"
        double dStdDev = std::sqrt(2.0 / (iInputSize + 1)); 
              
        std::normal_distribution<double> dist(0.0, dStdDev);

        m_vecWeights.resize(iInputSize);
        for (double &rdW : m_vecWeights)
        {
            rdW = dist(gen);
        }
    }
public:
    double Sigmoid(double dZ) { return 1 / (1 + exp(-dZ)); }
    double Predict(vector<double> vecInputs)
    {
        return Sigmoid(inner_product(vecInputs.begin(), vecInputs.end(), m_vecWeights.begin(), m_dBias));
    }
    void Train(vector<double> vecInputs, double target)
    {
        double a = Predict(vecInputs);
        double y = target;

        double dz = a - y;

        for (auto i = 0u; i < m_vecWeights.size(); ++i)
        {
            double &w_i = m_vecWeights[i];
            double  x_i = vecInputs[i];

            w_i = w_i - m_dLearningRate * dz * x_i;
        }

        m_dBias = m_dBias - m_dLearningRate * dz;
    }
};