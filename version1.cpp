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

//int main(int argc, char **argv)
//{
//    Perceptron perceptron(2, 0.1);
//    // 训练数据 (AND 逻辑)
//    std::vector<std::pair<std::vector<double>, double>> data = {
//        {{0, 0}, 0.0}, {{0, 1}, 0.0}, {{1, 0}, 0.0}, {{1, 1}, 1}
//    };
//
//    // 迭代 1000 次
//    for (int epoch = 0; epoch < 2000; ++epoch) {
//        for (auto& sample : data) {
//            perceptron.Train(sample.first, sample.second);
//        }
//    }
//
//    // 测试结果
//    std::cout << "测试 AND 逻辑预测结果:" << std::endl;
//    for (auto& sample : data) {
//        std::cout << sample.first[0] << " AND " << sample.first[1]
//            << " = " << (perceptron.Predict(sample.first) > 0.5 ? 1 : 0)
//            << " (概率: " << perceptron.Predict(sample.first) << ")" << std::endl;
//    }
//
//    return 0;
//}

int main() {
    // 1. 初始化感知机：3个输入 (血压高压, 血压低压, BMI), 学习率 0.1
    Perceptron perceptron(3, 0.2);

    // 2. 准备健康评估数据 (已手动归一化)
    // 特征顺序：{高压/200, 低压/100, BMI/40} | 标签：1.0(健康), 0.0(预警)
    struct Sample { vector<double> x; double y; };
    vector<Sample> trainingSet = {
        {{115.0 / 200, 75.0 / 100, 21.0 / 40}, 1.0}, // 理想
        {{120.0 / 200, 80.0 / 100, 24.0 / 40}, 1.0}, // 正常
        {{145.0 / 200, 95.0 / 100, 32.0 / 40}, 0.0}, // 风险
        {{150.0 / 200, 105.0 / 100, 26.0 / 40}, 0.0},// 风险
        {{110.0 / 200, 70.0 / 100, 22.0 / 40}, 1.0}  // 优秀
    };

    // 3. 迭代训练：使用您最初的逐样本更新方式
    cout << "正在通过在线学习模式训练模型..." << endl;
    for (int epoch = 0; epoch < 1000; ++epoch) {
        for (const auto& sample : trainingSet) {
            perceptron.Train(sample.x, sample.y);
        }
    }

    // 4. 测试预测
    // 使用您第10天的数据：117/78，假设 BMI 为 23.7
    vector<double> myCurrentData = { 117.0 / 200, 78.0 / 100, 23.7 / 40 };
    double dProb = perceptron.Predict(myCurrentData);

    cout << "\n--- 评估结果 ---" << endl;
    cout << "输入指标：收缩压 117, 舒张压 78, BMI 23.7" << endl;
    cout << "健康评估概率: " << (dProb * 100) << "%" << endl;
    cout << "系统建议: " << (dProb > 0.5 ? "继续保持当前运动习惯" : "注意休息并监测血压") << endl;

    return 0;
}
```

#### 基于BCE损失函数和sigmoid的多层感知机（解决XOR问题）

```c++
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <iostream>
#include <algorithm>

using namespace std;

// 激活函数
double Sigmoid(double dZ) { return 1.0 / (1.0 + exp(-dZ)); }
// 隐藏层仍需使用的导数 (Sigmoid 的导数)
double SigmoidDeriv(double dA) { return dA * (1.0 - dA); }

class NeuralNetwork
{
    int m_iInputSize, m_iHiddenSize, m_iOutputSize;
    double m_dLearningRate;

    vector<vector<double>> m_vecW1; // 输入 -> 隐藏
    vector<vector<double>> m_vecW2; // 隐藏 -> 输出
    vector<double> m_vecB1;         // 隐藏层偏置
    vector<double> m_vecB2;         // 输出层偏置

public:
    NeuralNetwork(int iIn, int iHidden, int iOut, double dLR)
        : m_iInputSize(iIn), m_iHiddenSize(iHidden), m_iOutputSize(iOut), m_dLearningRate(dLR)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 0.5);

        m_vecW1.resize(m_iHiddenSize, vector<double>(m_iInputSize));
        m_vecB1.resize(m_iHiddenSize, 0.0);
        for (auto& row : m_vecW1) for (auto& w : row) w = dist(gen);

        m_vecW2.resize(m_iOutputSize, vector<double>(m_iHiddenSize));
        m_vecB2.resize(m_iOutputSize, 0.0);
        for (auto& row : m_vecW2) for (auto& w : row) w = dist(gen);
    }

    pair<vector<double>, vector<double>> Forward(const vector<double>& vecInputs)
    {
        // 隐藏层前向
        vector<double> h(m_iHiddenSize);
        for (int i = 0; i < m_iHiddenSize; ++i) {
            h[i] = Sigmoid(inner_product(vecInputs.begin(), vecInputs.end(), m_vecW1[i].begin(), m_vecB1[i]));
        }
        // 输出层前向
        vector<double> o(m_iOutputSize);
        for (int i = 0; i < m_iOutputSize; ++i) {
            o[i] = Sigmoid(inner_product(h.begin(), h.end(), m_vecW2[i].begin(), m_vecB2[i]));
        }
        return { h, o };
    }

    void Train(const vector<double>& vecInputs, const vector<double>& vecTargets)
    {
        auto res = Forward(vecInputs);
        auto h = res.first;
        auto o = res.second;

        // --- 1. 输出层梯度 (BCE + Sigmoid 的组合简化) ---
        // 这里的公式由 dL/do * do/dz 推导而来，结果简化为 (o - y)
        vector<double> vecDzOutput(m_iOutputSize);
        for (int i = 0; i < m_iOutputSize; ++i) {
            vecDzOutput[i] = o[i] - vecTargets[i];
        }

        // --- 2. 隐藏层梯度 (保持链式法则) ---
        vector<double> vecDzHidden(m_iHiddenSize);
        for (int j = 0; j < m_iHiddenSize; ++j) {
            double dErrorFromOutput = 0.0;
            for (int i = 0; i < m_iOutputSize; ++i) {
                dErrorFromOutput += m_vecW2[i][j] * vecDzOutput[i];
            }
            vecDzHidden[j] = dErrorFromOutput * SigmoidDeriv(h[j]);
        }

        // --- 3. 更新权重和偏置 ---
        for (int i = 0; i < m_iOutputSize; ++i) {
            for (int j = 0; j < m_iHiddenSize; ++j) {
                m_vecW2[i][j] -= m_dLearningRate * vecDzOutput[i] * h[j];
            }
            m_vecB2[i] -= m_dLearningRate * vecDzOutput[i];
        }

        for (int j = 0; j < m_iHiddenSize; ++j) {
            for (int k = 0; k < m_iInputSize; ++k) {
                m_vecW1[j][k] -= m_dLearningRate * vecDzHidden[j] * vecInputs[k];
            }
            m_vecB1[j] -= m_dLearningRate * vecDzHidden[j];
        }
    }

    double Predict(const vector<double>& vecInputs) { return Forward(vecInputs).second[0]; }
};

int main() {
    NeuralNetwork nn(2, 2, 1, 0.5);

    struct Sample { vector<double> x, y; };
    vector<Sample> data = { {{0,0},{0}}, {{0,1},{1}}, {{1,0},{1}}, {{1,1},{0}} };

    std::random_device rd;
    std::mt19937 g(rd());

    cout << "使用 BCE 损失函数训练 XOR 网络..." << endl;

    for (int epoch = 0; epoch < 10000; ++epoch) {
        shuffle(data.begin(), data.end(), g);
        double epochLoss = 0;

        for (const auto& s : data) {
            nn.Train(s.x, s.y);
            double o = nn.Predict(s.x);
            // 计算 BCE Loss: -[y*log(o) + (1-y)*log(1-o)]
            epochLoss += -(s.y[0] * log(o + 1e-10) + (1.0 - s.y[0]) * log(1.0 - o + 1e-10));
        }

        if (epoch % 1000 == 0) cout << "Epoch " << epoch << " | Avg BCE Loss: " << epochLoss / 4.0 << endl;
        if (epochLoss / 4.0 < 0.01) {
            cout << "收敛成功，第 " << epoch << " 轮退出。" << endl;
            break;
        }
    }

    cout << "\n--- 验证结果 ---" << endl;
    for (auto& s : vector<Sample>{ {{0,0},{0}}, {{0,1},{1}}, {{1,0},{1}}, {{1,1},{0}} }) {
        cout << s.x[0] << "," << s.x[1] << " -> " << (nn.Predict(s.x) > 0.5 ? 1 : 0)
            << " (概率: " << nn.Predict(s.x) << ")" << endl;
    }

    return 0;
}



#include <iostream>
#include <vector>

void bubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break; // 提前优化：已排序则退出
    }
}

int main() {
    std::vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
    bubbleSort(arr);
    for (int v : arr) std::cout << v << " ";
    // 输出: 11 12 22 25 34 64 90
}