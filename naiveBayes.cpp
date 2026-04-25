#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

struct Sample {
    vector<string> attributes;
    string label;
};

struct NaiveBayesModel {
    vector<string> classLabels;
    map<string, int> classCounts;
    vector<set<string>> attributeDomains;
    vector<map<string, map<string, int>>> valueCountsByAttribute;
};

vector<string> splitTokens(const string& line) {
    vector<string> tokens;
    string token;
    stringstream ss(line);
    while (ss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

vector<Sample> readSamples(const string& filePath) {
    ifstream file(filePath);
    vector<Sample> samples;
    string line;

    if (!file.is_open()) {
        cerr << "Cannot open file: " << filePath << "\n";
        return samples;
    }

    while (getline(file, line)) {
        vector<string> tokens = splitTokens(line);
        if (tokens.size() < 2) {
            continue;
        }

        Sample s;
        s.attributes.assign(tokens.begin(), tokens.end() - 1);
        s.label = tokens.back();
        samples.push_back(s);
    }

    return samples;
}

NaiveBayesModel trainModel(const vector<Sample>& trainingSamples) {
    NaiveBayesModel model;
    if (trainingSamples.empty()) {
        return model;
    }

    const int attributeCount = static_cast<int>(trainingSamples[0].attributes.size());
    model.attributeDomains.assign(attributeCount, {});
    model.valueCountsByAttribute.assign(attributeCount, {});

    set<string> classSet;
    for (const Sample& sample : trainingSamples) {
        classSet.insert(sample.label);
        model.classCounts[sample.label]++;

        for (int attr = 0; attr < attributeCount; ++attr) {
            const string& value = sample.attributes[attr];
            model.attributeDomains[attr].insert(value);
            model.valueCountsByAttribute[attr][sample.label][value]++;
        }
    }

    model.classLabels.assign(classSet.begin(), classSet.end());
    return model;
}

double conditionalProbability(
    const NaiveBayesModel& model,
    int attributeIndex,
    const string& classLabel,
    const string& value,
    bool useSmoothing,
    double& beforeSmoothing
) {
    const int classCount = model.classCounts.at(classLabel);
    const int distinctValues = static_cast<int>(model.attributeDomains[attributeIndex].size());
    int count = 0;

    auto classIt = model.valueCountsByAttribute[attributeIndex].find(classLabel);
    if (classIt != model.valueCountsByAttribute[attributeIndex].end()) {
        auto valueIt = classIt->second.find(value);
        if (valueIt != classIt->second.end()) {
            count = valueIt->second;
        }
    }

    beforeSmoothing = static_cast<double>(count) / static_cast<double>(classCount);
    if (!useSmoothing) {
        return beforeSmoothing;
    }

    return static_cast<double>(count + 1) / static_cast<double>(classCount + distinctValues);
}

string predictClass(
    const NaiveBayesModel& model,
    const vector<string>& attributes,
    bool printSmoothingDetails,
    bool& smoothingUsedInThisPrediction
) {
    const int totalSamples = [&model]() {
        int total = 0;
        for (const auto& [label, count] : model.classCounts) {
            total += count;
        }
        return total;
    }();

    double bestScore = -1.0;
    string bestClass;

    for (const string& classLabel : model.classLabels) {
        const double prior = static_cast<double>(model.classCounts.at(classLabel)) / static_cast<double>(totalSamples);
        double score = prior;

        for (int attr = 0; attr < static_cast<int>(attributes.size()); ++attr) {
            double before = 0.0;
            const double withoutSmoothing = conditionalProbability(model, attr, classLabel, attributes[attr], false, before);
            bool needsSmoothing = (withoutSmoothing == 0.0);

            const double usedProbability =
                conditionalProbability(model, attr, classLabel, attributes[attr], needsSmoothing, before);

            if (needsSmoothing) {
                smoothingUsedInThisPrediction = true;
                if (printSmoothingDetails) {
                    cout << fixed << setprecision(8)
                         << "Smoothing needed -> class: " << classLabel
                         << ", attribute #" << (attr + 1)
                         << ", value: " << attributes[attr]
                         << ", before: " << before
                         << ", after: " << usedProbability << "\n";
                }
            }

            score *= usedProbability;
        }

        if (score > bestScore) {
            bestScore = score;
            bestClass = classLabel;
        }
    }

    return bestClass;
}

int classIndex(const vector<string>& classes, const string& label) {
    for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
        if (classes[i] == label) {
            return i;
        }
    }
    return -1;
}

void printConfusionMatrix(
    const vector<vector<int>>& matrix,
    const vector<string>& labels
) {
    cout << "\nConfusion matrix (rows = actual, columns = predicted):\n";
    cout << setw(18) << " ";
    for (const string& label : labels) {
        cout << setw(18) << label;
    }
    cout << "\n";

    for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
        cout << setw(18) << labels[i];
        for (int j = 0; j < static_cast<int>(labels.size()); ++j) {
            cout << setw(18) << matrix[i][j];
        }
        cout << "\n";
    }
}

void smoothingFallbackDemo(const NaiveBayesModel& model) {
    if (model.classLabels.empty() || model.attributeDomains.empty()) {
        return;
    }

    const string demoClass = model.classLabels[0];
    const int attributeIndex = 0;
    const string unseenValue = "__demo_unseen__";
    double before = 0.0;

    const double after = conditionalProbability(
        model, attributeIndex, demoClass, unseenValue, true, before
    );

    cout << fixed << setprecision(8)
         << "\nNo natural smoothing case occurred in test predictions.\n"
         << "Fallback smoothing demo (attribute #1): class: " << demoClass
         << ", value: " << unseenValue
         << ", before: " << before
         << ", after: " << after << "\n";
}

int main() {
    const string trainingPath = "../iris_training.txt";
    const string testPath = "../iris_test.txt";

    const vector<Sample> trainingSamples = readSamples(trainingPath);
    const vector<Sample> testSamples = readSamples(testPath);

    if (trainingSamples.empty() || testSamples.empty()) {
        cerr << "Training or test dataset is empty. Check file paths.\n";
        return 1;
    }

    const NaiveBayesModel model = trainModel(trainingSamples);

    int correct = 0;
    bool smoothingUsedAnywhere = false;
    vector<vector<int>> confusion(
        model.classLabels.size(),
        vector<int>(model.classLabels.size(), 0)
    );

    for (const Sample& sample : testSamples) {
        bool smoothingInPrediction = false;
        const string predicted = predictClass(model, sample.attributes, true, smoothingInPrediction);
        const int actualIdx = classIndex(model.classLabels, sample.label);
        const int predIdx = classIndex(model.classLabels, predicted);

        if (actualIdx >= 0 && predIdx >= 0) {
            confusion[actualIdx][predIdx]++;
        }

        if (predicted == sample.label) {
            ++correct;
        }
        if (smoothingInPrediction) {
            smoothingUsedAnywhere = true;
        }
    }

    const double accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(testSamples.size());
    cout << fixed << setprecision(2);
    cout << "Accuracy: " << accuracy << "% (" << correct << "/" << testSamples.size() << ")\n";
    printConfusionMatrix(confusion, model.classLabels);

    if (!smoothingUsedAnywhere) {
        smoothingFallbackDemo(model);
    }

    const int expectedAttributeCount = static_cast<int>(model.attributeDomains.size());

    cout << "\nManual classification mode (enter " << expectedAttributeCount << " attribute values).\n";
    cout << "Type 'q' to stop.\n";

    while (true) {
        cout << "\nEnter " << expectedAttributeCount << " values separated by spaces: ";
        string line;
        getline(cin >> ws, line);
        if (line == "q" || line == "Q") {
            break;
        }

        vector<string> attrs = splitTokens(line);
        if (static_cast<int>(attrs.size()) != expectedAttributeCount) {
            cout << "Invalid input. Please enter exactly " << expectedAttributeCount << " values.\n";
            continue;
        }

        bool smoothingInPrediction = false;
        const string predicted = predictClass(model, attrs, true, smoothingInPrediction);
        cout << "Predicted class: " << predicted << "\n";
    }

    return 0;
}