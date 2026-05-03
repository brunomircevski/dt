#pragma once

#include <memory>
#include <string>

class Node {
public:
    // This says whether the node is a final answer.
    // If isLeaf is true, the tree stops here and returns leafLabel.
    bool isLeaf;

    // Which feature this node looks at.
    // Example: "PetalLengthCm"
    // We store the feature index too, because using the index is simpler
    // during prediction than searching by name every time.
    std::string featureName;
    std::size_t featureIndex;

    // The numeric question asked by this node:
    // "Is sample.features[featureIndex] <= threshold?"
    double threshold;

    // If this node is a leaf, this is the predicted class.
    // Example: "Iris-setosa"
    std::string leafLabel;

    // Left child means the answer to the question was "yes":
    // feature value <= threshold
    std::unique_ptr<Node> leftChild;

    // Right child means the answer to the question was "no":
    // feature value > threshold
    std::unique_ptr<Node> rightChild;

    // Store sample count for printing tree
    std::size_t sampleCount = 0;

    // Constructor:
    // sets simple default values when we create a new node object.
    Node();

    // Static helper function:
    // creates a leaf node and returns ownership using std::unique_ptr.
    //
    // C++ note:
    // std::unique_ptr is a smart pointer. It automatically deletes the object
    // when nobody owns it anymore, so we do not need to call delete manually.
    static std::unique_ptr<Node> createLeaf(
        const std::string& label,
        std::size_t sampleCount
    );

    // Creates a decision node that asks one numeric question.
    static std::unique_ptr<Node> createDecision(
        const std::string& featureName,
        std::size_t featureIndex,
        double threshold,
        std::size_t sampleCount
    );
};
