#include "node.h"

Node::Node() : isLeaf(false), featureIndex(0), threshold(0.0) {}

std::unique_ptr<Node> Node::createLeaf(
    const std::string& label,
    std::size_t sampleCount
) {
    std::unique_ptr<Node> node = std::make_unique<Node>();
    node->isLeaf = true;
    node->leafLabel = label;
    node->sampleCount = sampleCount;
    return node;
}

std::unique_ptr<Node> Node::createDecision(
    const std::string& featureName,
    std::size_t featureIndex,
    double threshold,
    std::size_t sampleCount
) {
    std::unique_ptr<Node> node = std::make_unique<Node>();
    node->featureName = featureName;
    node->featureIndex = featureIndex;
    node->threshold = threshold;
    node->sampleCount = sampleCount;
    return node;
}
