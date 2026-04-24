#include "node.h"

// Constructor
Node::Node() : isLeaf(false), isNumerical(false), threshold(0.0) {}

// Destructor
Node::~Node() {
    for (auto const& [key, child] : children) {
        delete child;
    }
}