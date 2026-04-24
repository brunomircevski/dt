#pragma once
#include <string>
#include <map>

using namespace std;

class Node {
public:
    string attribute;
    bool isLeaf;
    string leafLabel;

    // Numerical/Categorical logic
    bool isNumerical;
    double threshold;

    // Categorical: {"Red": Node*, "Blue": Node*}
    // Numerical:   {"<=": Node*, ">": Node*}
    map<string, Node*> children;

    Node();
    ~Node();
};