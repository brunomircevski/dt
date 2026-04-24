#include <iostream>
#include <string>
#include "node.h"

using namespace std;

int main() {

    // Create a root node
    Node* root = new Node();
    root->attribute = "Outlook";

    // Create a leaf child
    root->children["Overcast"] = new Node();
    root->children["Overcast"]->isLeaf = true;
    root->children["Overcast"]->leafLabel = "Yes";

    cout << "Tree root attribute: " << root->attribute << endl;

    // Clean up the entire tree
    delete root;
    
    return 0;
}