#include "c45_tree.h"

void C45Tree::pruneWithReducedError(
    std::unique_ptr<Node>& node,
    const std::vector<std::size_t>& rowIndices
)
{
    // Reduced Error Pruning (REP) is a simple, effective bottom-up pruning method.
    // While formally it should use a separate validation set to be "Reduced Error",
    // in this lightweight implementation it acts as a baseline accuracy-driven
    // post-pruner.
    if (!node || node->isLeaf)
    {
        return;
    }

    const PartitionedRows partitions =
        partitionRows(rowIndices, node->featureIndex, node->threshold);

    pruneWithReducedError(node->leftChild, partitions.leftRows);
    pruneWithReducedError(node->rightChild, partitions.rightRows);

    // After children are potentially simplified, check the current node.
    if (node->isLeaf)
    {
        return;
    }

    const std::string majorityLabel = getMajorityLabel(rowIndices);
    std::size_t leafCorrect = 0;
    for (std::size_t rowIndex : rowIndices)
    {
        if (dataset_->samples[rowIndex].label == majorityLabel)
        {
            ++leafCorrect;
        }
    }

    const std::size_t subtreeCorrect = countCorrectPredictions(node.get(), rowIndices);

    // If the majority label leaf performs at least as well as the entire subtree,
    // we prune the subtree to simplify the model.
    if (leafCorrect >= subtreeCorrect)
    {
        node = Node::createLeaf(majorityLabel, rowIndices.size());
    }
}
