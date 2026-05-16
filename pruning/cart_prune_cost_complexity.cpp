#include "c45_tree.h"



void C45Tree::pruneWithCostComplexity(
    std::unique_ptr<Node>& node,
    const std::vector<std::size_t>& rowIndices
)
{
    if (!node || node->isLeaf)
    {
        return;
    }

    const PartitionedRows partitions =
        partitionRows(rowIndices, node->featureIndex, node->threshold);

    pruneWithCostComplexity(node->leftChild, partitions.leftRows);
    pruneWithCostComplexity(node->rightChild, partitions.rightRows);

    // After children are pruned, check if this node should also be pruned.
    if (node->isLeaf)
    {
        return;
    }

    const std::string majorityLabel = getMajorityLabel(rowIndices);
    std::size_t errorsAsLeaf = 0;
    for (std::size_t rowIndex : rowIndices)
    {
        if (dataset_->samples[rowIndex].label != majorityLabel)
        {
            ++errorsAsLeaf;
        }
    }

    const std::size_t errorsInSubtree = rowIndices.size() - countCorrectPredictions(node.get(), rowIndices);
    const std::size_t leavesInSubtree = countLeafNodes(node.get());

    // Cost(T) = MisclassificationError(T) + alpha * |Leaves(T)|
    const double costAsLeaf = static_cast<double>(errorsAsLeaf) + options_.ccpAlpha;
    const double costAsSubtree = static_cast<double>(errorsInSubtree) + options_.ccpAlpha * static_cast<double>(leavesInSubtree);

    if (costAsLeaf <= costAsSubtree + options_.epsilon)
    {
        node = Node::createLeaf(majorityLabel, rowIndices.size());
    }
}
