#include "c45_tree.h"

void C45Tree::pruneWithTrainingAccuracy(
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

    // Bottom-up pruning:
    // simplify children first, then decide whether the whole subtree should
    // become one majority-class leaf.
    pruneWithTrainingAccuracy(node->leftChild, partitions.leftRows);
    pruneWithTrainingAccuracy(node->rightChild, partitions.rightRows);

    const std::size_t subtreeCorrect = countCorrectPredictions(node.get(), rowIndices);
    const std::string majorityLabel = getMajorityLabel(rowIndices);

    std::size_t leafCorrect = 0;
    for (std::size_t rowIndex : rowIndices)
    {
        if (dataset_->samples[rowIndex].label == majorityLabel)
        {
            ++leafCorrect;
        }
    }

    // This pruning rule is intentionally simple:
    // if one majority leaf is at least as accurate on the training rows that
    // reached this node, we keep the simpler model.
    if (leafCorrect >= subtreeCorrect)
    {
        node = Node::createLeaf(majorityLabel, rowIndices.size());
    }
}
