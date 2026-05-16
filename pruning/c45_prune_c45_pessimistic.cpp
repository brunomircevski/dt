#include "c45_tree.h"

void C45Tree::pruneWithC45Pessimistic(
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

    pruneWithC45Pessimistic(node->leftChild, partitions.leftRows);
    pruneWithC45Pessimistic(node->rightChild, partitions.rightRows);

    const std::string majorityLabel = getMajorityLabel(rowIndices);
    std::size_t leafCorrect = 0;
    for (std::size_t rowIndex : rowIndices)
    {
        if (dataset_->samples[rowIndex].label == majorityLabel)
        {
            ++leafCorrect;
        }
    }

    const std::size_t observedLeafErrors = rowIndices.size() - leafCorrect;
    const double estimatedLeafErrors =
        estimatePessimisticLeafErrorCount(observedLeafErrors, rowIndices.size());
    const double estimatedSubtreeErrors =
        estimateSubtreePessimisticErrorCount(node.get(), rowIndices);

    // This mode is closer to the spirit of C4.5:
    // compare the pessimistic error estimate of a replacement leaf against
    // the summed pessimistic estimates of all leaves in the subtree.
    if (estimatedLeafErrors <= estimatedSubtreeErrors + options_.epsilon)
    {
        node = Node::createLeaf(majorityLabel, rowIndices.size());
    }
}
