#include "c45_tree.h"

void C45Tree::pruneWithPessimisticError(
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

    pruneWithPessimisticError(node->leftChild, partitions.leftRows);
    pruneWithPessimisticError(node->rightChild, partitions.rightRows);

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

    const double sampleCount = static_cast<double>(rowIndices.size());
    const double subtreeErrors = sampleCount - static_cast<double>(subtreeCorrect);
    const double leafErrors = sampleCount - static_cast<double>(leafCorrect);

    // This is the older project heuristic:
    // add 0.5 estimated error per leaf so a bigger subtree must "pay"
    // for its extra complexity.
    const double estimatedSubtreeErrorRate =
        (subtreeErrors + 0.5 * static_cast<double>(countLeafNodes(node.get())))
        / sampleCount;
    const double estimatedLeafErrorRate = (leafErrors + 0.5) / sampleCount;

    if (estimatedLeafErrorRate <= estimatedSubtreeErrorRate + options_.epsilon)
    {
        node = Node::createLeaf(majorityLabel, rowIndices.size());
    }
}
