#include "c45_tree.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
    double clampConfidenceFactor(double value)
    {
        // C4.5 / Weka J48: CF is in (0, 0.5]; values above 0.5 disable pessimistic inflation.
        return std::max(0.0, std::min(0.5, value));
    }

    double inverseStandardNormalCdf(double probability)
    {
        const double a1 = -39.6968302866538;
        const double a2 = 220.946098424521;
        const double a3 = -275.928510446969;
        const double a4 = 138.357751867269;
        const double a5 = -30.6647980661472;
        const double a6 = 2.50662827745924;

        const double b1 = -54.4760987982241;
        const double b2 = 161.585836858041;
        const double b3 = -155.698979859887;
        const double b4 = 66.8013118877197;
        const double b5 = -13.2806815528857;

        const double c1 = -0.00778489400243029;
        const double c2 = -0.322396458041136;
        const double c3 = -2.40075827716184;
        const double c4 = -2.54973253934373;
        const double c5 = 4.37466414146497;
        const double c6 = 2.93816398269878;

        const double d1 = 0.00778469570904146;
        const double d2 = 0.32246712907004;
        const double d3 = 2.445134137143;
        const double d4 = 3.75440866190742;

        const double pLow = 0.02425;
        const double pHigh = 1.0 - pLow;

        if (probability <= 0.0)
        {
            return -std::numeric_limits<double>::infinity();
        }
        if (probability >= 1.0)
        {
            return std::numeric_limits<double>::infinity();
        }

        if (probability < pLow)
        {
            const double q = std::sqrt(-2.0 * std::log(probability));
            return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }

        if (probability <= pHigh)
        {
            const double q = probability - 0.5;
            const double r = q * q;
            return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
                   (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
        }

        const double q = std::sqrt(-2.0 * std::log(1.0 - probability));
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }

    // Weka J48 / C4.5: extra predicted errors beyond the observed count e.
    double addErrs(double sampleCount, double observedErrors, double confidenceFactor)
    {
        if (confidenceFactor > 0.5)
        {
            return 0.0;
        }

        if (observedErrors < 1.0)
        {
            const double base =
                sampleCount * (1.0 - std::pow(confidenceFactor, 1.0 / sampleCount));
            if (observedErrors == 0.0)
            {
                return base;
            }

            return base + observedErrors * (addErrs(sampleCount, 1.0, confidenceFactor) - base);
        }

        if (observedErrors + 0.5 >= sampleCount)
        {
            return std::max(sampleCount - observedErrors, 0.0);
        }

        const double z = inverseStandardNormalCdf(1.0 - confidenceFactor);
        const double errorRate = (observedErrors + 0.5) / sampleCount;
        const double zSquared = z * z;
        const double upperBoundRate =
            (errorRate + zSquared / (2.0 * sampleCount) +
             z * std::sqrt((errorRate / sampleCount) - (errorRate * errorRate / sampleCount) +
                           (zSquared / (4.0 * sampleCount * sampleCount)))) /
            (1.0 + zSquared / sampleCount);

        return std::max(upperBoundRate * sampleCount - observedErrors, 0.0);
    }
}

double C45Tree::estimatePessimisticLeafErrorCount(
    std::size_t observedErrors,
    std::size_t sampleCount
) const
{
    if (sampleCount == 0)
    {
        return 0.0;
    }

    const double confidenceFactor = clampConfidenceFactor(options_.pruningConfidenceFactor);
    const double sampleCountDouble = static_cast<double>(sampleCount);
    const double observedErrorsDouble = static_cast<double>(observedErrors);

    return observedErrorsDouble +
           addErrs(sampleCountDouble, observedErrorsDouble, confidenceFactor);
}

double C45Tree::estimateSubtreePessimisticErrorCount(
    const Node* node,
    const std::vector<std::size_t>& rowIndices
) const
{
    if (!node)
    {
        return 0.0;
    }

    if (node->isLeaf)
    {
        std::size_t observedErrors = 0;
        for (std::size_t rowIndex : rowIndices)
        {
            if (dataset_->samples[rowIndex].label != node->leafLabel)
            {
                ++observedErrors;
            }
        }

        return estimatePessimisticLeafErrorCount(observedErrors, rowIndices.size());
    }

    const PartitionedRows partitions =
        partitionRows(rowIndices, node->featureIndex, node->threshold);

    return estimateSubtreePessimisticErrorCount(node->leftChild.get(), partitions.leftRows) +
           estimateSubtreePessimisticErrorCount(node->rightChild.get(), partitions.rightRows);
}


void C45Tree::prunePessimisticError(
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

    prunePessimisticError(node->leftChild, partitions.leftRows);
    prunePessimisticError(node->rightChild, partitions.rightRows);

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

    if (estimatedLeafErrors <= estimatedSubtreeErrors + options_.epsilon)
    {
        node = Node::createLeaf(majorityLabel, rowIndices.size());
    }
}
