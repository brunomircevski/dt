#include "build/cpu_sweep.h"

namespace dt {

SweepSetup CpuTreeBuilder::sweepSetup() const {
  return {codec_,           tables_.xlog,      tables_.inverse, rules_.minValueGap(),
          rules_.logTable(), pool_ != nullptr, classCount_};
}

CutCandidate CpuTreeBuilder::scanFeature(const Entry *entries, std::uint32_t count,
                                         const std::uint32_t *total, double parentWeighted,
                                         std::uint32_t minChild, std::uint32_t *bestLeft) const {
  if (rules_.criterion() == Criterion::Gini) {
    return scanFeatureFor<Criterion::Gini>(entries, count, total, parentWeighted, minChild,
                                           bestLeft);
  }
  return scanFeatureFor<Criterion::Entropy>(entries, count, total, parentWeighted, minChild,
                                            bestLeft);
}

template <Criterion Crit>
CutCandidate CpuTreeBuilder::scanFeatureFor(const Entry *entries, std::uint32_t count,
                                            const std::uint32_t *total, double parentWeighted,
                                            std::uint32_t minChild,
                                            std::uint32_t *bestLeft) const {
  switch (classCount_) {
  case 2:
    if (count < tables_.size) {
      return rules_.minValueGap() > 0.0
                 ? scanTwoClasses<Crit, true>(entries, count, total, parentWeighted, minChild,
                                              bestLeft)
                 : scanTwoClasses<Crit, false>(entries, count, total, parentWeighted, minChild,
                                               bestLeft);
    }
    return scanFeatureK<2, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 3: return scanFeatureK<3, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 4: return scanFeatureK<4, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 5: return scanFeatureK<5, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 6: return scanFeatureK<6, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 7: return scanFeatureK<7, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 8: return scanFeatureK<8, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  default: return scanFeatureK<0, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  }
}

template <Criterion Crit, bool HasGap>
CutCandidate CpuTreeBuilder::scanTwoClasses(const Entry *entries, std::uint32_t count,
                                            const std::uint32_t *total, double parentWeighted,
                                            std::uint32_t minChild,
                                            std::uint32_t *bestLeft) const {
  TwoClassSweep<Crit, HasGap> sweep(sweepSetup(), count, total, parentWeighted, minChild);
  sweep.advance(entries, count);
  return sweep.finish(bestLeft);
}

template <int FixedK, Criterion Crit>
CutCandidate CpuTreeBuilder::scanFeatureK(const Entry *entries, std::uint32_t count,
                                          const std::uint32_t *total, double parentWeighted,
                                          std::uint32_t minChild, std::uint32_t *bestLeft) const {
  KClassSweep<FixedK, Crit> sweep(sweepSetup(), count, total, parentWeighted, minChild);
  sweep.advance(entries, count);
  return sweep.finish(bestLeft);
}

} // namespace dt
