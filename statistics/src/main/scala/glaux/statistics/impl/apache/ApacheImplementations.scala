package glaux.statistics.impl.apache

import glaux.statistics.{RealDistribution, DistributionImplementations}
import org.apache.commons.math3.distribution.{RealDistribution ⇒ ApacheDistribution, UniformRealDistribution, NormalDistribution}

object ApacheImplementations extends DistributionImplementations {
  implicit class ApacheBackedRealDist(ad: ApacheDistribution) extends RealDistribution {
    def sample(size: Int) = ad.sample(size)
  }

  def normal(mean: Double, std: Double): RealDistribution = new NormalDistribution(mean, std)

  def uniform(min: Double = 0, max: Double = 1): RealDistribution = new UniformRealDistribution(min, max)
}
