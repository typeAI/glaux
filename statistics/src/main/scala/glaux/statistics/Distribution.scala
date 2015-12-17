package glaux.statistics

trait Distribution {
  type Value

  def sample(size: Int): Iterable[Value]
  def sample: Value = sample(1).head
}

trait RealDistribution extends Distribution {
  type Value = Double
}

trait DistributionImplementations {
  def normal(mean: Double, std: Double): RealDistribution
  def uniform(min: Double = 0, max: Double = 1): RealDistribution
}