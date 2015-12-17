package glaux

import glaux.statistics.impl.apache.ApacheImplementations

import scala.util.Random

package object statistics {
  val distributions: DistributionImplementations = ApacheImplementations

  /**
   *
   * @param value must be between 0 and 1, note this is not enforced due to performance concern.
   */
  implicit class Probability(val value: Double) extends AnyVal with Ordered[Probability] {
    def +(that: Probability): Probability = value + that.value
    def -(that: Probability): Probability = value - that.value
    def *(that: Probability): Probability = value * that.value
    def /(factor: Double): Probability = value / factor
    def nextBoolean(): Boolean = Random.nextDouble() < value
    override def toString: String = "Probability of " + value
    def compare(that: Probability): Int = value.compare(that.value)

  }

}
