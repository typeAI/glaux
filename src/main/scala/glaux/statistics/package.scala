package glaux

import glaux.statistics.impl.apache.ApacheImplementations

import scala.util.Random

package object statistics {
  val distributions: DistributionImplementations = ApacheImplementations

  /**
   *
   * @param value must be between 0 and 1, note this is not enforced due to performance concern.
   */
  implicit class Probability(val value: Double) extends AnyVal {
    def +(that: Probability): Probability = value + that.value
    def -(that: Probability): Probability = value - that.value
    def *(that: Probability): Probability = value * that.value
    def /(factor: Double): Probability = value / factor
    def > (that: Probability): Boolean = value > that.value
    def < (that: Probability): Boolean = value < that.value
    def nextBoolean() : Boolean = Random.nextDouble() < value
  }
}
