package glaux.nn.layers

import org.nd4j.api.linalg.DSL._
import glaux.nn._

case class Regression private(inDimension: DimensionArray) extends LossLayer {
  def loss(target: Output, actual: Output): (Loss, InGradient) = {
    assert(target.dimension == outDimension && actual.dimension == outDimension)
    val gradient: Output = actual - target
    val losses: Output = (gradient * gradient) * 0.5
    (losses.sum, gradient)
  }

  val outDimension: OutDimension = inDimension

  def forward(input: Input, isTraining: Boolean = false): Output = input
}

object Regression {
  def apply(numOfOutputs: Int) : Regression = Regression(DimensionArray(numOfOutputs))
}
