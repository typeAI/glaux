package glaux.nn.layers

import glaux.linalg._
import Vol._
import glaux.nn.{Loss, LossLayer}

case class Softmax(inDimension: Dimension.Row) extends LossLayer {

  def loss(target: RowVector, actual: RowVector): (Loss, InGradient) = {
    assert( target.dimension == actual.dimension &&
            target.sumAll == 1 &&
            target.seqView.forall(v => v == 0 || v == 1),
            "target must be same dimension as output with exact one 1 value and the rest 0s")

    val inGradient = actual - target
    val loss = - Math.log(actual dot target)
    (loss, inGradient)
  }

  val outDimension: OutDimension = inDimension

  def forward(input: RowVector, isTraining: Boolean): RowVector = {
    val maxV = input.seqView.max
    val exp = input.map( v => Math.exp(v - maxV))
    exp.normalize
  }
}
