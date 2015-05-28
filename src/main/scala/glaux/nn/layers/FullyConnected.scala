package glaux.nn.layers

import glaux.nn.Dimension
import glaux.nn._
import glaux.nn.layers.FullyConnected.{Bias, Filter}

case class FullyConnected(filter: Filter, bias: Bias) extends HiddenLayer {

  type Output = RowVector
  type Input = RowVector

  def backward(input: Input, outGradient: OutGradient): (InGradient, Seq[ParamGradient]) = ???

  def updateParams(params: Iterable[LayerParam]): HiddenLayer = ???

  def outDimension: OutDimension = Dimension.Row(filter.dimension.y)

  def inDimension: InDimension = Dimension.Row(filter.dimension.x)

  def forward(input: Input, isTraining: Boolean): Output = ???

}

object FullyConnected {
  type Filter = Vol3D
  type Bias = RowVector

}
