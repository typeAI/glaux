package glaux.nn.layers

import glaux.nn.Dimension
import glaux.nn._
import glaux.nn.layers.FullyConnected.{Bias, Filter}
import org.nd4j.api.linalg.DSL._


case class FullyConnected(filter: Filter, bias: Bias) extends HiddenLayer {
  val inDimension: InDimension = Dimension.Row(filter.dimension.x)
  val outDimension: OutDimension = Dimension.Row(filter.dimension.y)

  lazy val filterParam: LayerParam = LayerParam("filter", filter, this)
  lazy val biasParam: LayerParam = LayerParam("bias", bias, this)

  assert(bias.dimension == outDimension)

  type Output = RowVector
  type Input = RowVector

  def backward(input: Input, outGradient: OutGradient): (InGradient, Seq[ParamGradient]) = {
    val filterGradient: Matrix = input.T ** outGradient
    val biasGradient: RowVector = outGradient
    (outGradient ** filter.T, Seq[ParamGradient](
      ParamGradient(filterParam, filterGradient, 0, 0),
      ParamGradient(biasParam, biasGradient, 0, 0))
    )
  }

  def updateParams(params: Iterable[LayerParam]): HiddenLayer = ???

  def forward(input: Input, isTraining: Boolean = false): Output = {
    (input ** filter).add(bias)
  }

}

object FullyConnected {
  type Filter = Matrix
  type Bias = RowVector

}
