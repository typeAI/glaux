package glaux.nn.layers

import glaux.linalg.{Vol, RowVector, Matrix, Dimension}
import glaux.nn._
import glaux.nn.layers.FullyConnected.{Bias, Filter}


case class FullyConnected(filter: Filter,
                          bias: Bias,
                          filterRegularization: RegularizationSetting = RegularizationSetting(0, 1),
                          id: String = HiddenLayer.newId()) extends HiddenLayer {
  val inDimension: InDimension = Dimension.Row(filter.dimension.x)
  val outDimension: OutDimension = Dimension.Row(filter.dimension.y)
  private val biasRegularization = RegularizationSetting(0, 0)
  lazy val filterParam: LayerParam = LayerParam("filter", filter, filterRegularization)
  lazy val biasParam: LayerParam = LayerParam("bias", bias, biasRegularization)
  assert(bias.dimension == outDimension, s"bias dimension ${bias.dimension} must match out dimension $outDimension")

  type Output = RowVector
  type Input = RowVector

  def params = Seq(filterParam, biasParam)

  def backward(input: Input, outGradient: OutGradient): (InGradient, Seq[ParamGradient]) = {
    val og = outGradient.gradient
    val filterGradient: Matrix = input.T ** og
    val biasGradient: RowVector = og
    (og ** filter.T, Seq[ParamGradient](
      ParamGradient(filterParam, filterGradient),
      ParamGradient(biasParam, biasGradient))
    )
  }

  def updateParams(params: Iterable[LayerParam]): HiddenLayer = {
    val f = params.find(_.id == "filter").get.value.asInstanceOf[Matrix]
    val b = params.find(_.id == "bias").get.value.asInstanceOf[RowVector]
    copy(f, b)
  }

  def forward(input: Input, isTraining: Boolean = false): Output = {
    (input ** filter) + bias
  }

}

object FullyConnected {
  type Filter = Matrix
  type Bias = RowVector
}
