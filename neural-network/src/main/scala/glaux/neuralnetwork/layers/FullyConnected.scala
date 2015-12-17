package glaux.neuralnetwork.layers

import glaux.linearalgebra.Dimension.{Row, TwoD}
import glaux.linearalgebra.Tensor.TensorBuilder
import glaux.linearalgebra.{Tensor, RowVector, Matrix, Dimension}
import glaux.neuralnetwork._
import glaux.neuralnetwork.layers.FullyConnected.{Bias, Filter}

case class FullyConnected[IT <: Tensor : TensorBuilder]( filter: Filter,
                                                         bias: Bias,
                                                         filterRegularization: RegularizationSetting,
                                                         inDimension: IT#Dimensionality,
                                                         id: String ) extends HiddenLayer {
  private val biasRegularization = RegularizationSetting(0, 0)
  lazy val filterParam: LayerParam = LayerParam("filter", filter, filterRegularization)
  lazy val biasParam: LayerParam = LayerParam("bias", bias, biasRegularization)
  type Output = RowVector
  type Input = IT

  val outDimension: OutDimension = Dimension.Row(filter.dimension.y)

  assert(bias.dimension == outDimension, s"bias dimension ${bias.dimension} must match out dimension $outDimension")

  def params = Seq(filterParam, biasParam)

  def updateParams(params: Iterable[LayerParam]): HiddenLayer = {
    val f = params.find(_.id == "filter").get.value.asInstanceOf[Matrix]
    val b = params.find(_.id == "bias").get.value.asInstanceOf[RowVector]
    copy(f, b)
  }

  def backward(input: Input, outGradient: OutGradient): (InGradient, Seq[ParamGradient]) = {
    val og = outGradient.gradient
    val filterGradient: Matrix = input.toRowVector.T ** og
    val biasGradient: RowVector = og
    val inGradient: Input = toInputFormat(og ** filter.T)
    ( inGradient, Seq[ParamGradient](
      ParamGradient(filterParam, filterGradient),
      ParamGradient(biasParam, biasGradient))
    )
  }

  def forward(input: Input, isTraining: Boolean = false): Output = {
    assert(input.dimension == inDimension, s"incorrect input dimension ${input.dimension} vs ${inDimension}")
    (input.toRowVector ** filter) + bias
  }

  def updateFilterBias(filter: Filter, bias: Bias): HiddenLayer = copy(filter, bias)

  private def toInputFormat(vector: RowVector): Input = {
    if(vector.dimension == inDimension) vector.asInstanceOf[Input] else (inDimension, vector.seqView)
  }
}

object FullyConnected {
  type Filter = Matrix
  type Bias = RowVector

  private def bias(numOfNeurons: Int): Bias = RowVector.fill(Row(numOfNeurons), 0)
  private  def filter(numOfFeatures: Int, numOfNeurons: Int): Filter = Matrix.normalized(TwoD(numOfFeatures, numOfNeurons), numOfFeatures)

  def apply(numOfFeatures: Int, numOfNeurons: Int): FullyConnected[RowVector] =
    FullyConnected(filter(numOfFeatures, numOfNeurons), bias(numOfNeurons))


  def apply[T <: Tensor: TensorBuilder](inputDimension: T#Dimensionality, numOfNeurons: Int): FullyConnected[T] =
    FullyConnected[T](filter(inputDimension.totalSize, numOfNeurons), bias(numOfNeurons), RegularizationSetting(0, 1), inputDimension, HiddenLayer.newId())


 private[glaux] def apply(filter: Filter, bias: Bias): FullyConnected[RowVector] =
   FullyConnected[RowVector](filter, bias, RegularizationSetting(0, 1), Row(filter.dimension.x), HiddenLayer.newId())

}
