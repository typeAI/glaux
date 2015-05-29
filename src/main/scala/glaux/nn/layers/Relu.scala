package glaux.nn.layers


import scala.math.max
import glaux.nn._

case class Relu[DataType <: Vol : VolOps](dimension: DataType#Dimensionality) extends HiddenLayer {

  type Output = DataType
  type Input = DataType

  val ops = implicitly[VolOps[DataType]]

  def backward(input: Input, outGradient: OutGradient): (InGradient, Seq[ParamGradient]) = {
    val inGradient =ops.map2(outGradient.data, outGradient.gradient, (o, g) => if(o <= 0) 0 else g)
    (inGradient, Nil)
  }

  def updateParams(params: Iterable[LayerParam]): HiddenLayer = this

  def outDimension: OutDimension = dimension

  def inDimension: InDimension = dimension

  def forward(input: Input, isTraining: Boolean = false): Output =
    ops.map(input, max(_, 0))

}
