package glaux.nn.layers

import glaux.linalg.Vol
import Vol.CanBuildFrom
import glaux.linalg.Vol

import scala.math.max
import glaux.nn._

case class Relu[DataType <: Vol : CanBuildFrom](dimension: DataType#Dimensionality) extends HiddenLayer {

  type Output = DataType
  type Input = DataType


  def backward(input: Input, outGradient: OutGradient): (InGradient, Seq[ParamGradient]) = {
    val inGradient = outGradient.data.merge(outGradient.gradient)((o, g) => if(o <= 0) 0 else g)
    (inGradient, Nil)
  }

  def updateParams(params: Iterable[LayerParam]): HiddenLayer = this

  def outDimension: OutDimension = dimension

  def inDimension: InDimension = dimension

  def forward(input: Input, isTraining: Boolean = false): Output =
    input.map(max(_, 0))

}
