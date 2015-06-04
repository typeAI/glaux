package glaux.nn.layers

import glaux.linalg.Vol
import Vol.VolBuilder
import Vol.VolOps

import scala.math.max
import glaux.nn._

case class Relu[DataType <: Vol : VolBuilder](dimension: DataType#Dimensionality, uuid: String = java.util.UUID.randomUUID.toString) extends HiddenLayer {

  type Output = DataType
  type Input = DataType


  def backward(input: Input, outGradient: OutGradient): (InGradient, Seq[ParamGradient]) = {
    val inGradient = VolOps(outGradient.data).merge(outGradient.gradient)((o, g) => if(o <= 0) 0 else g)
    (inGradient, Nil)
  }

  def updateParams(params: Iterable[LayerParam]): HiddenLayer = this

  def params: Seq[LayerParam] = Nil

  def outDimension: OutDimension = dimension

  def inDimension: InDimension = dimension

  def forward(input: Input, isTraining: Boolean = false): Output =
    input.map(max(_, 0))

}
