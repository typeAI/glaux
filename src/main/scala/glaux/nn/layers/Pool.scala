package glaux.nn.layers

import glaux.linalg.Dimension.ThreeD
import glaux.linalg.Vol
import glaux.nn.{ParamGradient, LayerParam, Rectangle, HiddenLayer}

case class Pool( filterSize: Rectangle,
                 stride: Int,
                 pad: Rectangle,
                 inDimension: ThreeD,
                 id: String
               ) extends HiddenLayer with MovingFilter {
  type Output = Vol
  type Input = Vol
  def inputSize = Rectangle.planeSize(inDimension)
  def outDimension: OutDimension = ThreeD(outSize.x, outSize.y, inDimension.z)
  def params: Seq[LayerParam] = Nil
  def updateParams(params: Iterable[LayerParam]): HiddenLayer = this

  def backward(input: Vol, outGradient: OutGradient): (InGradient, Seq[ParamGradient]) = ???

  def forward(input: Vol, isTraining: Boolean): Vol = ???
}
