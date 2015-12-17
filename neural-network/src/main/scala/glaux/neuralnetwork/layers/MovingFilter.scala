package glaux.neuralnetwork.layers

import glaux.linearalgebra.Dimension.ThreeD
import glaux.linearalgebra.{ Vol, Tensor4 }
import glaux.neuralnetwork._

trait MovingFilter {

  def inputSize: Rectangle
  def stride: Int
  def pad: Rectangle
  def filterSize: Rectangle

  def inputPlaneRanges = RectangleRange(
    Range.inclusive(-pad.x, inputSize.x + pad.x - filterSize.x, stride),
    Range.inclusive(-pad.y, inputSize.y + pad.y - filterSize.y, stride)
  )

  def outSize: Rectangle = Rectangle(
    (inputSize.x + (pad.x * 2) - filterSize.x) / stride + 1,
    (inputSize.y + (pad.y * 2) - filterSize.y) / stride + 1
  )

  def inPaddedArea(x: Int, y: Int) = !inputSize.contains(x, y)

  def mappedOutCoordinate(inputX: Int, filterX: Int, inputY: Int, filterY: Int): Option[(Int, Int)] = {
    val outX = (inputX - filterX + pad.x) / stride
    val outY = (inputY - filterY + pad.y) / stride
    if (outSize.contains(outX, outY))
      Some((outX, outY))
    else
      None
  }

}
