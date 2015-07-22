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

  private lazy val Array(inputXs, inputYs, inputZs) = inDimension.ranges
  private lazy val (filterXs, filterYs) = (0 until filterSize.x, 0 until filterSize.y)

  def forward(input: Vol, isTraining: Boolean): Vol = {
    val values = for {
      z <- 0 until inDimension.z
      offsetY <- inputPlaneRanges.ys
      offsetX <- inputPlaneRanges.xs
    } yield ( for (x <- 0 until filterSize.x; y <- 0 until filterSize.y)
        yield {
          val (ix, iy) = (x + offsetX, y + offsetY)
          if(inPaddedArea(ix, iy)) 0 else input(ix, iy, z)
        }).max
    Vol(outDimension, values)
  }

  def backward(input: Vol, outGradient: OutGradient): (InGradient, Seq[ParamGradient]) = {

    val inGradValues: Seq[Double] = for (z <- inputZs; y <- inputYs; x <- inputXs)
      yield ( for (filterX <- filterXs; filterY <- filterYs)
        yield {
          mappedOutCoordinate(x, filterX, y, filterY).map {
            case (outX, outY) => (outGradient.data(outX, outY, z), outGradient.gradient(outX, outY, z))
          }
        }).find {
        case Some((outValue, outGrad)) => outValue == input(x, y, z)
        case None => false
      }.flatten.map(_._2).getOrElse(0d)

    val inGrad = Vol(inDimension, inGradValues)
    (inGrad, Nil)
  }

}


object Pool {
  def apply( inDimension: ThreeD,
             filterSize: Rectangle,
             stride: Int,
             padding: Boolean = false): Pool = {
    val pad: Rectangle = if(padding) {
      val x = ((stride - (inDimension.x - filterSize.x) % stride) % stride) / 2
      val y = ((stride - (inDimension.y - filterSize.y) % stride) % stride) / 2
      Rectangle(x, y)
    } else Rectangle(0, 0)

    Pool(filterSize, stride, pad, inDimension, HiddenLayer.newId())

  }
}
