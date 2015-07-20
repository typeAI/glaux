package glaux.nn.layers

import glaux.linalg.Dimension.{FourD, ThreeD, TwoD}
import glaux.linalg.{Tensor4, RowVector, Vol, Matrix}
import glaux.nn._

case class Convolution( filters: Tensor4,
                        bias: Vol,
                        stride: Int,
                        pad: Rectangle,
                        inputSize: Rectangle,
                        filterRegularization: RegularizationSetting,
                        id: String) extends HiddenLayer {

  private val biasRegularization = RegularizationSetting(0, 0)
  assert(bias.dimension ==  ThreeD(1, 1, filters.dimension.f), "bias size matches number of filters")

  val inputDepth = filters.dimension.z
  lazy val filtersParam: LayerParam = LayerParam("filters", filters, filterRegularization)
  lazy val biasParam: LayerParam = LayerParam("bias", bias, biasRegularization)

  type Output = Vol
  type Input = Vol

  def outDimension: OutDimension = ThreeD(inputSize.x, inputSize.y, filters.dimension.f)

  def inDimension: InDimension = ThreeD(inputSize.x, inputSize.y, filters.dimension.z)

  def params: Seq[LayerParam] = Seq(filtersParam, biasParam)

  def forward(input: Input, isTraining: Boolean): Output = {
    Vol(4, 4, 3, Seq(5, 1.5, 2.7, 4.9, -0.2, 2.9, 5.1, -0.3, 1.5, 3.2, -1.4, 1, 4.37, 2.1, -1.3, 5.87, 1.8, 1.2, 9.17, 0.5, 1.9, 5.6, -0.2, -0.8, 7.27, -0.8, 1.1, 11.27, -5.8, -3.8, 5.97, -1.4, -3.0, 4, -1.8, -2.4, 4.17, -0.3, -2.8, 7.27, 0.6, -6.1, 9.27, 0.5, -7.0, 4.6, 2.7, -3.4))
  }

  def backward(input: Input, outGradient: OutGradient): (InGradient, Seq[ParamGradient]) = ???

  def updateParams(params: Iterable[LayerParam]): HiddenLayer = ???

}


object Convolution {

  def convolute(from: Vol, weights: Tensor4): Vol = {
    assert(from.dimension ==  ThreeD(weights.dimension.x, weights.dimension.y, weights.dimension.z))
    val Array(sx, sy, sz, sf) = weights.dimension.shape.map(Range(0, _))
    val values = for(f <- sf) //todo: might be a performance hotspot
                   yield (for (x <- sx; y <- sy; z <- sz)
                       yield from(x, y, z) * weights(x, y, z, f)).sum
    Vol(1, 1, weights.dimension.f, values)
  }
  /**
   *
   * @param numOfFilters
   * @param filterSize
   * @param inputDimension
   * @param stride
   * @param padding whether to use padding to ensure the same output size (the x*y dimension of the vol)
   * @return
   */
  def apply(numOfFilters: Int,
            filterSize: Rectangle,
            inputDimension: ThreeD,
            stride: Int = 1,
            padding: Boolean = false,
            filterRegularization: RegularizationSetting = RegularizationSetting(0, 1)): Convolution = {
    if(padding)
      assert(filterSize.x % 2 == 1 && filterSize.y % 2 == 1, "filter dimension should be odd for the convenience of padding")

    val inputPlain = Rectangle(inputDimension.x, inputDimension.y)
    val pad: Rectangle = if(padding) filterSize + (inputPlain * (stride - 1)) - stride  else Rectangle(0, 0)
    Convolution(
      Tensor4.normalized(FourD(filterSize.x, filterSize.y, inputDimension.z, numOfFilters), filterSize.x * filterSize.y * inputDimension.z),
      Vol.fill(ThreeD(1, 1, numOfFilters), 0),
      stride,
      pad,
      Rectangle(inputDimension.x, inputDimension.y),
      filterRegularization,
      HiddenLayer.newId()
    )
  }
}
