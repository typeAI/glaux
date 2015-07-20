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

  def outDimension: OutDimension = ThreeD(
    (inputSize.x + (pad.x * 2) - filters.dimension.x ) / stride + 1,
    (inputSize.y + (pad.y * 2) - filters.dimension.y ) / stride + 1,
    filters.dimension.f)

  def inDimension: InDimension = ThreeD(inputSize.x, inputSize.y, inputDepth)

  def params: Seq[LayerParam] = Seq(filtersParam, biasParam)

  def forward(input: Input, isTraining: Boolean): Output = {
    val Array(inputXRange, inputYRange, _) = input.dimension.ranges
    val Array(filterXRange, filterYRange, filterZRange, filterFRange) = filters.dimension.ranges
    val values = for {
                   f <- filterFRange
                   offsetY <- Range.inclusive(- pad.y, input.dimension.y + pad.y - filters.dimension.y, stride)
                   offsetX <- Range.inclusive(- pad.x, input.dimension.x + pad.x - filters.dimension.x, stride)
                 } yield ( for (x <- filterXRange; y <- filterYRange; z <- filterZRange)
                           yield {
                            val (ix, iy) = (x + offsetX, y + offsetY)
                            val inPaddedArea = !(inputXRange.contains(ix) && inputYRange.contains(iy))
                            filters(x, y, z, f) * (if(inPaddedArea) 0 else input(ix, iy, z))
                          }).sum
    Vol(outDimension, values)
  }

  def backward(input: Input, outGradient: OutGradient): (InGradient, Seq[ParamGradient]) = {
    val Array(inputXRange, inputYRange, inputZRange) = input.dimension.ranges
    val Array(filterXRange, filterYRange, _, filterFRange) = filters.dimension.ranges

    val inGradValues = for (z <- inputZRange; y <- inputYRange; x <- inputXRange)
                       yield ( for (fx <- filterXRange; fy <- filterYRange; ff <- filterFRange)
                               yield {
                                 println((x - fx + 1, y - fy + 1, z, ff))
                                  outGradient.gradient(fx, fy, ff) * filters(x - fx + 1, y - fy + 1, z, ff)
                                }).sum
    val inGrad = Vol(inDimension, inGradValues)


    (input, Nil)
  }

  def updateParams(params: Iterable[LayerParam]): HiddenLayer = ???

}


object Convolution {
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
    val pad: Rectangle = if(padding) (filterSize + (inputPlain * (stride - 1)) - stride) / 2  else Rectangle(0, 0)
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
