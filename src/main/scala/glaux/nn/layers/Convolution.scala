package glaux.nn.layers

import glaux.linalg.Dimension.{FourD, ThreeD, TwoD}
import glaux.linalg.{Tensor4, RowVector, Vol, Matrix}
import glaux.nn._

/**
 *
 * @param filters a {@code Tensor4} to represent all filters
 * @param bias
 * @param stride
 * @param pad
 * @param inputSize
 * @param filterRegularization
 * @param id
 */
case class Convolution( filters: Tensor4,
                        bias: Vol,
                        stride: Int,
                        pad: Rectangle,
                        inputSize: Rectangle,
                        filterRegularization: RegularizationSetting,
                        id: String) extends HiddenLayer with MovingFilter {

  private val biasRegularization = RegularizationSetting(0, 0)
  assert(bias.dimension ==  ThreeD(1, 1, filters.dimension.f), "bias size matches number of filters")
  val filterSize = Rectangle(filters.dimension.x, filters.dimension.y)
  val inputDepth = filters.dimension.z
  lazy val filtersParam: LayerParam = LayerParam("filters", filters, filterRegularization)
  lazy val biasParam: LayerParam = LayerParam("bias", bias, biasRegularization)

  type Output = Vol
  type Input = Vol

  def outDimension: OutDimension = ThreeD(outSize.x, outSize.y, filters.dimension.f)

  def inDimension: InDimension = ThreeD(inputSize.x, inputSize.y, inputDepth)

  def params: Seq[LayerParam] = Seq(filtersParam, biasParam)

  def forward(input: Input, isTraining: Boolean): Output = {
    val Array(inputXRange, inputYRange, _) = input.dimension.ranges
    val Array(filterXRange, filterYRange, filterZRange, filterFRange) = filters.dimension.ranges
    val values = for {
                   f <- filterFRange
                   offsetY <- inputPlaneRanges.y
                   offsetX <- inputPlaneRanges.x
                 } yield ( for (x <- filterXRange; y <- filterYRange; z <- filterZRange)
                           yield {
                            val (ix, iy) = (x + offsetX, y + offsetY)
                            filters(x, y, z, f) * (if(inPaddedArea(ix, iy)) 0 else input(ix, iy, z))
                          }).sum + bias(0, 0, f)
    Vol(outDimension, values)
  }

  def backward(input: Input, outGradient: OutGradient): (InGradient, Seq[ParamGradient]) = {
    val Array(inputXRange, inputYRange, inputZRange) = input.dimension.ranges
    val Array(filterXRange, filterYRange, filterZRange, filterFRange) = filters.dimension.ranges
    val Array(outXRange, outYRange, outZRange) = outDimension.ranges

    def outGradValue(x: Int, filterX: Int, y: Int, filterY: Int, filterF: Int): Double =
      mappedOutCoordinate(x, filterX, y, filterY).map {
        case (outX, outY) => outGradient.gradient(outX, outY, filterF)
      }.getOrElse(0)

    val inGradValues = for (z <- inputZRange; y <- inputYRange; x <- inputXRange)
                       yield ( for (fx <- filterXRange; fy <- filterYRange; ff <- filterFRange)
                               yield outGradValue(x, fx, y, fy, ff) * filters(fx, fy, z, ff)).sum

    val inGrad = Vol(inDimension, inGradValues)

    val filtersGradValues = for (ff <- filterFRange; z <- filterZRange; fy <- filterYRange; fx <- filterXRange)
                       yield ( for (y <- inputYRange; x <- inputXRange)
                               yield outGradValue(x, fx, y, fy, ff) * input(x, y, z)).sum
    val filtersGrad = Tensor4(filters.dimension, filtersGradValues)
    val biasGrad = Vol(1, 1, 3,
      for(f <- outZRange) yield (for(x <- outXRange; y <- outYRange) yield outGradient.gradient(x,y,f)).sum
    )
    (inGrad, Seq(ParamGradient(biasParam, biasGrad), ParamGradient(filtersParam, filtersGrad)))
  }

  def updateParams(params: Iterable[LayerParam]): HiddenLayer = copy(
    filters = params.find(_.id == "filters").get.value.asInstanceOf[Tensor4],
    bias = params.find(_.id == "bias").get.value.asInstanceOf[Vol]
  )

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
