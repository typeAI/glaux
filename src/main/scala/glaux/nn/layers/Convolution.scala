package glaux.nn.layers

import glaux.linalg.Dimension.{ThreeD, TwoD}
import glaux.linalg.{Tensor4, RowVector, Vol, Matrix}
import glaux.nn._
import glaux.nn.layers.Convolution.Filter

case class Convolution( filters: Tensor4,
                        bias: RowVector,
                        stride: Int,
                        pad: Rectangle,
                        inputSize: Rectangle,
                        filterRegularization: RegularizationSetting,
                        id: String) extends HiddenLayer {

  private val biasRegularization = RegularizationSetting(0, 0)
  assert(bias.dimension.size == filters.dimension.f, "bias size matches number of filters")

  val inputDepth = filters.dimension.z
  lazy val filterParam: LayerParam = LayerParam("filters", filters, filterRegularization)
  lazy val biasParam: LayerParam = LayerParam("bias", bias, biasRegularization)

  type Output = Vol
  type Input = Vol

  def outDimension: OutDimension = ThreeD(inputSize.x, inputSize.y, filters.dimension.f)

  def inDimension: InDimension = ThreeD(inputSize.x, inputSize.y, filters.dimension.z)

  def params: Seq[LayerParam] = ???

  def forward(input: Input, isTraining: Boolean): Output = ???

  def backward(input: Input, outGradient: OutGradient): (InGradient, Seq[ParamGradient]) = ???

  def updateParams(params: Iterable[LayerParam]): HiddenLayer = ???

}


object Convolution {
  case class Filter(weights: Vol, bias: Double)
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
    val pad = if(padding) filterSize + ( inputPlain * (stride - 1)) - stride else Rectangle(0, 0)
    val id = HiddenLayer.newId()
    ???
  }
}
