package glaux.nn

trait Net {
  def inputLayer: InputLayer
  def layers: Seq[HiddenLayer]
  def lossLayer: LossLayer

  private def allLayers: Seq[Layer] = inputLayer +: layers :+ lossLayer

  def forward(input: Input, isTraining: Boolean = false): Output =
    allLayers.foldLeft(input)((act, layer) => layer.forward(act, isTraining))

  def backward(target: Output, actual: Output): (Loss, Seq[ParamGradient]) = {
    val (loss, lossLayerInGrad) = lossLayer.loss(target, actual)
    val (_, netParamGrads) = layers.foldRight((lossLayerInGrad, Seq[ParamGradient]())) { (layer, pair) =>
      val (outGradient, accuParamGrads) = pair
      val (inGradient, layerParamGrads) = layer.backward(outGradient)
      ( inGradient, layerParamGrads ++: accuParamGrads )
    }
    (loss, netParamGrads)
  }
}
