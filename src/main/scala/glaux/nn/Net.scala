package glaux.nn

trait Net {
  def inputLayer: InputLayer
  def layers: Seq[HiddenLayer]
  def lossLayer: LossLayer

  private def allLayers: Seq[Layer] = inputLayer +: layers :+ lossLayer

  def forward(input: Input, isTraining: Boolean = false): Output =
    allLayers.foldLeft(input)((act, layer) => layer.forward(act, isTraining))

  def backward(target: Output): (Loss, Seq[ParameterGradient]) = {
    val (loss, lossLayerInGrad) = lossLayer.loss(target)
    val (_, netParamGrads) = layers.foldRight((lossLayerInGrad, Seq[ParameterGradient]())) { (layer, pair) =>
      val (outGradient, accuParamGrads) = pair
      val (inGradient, layerParamGrads) = layer.backward(outGradient)
      ( inGradient, layerParamGrads ++: accuParamGrads )
    }
    (loss, netParamGrads)
  }
}
