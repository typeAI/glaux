package glaux.nn

trait Net {
  def inputLayer: InputLayer
  def layers: Seq[HiddenLayer]
  def lossLayer: LossLayer

  private def allLayers: Seq[Layer] = inputLayer +: layers :+ lossLayer

  def forward(input: Input): DataFlow = {
    val (_, dataFlow) = allLayers.foldLeft((input, Vector[LayerData]())) { (pair, layer) =>
      val (in, dataFlow) = pair
      val out = layer.forward(in, true)
      (out, dataFlow :+ LayerData(in, out, layer))
    }
    dataFlow
  }

  def predict(input: Input): Output = {
    forward(input).last.out
  }

  def backward(target: Output, dataFlow: DataFlow): (Loss, Seq[ParamGradient]) = {
    val (loss, lossLayerInGrad) = lossLayer.loss(target, dataFlow.last.out)
    val (_, netParamGrads) = layers.foldRight((lossLayerInGrad, Seq[ParamGradient]())) { (layer, pair) =>
      val (outGradient, accuParamGrads) = pair
      val layerInput = dataFlow.find(_.layer == layer).get.in
      val (inGradient, layerParamGrads) = layer.backward(layerInput, outGradient)
      ( inGradient, layerParamGrads ++: accuParamGrads )
    }
    (loss, netParamGrads)
  }

}
