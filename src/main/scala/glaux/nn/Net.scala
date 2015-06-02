package glaux.nn

import glaux.linalg.Vol

trait Net {
  type Input <: Vol
  type Output = LossLayer#Output
  def inputLayer: InputLayer[Input]
  def layers: Seq[HiddenLayer]
  def lossLayer: LossLayer

  def allLayers: Seq[Layer] = inputLayer +: layers :+ lossLayer

  def forward(input: Input): DataFlow = {
    val (_, dataFlow) = allLayers.foldLeft[(Vol, DataFlow)]((input, Vector[LayerData[_]]())) { (pair, layer) =>
      val (lastOutput, dataFlow) = pair
      val in = lastOutput.asInstanceOf[layer.Input] //cast to this input
      val out = layer.forward(in, true)
      (out, dataFlow :+ LayerData(in, out, layer))
    }
    dataFlow
  }

  def predict(input: Input): Output = finalOutput(forward(input))

  def backward(target: Output, dataFlow: DataFlow): (Loss, NetParamGradients) = {
    val (loss, lossLayerInGrad) = lossLayer.loss(target, finalOutput(dataFlow))
    val (_, netParamGrads) = layers.foldRight[(Vol, NetParamGradients)]((lossLayerInGrad, Map())) { (layer, pair) =>
      val (outGradientValue, accuParamGrads) = pair
      val layerData = findData[layer.type](dataFlow, layer)
      val (inGradient, layerParamGrads) = layer.backward(layerData.in, Gradient(layerData.out, outGradientValue.asInstanceOf[layer.Output]))
      (inGradient, accuParamGrads + (layer -> layerParamGrads))
    }
    (loss, netParamGrads)
  }

  private def finalOutput(dataFlow: DataFlow): Output = dataFlow.last.out.asInstanceOf[Output]
  private def findData[L <: Layer](dataFlow: DataFlow, layer: L): LayerData[L] =
    dataFlow.find(_.layer == layer).get.asInstanceOf[LayerData[L]]

}



object Net {
  type CanBuildFrom[N <: Net] = (N, Iterable[HiddenLayer]) => N

  case class SimpleNet[InputT <: Vol](inputLayer: InputLayer[InputT], layers: Seq[HiddenLayer], lossLayer: LossLayer) extends Net {
    type Input = InputT
    val assertDimensionIntegrity = allLayers.reduce { (lastLayer, thisLayer) =>
        assert(lastLayer.outDimension == thisLayer.inDimension)
        thisLayer
      }
  }

  implicit def simpleUpdater[Input <: Vol]: CanBuildFrom[SimpleNet[Input]] = (net, newLayers) => net.copy(layers = newLayers.toSeq)

  def apply[Input <: Vol](inputDimension: Input#Dimensionality, hiddenLayers: Seq[HiddenLayer], lossLayer: LossLayer): Net = SimpleNet(
    InputLayer[Input](inputDimension), hiddenLayers, lossLayer
  )
}