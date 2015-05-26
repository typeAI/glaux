package glaux.nn

trait Net[Input <: Vol] {
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

  def backward(target: Output, dataFlow: DataFlow): (Loss, Seq[ParamGradient]) = {
    val (loss, lossLayerInGrad) = lossLayer.loss(target, finalOutput(dataFlow))
    val (_, netParamGrads) = layers.foldRight[(Vol, Seq[ParamGradient])]((lossLayerInGrad, Nil)) { (layer, pair) =>
      val (outGradient, accuParamGrads) = pair
      val layerInput = findInput[layer.type](dataFlow, layer)
      val (inGradient, layerParamGrads) = layer.backward(layerInput, outGradient.asInstanceOf[layer.OutGradient])
      (inGradient, layerParamGrads ++: accuParamGrads)
    }
    (loss, netParamGrads)
  }

  private def finalOutput(dataFlow: DataFlow): Output = dataFlow.last.out.asInstanceOf[Output]
  private def findInput[L <: Layer](dataFlow: DataFlow, layer: L): L#Input =
    dataFlow.find(_.layer == layer).get.asInstanceOf[LayerData[L]].in

}


trait Updater[N <: Net[_]] {
  def update(net: N, newLayers: Iterable[HiddenLayer]): N
}

object Net {
  case class SimpleNet[Input <: Vol](inputLayer: InputLayer[Input], layers: Seq[HiddenLayer], lossLayer: LossLayer) extends Net[Input] {
    val assertDimensionIntegrity = allLayers.reduce { (lastLayer, thisLayer) =>
        assert(lastLayer.outDimension == thisLayer.inDimension)
        thisLayer
      }
  }

  implicit def simpleUpdater[Input <: Vol]: Updater[SimpleNet[Input]] = new Updater[SimpleNet[Input]]{
    def update(net: SimpleNet[Input], newLayers: Iterable[HiddenLayer]): SimpleNet[Input] = net.copy(layers = newLayers.toSeq)
  }

  def apply[Input <: Vol](inputDimension: Input#Dimensionality, hiddenLayers: Seq[HiddenLayer], lossLayer: LossLayer): Net[Input] = SimpleNet(
    InputLayer[Input](inputDimension), hiddenLayers, lossLayer
  )
}