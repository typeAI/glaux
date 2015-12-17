package glaux.neuralnetwork

import glaux.linearalgebra.{RowVector, Tensor}

trait Net {
  type Input <: Tensor
  final type Output = LossLayer#Output
  def inputLayer: InputLayer[Input]
  def hiddenLayers: Seq[HiddenLayer]
  def lossLayer: LossLayer
  def inputDimension: Input#Dimensionality = inputLayer.inDimension
  def outputDimension: Output#Dimensionality = lossLayer.outDimension
  def allLayers: Seq[Layer] = inputLayer +: hiddenLayers :+ lossLayer

  //throws assertion exceptions
  def validate(): Unit = {
    def assertUniqueness[T](seq: Seq[T], message: String): Unit = assert(seq.distinct.size == seq.size, message)
    allLayers.reduce { (lastLayer, thisLayer) =>
      assert(lastLayer.outDimension == thisLayer.inDimension, s"${lastLayer.getClass.getSimpleName}'s output ${lastLayer.outDimension} doesn't match ${thisLayer.getClass.getSimpleName}'s input ${thisLayer.inDimension}")
      thisLayer
    }
    assertUniqueness(hiddenLayers.map(_.id), "Some hidden layers share the same id")
    hiddenLayers.foreach { l =>
      assertUniqueness(l.params.map(_.id), "Some layers have params that share the same id")
    }
  }

  def forward(input: Input): DataFlow = {
    val (_, dataFlow) = allLayers.foldLeft[(Tensor, DataFlow)]((input, Vector[LayerData[_]]())) { (pair, layer) =>
      val (lastOutput, dataFlow) = pair
      val in = lastOutput.asInstanceOf[layer.Input] //cast to this input, hard to have compiler check for dynamically constructed layers.
      val out = layer.forward(in, true)
      (out, dataFlow :+ LayerData(in, out, layer))
    }
    dataFlow
  }

  def predict(input: Input): Output = finalOutput(forward(input))

  def backward(target: Output, dataFlow: DataFlow): (Loss, NetParamGradients) = {
    val (loss, lossLayerInGrad) = lossLayer.loss(target, finalOutput(dataFlow))
    val (_, netParamGrads) = hiddenLayers.foldRight[(Tensor, NetParamGradients)]((lossLayerInGrad, Map())) { (layer, pair) =>
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
  type Updater[N <: Net] = (N, Iterable[HiddenLayer]) => N

  case class DefaultNet[InputT <: Tensor](inputLayer: InputLayer[InputT], hiddenLayers: Seq[HiddenLayer], lossLayer: LossLayer) extends Net {
    type Input = InputT
    validate()
  }


  implicit def simpleUpdater[Input <: Tensor]: Updater[DefaultNet[Input]] = (net, newLayers) => {
    net.hiddenLayers.map(_.id).zip(newLayers.map(_.id)).foreach {
      case (id1, id2) => assert(id1 == id2, "update layer cannot change layer ids and sequence")
    }
    net.copy(hiddenLayers = newLayers.toSeq)
  }

  def apply[Input <: Tensor](inputDimension: Input#Dimensionality, hiddenLayers: Seq[HiddenLayer], lossLayer: LossLayer): Net = DefaultNet(
    InputLayer[Input](inputDimension), hiddenLayers, lossLayer
  )
}
