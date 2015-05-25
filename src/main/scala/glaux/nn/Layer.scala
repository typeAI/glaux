package glaux.nn

trait Layer {
  def inputDimension: Dimension
  def outDimension: Dimension
  def forward(input: Input, isTraining: Boolean): Output
}

trait HiddenLayer extends Layer {
  def backward(input: Input, outGradient: OutGradient): (InGradient, Seq[ParamGradient])
  def updateParams(params: Iterable[LayerParam]): HiddenLayer
}

trait LossLayer extends Layer {
  def loss(target: Output, actual: Output): (Loss, InGradient)
}

trait InputLayer extends Layer

case class LayerParam(id: String, value: Vol, layer: HiddenLayer )

case class LayerData(in: Input, out: Output, layer: Layer)