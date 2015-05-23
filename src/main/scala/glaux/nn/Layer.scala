package glaux.nn

trait Layer {
  def inputDimension: Dimension
  def outDimension: Dimension
  def forward(input: Input, isTraining: Boolean): Output
}

trait HiddenLayer extends Layer {
  def backward(outGradient: OutGradient): (InGradient, Seq[ParamGradient])
  def updateParams(params: Seq[Param]): HiddenLayer
}

trait LossLayer extends Layer {
  def loss(target: Output, actual: Output): (Loss, InGradient)
}

trait InputLayer extends Layer

case class Param(id: String, value: Vol, layer: HiddenLayer )