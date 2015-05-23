package glaux.nn

trait Layer {
  def inputDimension: Dimension
  def outDimension: Dimension
  def forward(input: Input, isTraining: Boolean): Output
}

trait HiddenLayer extends Layer {
  def backward(outGradient: OutGradient): (InGradient, Seq[ParameterGradient])
}

trait LossLayer extends Layer {
  def loss(target: Output): (Loss, InGradient)
}

trait InputLayer extends Layer