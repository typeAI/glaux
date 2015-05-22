package glaux.nn

trait Layer {
  def outDimension: Dimension
  def inputDimension: Dimension



  def forward(input: Input, isTraining: Boolean): Output

  def backward(outGradient: OutGradient): (InGradient, Seq[ParameterGradient])


}
