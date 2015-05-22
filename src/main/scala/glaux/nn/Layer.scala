package glaux.nn

trait Layer {
  def outDimension: Dimension
  def inputDimension: Dimension

  def forward(v: Vol, isTraining: Boolean): Vol

  def backward

  def gradients: Seq[Gradient]

}
