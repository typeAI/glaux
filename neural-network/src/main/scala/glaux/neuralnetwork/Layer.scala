package glaux.neuralnetwork

import glaux.linearalgebra.{RowVector, Tensor}

trait Layer {

  type Input <: Tensor
  type Output <: Tensor
  type InDimension = Input#Dimensionality
  type OutDimension = Output#Dimensionality

  type InGradient = Input
  type OutGradient = Gradient[Output]
  def inDimension: InDimension
  def outDimension: OutDimension

  def forward(input: Input, isTraining: Boolean): Output

}

case class Gradient[T <: Tensor](data: T, gradient: T) {
  assert(data.dimension == gradient.dimension)
}

trait HiddenLayer extends Layer {
  def id: String //should be unique within a network
  def backward(input: Input, outGradient: OutGradient): (InGradient, Seq[ParamGradient])
  def updateParams(params: Iterable[LayerParam]): HiddenLayer //This should not change the unique id of the layer
  def params: Seq[LayerParam]
}

object HiddenLayer {
  def newId(): String = java.util.UUID.randomUUID.toString
}

trait LossLayer extends Layer {
  type Input = RowVector
  type Output = Input
  def loss(target: Output, actual: Output): (Loss, InGradient)
}

case class InputLayer[I <: Tensor](inDimension: I#Dimensionality) extends Layer {
  type Input = I
  type Output = Input
  def outDimension: OutDimension = inDimension
  def forward(input: Input, isTraining: Boolean) = input
}

case class LayerParam(id: String, value: Tensor, regularizationSetting: RegularizationSetting)
case class LayerData[L <: Layer](in: L#Input, out: L#Output, layer: L)

case class RegularizationSetting(l1DM: DecayMultiplier, l2DM: DecayMultiplier)
