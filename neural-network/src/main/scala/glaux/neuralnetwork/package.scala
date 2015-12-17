package glaux

package object neuralnetwork {
  type DecayMultiplier = Double
  type Loss = Double
  type Decay = Double
  type DataFlow = Vector[LayerData[_]]
  type NetParamGradients = Map[HiddenLayer, Seq[ParamGradient]]
  type NetParams = Map[HiddenLayer, Seq[LayerParam]]
}
