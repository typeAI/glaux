package glaux

package object nn {
  type DecayMultiplier = Double
  type Loss = Double
  type DataFlow = Vector[LayerData[_]]
  type NetParamGradients = Map[HiddenLayer, Seq[ParamGradient]]
  type NetParams = Map[HiddenLayer, Seq[LayerParam]]
}
