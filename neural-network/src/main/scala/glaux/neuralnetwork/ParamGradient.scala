package glaux.neuralnetwork

import glaux.linearalgebra.Tensor

case class ParamGradient(param: LayerParam, value: Tensor)

