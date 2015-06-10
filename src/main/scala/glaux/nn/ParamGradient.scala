package glaux.nn

import glaux.linalg.Tensor

case class ParamGradient(param: LayerParam, value: Tensor)

