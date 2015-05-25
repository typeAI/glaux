package glaux.nn

case class ParamGradient(param: LayerParam, gradientValue: Vol, l1DM: DecayMul, l2DM: DecayMul)