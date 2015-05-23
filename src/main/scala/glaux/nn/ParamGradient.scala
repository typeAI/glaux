package glaux.nn

case class ParamGradient(param: Param, gradientValue: Vol, l1DM: DecayMul, l2DM: DecayMul)