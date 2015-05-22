package glaux.nn

case class ParameterGradient(params: Vol, gradientValue: Vol, l1DM: DecayMul, l2DM: DecayMul)