package glaux.nn

case class Gradient(params: Vol, gradientValue: Vol, l1DecayMul: Float, l2DecayMul: Float)