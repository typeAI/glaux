package glaux

package object nn {
  type Input = Vol
  type Output = Vol
  type OutGradient = Vol
  type InGradient = Vol
  type DecayMul = Float
  type Loss = Float
  type DataFlow = Vector[LayerData]
}
