package glaux.nn.layers

import glaux.nn.{DimensionArray, Vol1D}
import org.specs2.mutable.Specification

class RegressSpec extends Specification {
  "Dimension" >> {
    Regression(2).inDimension must_== DimensionArray(2)
    Regression(2).outDimension must_== DimensionArray(2)
  }
  "forward" >> {
    Regression(2).forward(Vol1D(1,2)) must_== Vol1D(1,2)
  }
  "Loss" >> {
    val rl = Regression(3)
    val (loss, inGradient) = rl.loss(Vol1D(2, 3, 4), Vol1D(1, 2, 3))
    loss must_== 1.5
    inGradient must_== Vol1D(-1, -1, -1)
  }
}
