package glaux.nn.layers

import glaux.nn.{Gradient, ParamGradient, Dimension, RowVector}
import org.specs2.mutable.Specification

class ReluSpec extends Specification {

  val rl = Relu[RowVector](Dimension.Row(3))
  val input = RowVector(1, -2, 3)

  "forward correctly" >> {
    val output = rl.forward(input)
    output must_== RowVector(1, 0, 3)
  }

  "backward returns correctly" >> {
    val (inGraident, paramGradients: Seq[_]) = rl.backward(input, Gradient(RowVector(1, 0, 3), RowVector(3, 4, 6)))
    "the input gradient" >> {
      inGraident must_== RowVector(3, 0, 6)
    }

    "no parameters" >> {
      paramGradients.isEmpty must beTrue
    }

  }


}
