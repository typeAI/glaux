package glaux.nn.layers

import glaux.nn.Dimension.ThreeD
import glaux.nn.{Dimension, Matrix, RowVector, Vol3D}
import org.nd4j.linalg.factory.Nd4j
import org.specs2.mutable.Specification

class FullyConnectedSpec extends Specification {

  val fc = FullyConnected(Matrix(3, 2, Seq(1d, 2d, 3d, 4d, 5d, 6d)), RowVector(100, 200))
  val input = RowVector(1, 1, 1)
  "forward correctly" >> {

    val output = fc.forward(input)

    output must_== RowVector(106, 215)
  }

  "backward returns correctly" >> {
    val (inGraident, paramGradients) = fc.backward(input, RowVector(3, 4))
    "the input gradient" >> {
      inGraident must_== RowVector(19, 26, 33)
    }

    "the filter gradient" >> {
      val fg = paramGradients.find(_.param.id == "filter").get
      fg.gradientValue must_== Matrix(3, 2, Seq(3,3,3,4,4,4))
    }

    "the bias gradient" >> {
      val bg = paramGradients.find(_.param.id == "bias").get
      bg.gradientValue must_== RowVector(3, 4)
    }
  }


}
