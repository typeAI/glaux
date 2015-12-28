package glaux.neuralnetwork.layers

import glaux.linearalgebra.Dimension.{TwoD, Row, ThreeD}
import glaux.linearalgebra.{Vol, Matrix, RowVector}
import glaux.neuralnetwork.{HiddenLayer, RegularizationSetting, Gradient}
import org.specs2.mutable.Specification

class FullyConnectedSpec extends Specification {

  "RowVector as input" >> {
    val fc = FullyConnected(Matrix(3, 2, Seq(1d, 2d, 3d, 4d, 5d, 6d)), RowVector(100, 200))
    val input = RowVector(1, 1, 1)
    "forward correctly" >> {
      val output = fc.forward(input)
      output must_== RowVector(106, 215)
    }

    "backward returns correctly" >> {
      val (inGraident, paramGradients) = fc.backward(input, Gradient(RowVector(106, 215), RowVector(3, 4)))
      "the input gradient" >> {
        inGraident must_== RowVector(19, 26, 33)
      }

      "the filter gradient" >> {
        val fg = paramGradients.find(_.param.id == "filter").get
        fg.value must_== Matrix(3, 2, Seq(3, 3, 3, 4, 4, 4))
      }

      "the bias gradient" >> {
        val bg = paramGradients.find(_.param.id == "bias").get
        bg.value must_== RowVector(3, 4)
      }
    }

  }

  "Vol as input" >> {
    val fc = FullyConnected[Vol](ThreeD(2, 2, 2), 2).copy(filter = Matrix.fill(TwoD(8, 2), 1))
    val input = Vol(2, 2, 2, Seq(1, 2, 3, 4, 5, 6, 7, 8))
    "forward correctly" >> {
      val output = fc.forward(input)
      output must_== RowVector(36, 36)
    }

    "backward returns correctly" >> {
      val (inGraident, paramGradients) = fc.backward(input, Gradient(RowVector(36, 36), RowVector(1, 2)))
      "the input gradient" >> {
        inGraident must_== Vol.fill(ThreeD(2, 2, 2), 3)
      }
    }
  }

  "initialize with random normalized filters" >> {
    val created = FullyConnected(100, 100)
    val weights = created.filter.seqView
    val within3Std = weights.count(w ⇒ Math.abs(w) < 0.3)
    (within3Std.toDouble / weights.length) must be_>(0.9)

  }

}
