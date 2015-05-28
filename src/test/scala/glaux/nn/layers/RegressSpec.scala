package glaux.nn.layers

import glaux.nn.{Dimension, RowVector}
import org.specs2.mutable.Specification

class RegressSpec extends Specification {
  "RegressionLayer" >> {
    "Dimension" >> {
      Regression(2).inDimension must_== Dimension.Row(2)
      Regression(2).outDimension must_== Dimension.Row(2)
    }

    "forward" >> {
      Regression(2).forward(RowVector(1, 2)) must_== RowVector(1, 2)
    }

    "Loss" >> {
      val rl = Regression(3)
      val (loss, inGradient) = rl.loss(RowVector(2, 3, 4), RowVector(1, 2, 3))
      loss must_== 1.5
      inGradient must_== RowVector(-1, -1, -1)
    }
  }
}
