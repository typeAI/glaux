package glaux.nn.layers

import glaux.nn.Dimension.ThreeD
import glaux.nn.{Dimension, Matrix, RowVector, Vol3D}
import org.nd4j.linalg.factory.Nd4j
import org.specs2.mutable.Specification

class FullyConnectedSpec extends Specification {

  "forward correctly" >> {
    val fc = FullyConnected(Matrix(Dimension.TwoD(3, 2), Seq(1d, 2d, 3d, 4d, 5d, 6d)), RowVector(100, 200))

    val output = fc.forward(RowVector(1, 1, 1))

    output must_== RowVector(106, 215)
  }


}
