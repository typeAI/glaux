package glaux.nn.layers

import glaux.linalg.Dimension.Row
import glaux.linalg.{RowVector, Matrix}
import org.specs2.mutable.Specification

class SoftmaxSpec extends Specification {
  val layer = Softmax(Row(3))
  val input = RowVector(1, 3, 2)
  val expectedOutput = RowVector(0.09003057, 0.66524096, 0.24472847)
  "forward correctly" >> {
    val output = layer.forward(input)
    output must_== expectedOutput
  }

  "loss returns correctly" >> {
    val (loss, inGradient) = layer.loss(RowVector(1, 0, 0), expectedOutput)
    loss must beCloseTo(2.40760596 within 6.significantFigures)
    inGradient must_== RowVector(-0.909969427, 0.665240956, 0.24472847)
  }

  "loss returns correctly for incorrect guess" >> {
    val (loss, inGradient) = layer.loss(RowVector(0, 1, 0), expectedOutput)
    loss must beCloseTo(0.40760596 within 6.significantFigures)
    inGradient must_== RowVector(0.09003057, -0.33475904, 0.24472847)
  }
}


/** *
  *
  *
  * var l = new convnetjs.SoftmaxLayer({in_sx:1, in_sy: 1, in_depth: 3})
  * var input = new convnetjs.Vol([1,3,2])
  * l.forward(input)
  * l.loss(1)
  * l.loss(2)
  */


