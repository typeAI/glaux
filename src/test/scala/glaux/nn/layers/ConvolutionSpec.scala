package glaux.nn.layers

import glaux.linalg._
import Dimension._
import glaux.nn.Rectangle
import org.specs2.mutable.Specification
import Convolution._
class ConvolutionSpec extends Specification {

  "math" >> {

    "convolute" >> {
      val v = Vol(2, 1, 3, Seq(1, 2, 3, 4, 5, 6) )
      val t4 = Tensor4(2, 1, 3, 2,  Seq(1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6) )
      val result = convolute(v, t4)
      result.dimension === ThreeD(1, 1, 2)
      val expected = Range(1, 7).map(i => i * i).sum
      result.seqView === Seq(expected, expected)
    }
  }

  "forward" >> {

    "calculate output correctly" >> {
      val inputDim = ThreeD(4, 4, 2)
      val layer = Convolution(3, Rectangle(3, 3), inputDim, 1, true)
      val input = Vol(inputDim, Seq(1, 1, 2, 1, 5, 1, 3, 1, 9, 1, 1, 1, 5, 1, 1, 1, 4, 1, 11, 1, 7, 1, 3, 1, 6, 1, 1, 8, 1, 9, 1, 1))
      val output = layer.forward(input, false)
      output === Vol(4, 4, 3, Seq(5, 1.5, 2.7, 4.9, -0.2, 2.9, 5.1, -0.3, 1.5, 3.2, -1.4, 1, 4.37, 2.1, -1.3, 5.87, 1.8, 1.2, 9.17, 0.5, 1.9, 5.6, -0.2, -0.8, 7.27, -0.8, 1.1, 11.27, -5.8, -3.8, 5.97, -1.4, -3.0, 4, -1.8, -2.4, 4.17, -0.3, -2.8, 7.27, 0.6, -6.1, 9.27, 0.5, -7.0, 4.6, 2.7, -3.4))
    }
  }

}

/* convnetjs test code
  var l = new convnetjs.ConvLayer({in_sx: 4, in_sy: 4, in_depth: 2, filters: 3, sx: 3, sy: 3, stride: 1, pad: 1});
  var input = new convnetjs.Vol(4, 4, 2, 1)
  input.w = [1, 1, 2, 1, 5, 1, 3, 1, 9, 1, 1, 1, 5, 1, 1, 1, 4, 1, 11, 1, 7, 1, 3, 1, 6, 1, 1, 8, 1, 9, 1, 1]
  l.filters[0].w = [0.3, -0.4, 0.1, 0.2, -0.1, -0.03, -0.1, 0.2, 0.3, 0.4, 0.1, 0.3, 0.3, 0.1, 0.4, 0.2, -0.1, 0.1]
  l.filters[1].w = [0, -0.2, 0.2, 0.2, -0.2, 0.1, -0.1, 0.3, -0.2, -0.3, -0.1, 0.3, -0.1, -0.1, 0.2, 0.1, 0.3, -0.3]
  l.filters[2].w = [-0.4, -0.5, -0.2, -0.1, -0.2, 0.1, -0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.3, 0.1, 0.3, -0.1, -0.1, 0.1]

  var output = l.forward(input);
  output.w
  // [5, 1.5, 2.7, 4.9, -0.2, 2.9, 5.1, -0.3, 1.5, 3.2, -1.4, 1, 4.37, 2.1, -1.3, 5.87, 1.8, 1.2, 9.17, 0.5, 1.9, 5.6, -0.2, -0.8, 7.27, -0.8, 1.1, 11.27, -5.8, -3.8, 5.97, -1.4, -3.0, 4, -1.8, -2.4, 4.17, -0.3, -2.8, 7.27, 0.6, -6.1, 9.27, 0.5, -7.0, 4.6, 2.7, -3.4]

*/
